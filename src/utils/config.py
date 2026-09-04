"""``params.yaml`` loading, ``${ENV}`` resolution, derived paths and validation.

Three rules the rest of the codebase depends on:

1. ``params.yaml`` is the only place a tunable lives. No module re-reads YAML and
   no module re-derives a path - the derived path helpers below are the API.
2. ``${VAR}`` placeholders are resolved at load time. An unset **or empty** var
   raises.
3. ``validate_config()`` runs in the API lifespan and in every orchestration
   entrypoint, so a misconfigured deployment fails at boot rather than on the
   first request.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import yaml

from src.utils.custom_error import CustomError

ROOT_DIR = Path(__file__).resolve().parents[2]
PARAMS_PATH = ROOT_DIR / "params.yaml"

# ${VAR}, ${VAR:-}, ${VAR:-default}
_PLACEHOLDER = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}")

_REQUIRED_SECTIONS = (
    "data_paths",
    "schema",
    "preprocessing",
    "pillars",
    "grades",
    "model_params",
    "policy",
    "anomaly",
    "explainability",
    "mlflow_config",
    "api",
    "evaluation",
)
_WEIGHT_TOLERANCE = 1e-6


def _resolve_placeholders(value: str) -> str:
    """Substitute every ``${VAR}`` in ``value``, honouring ``:-`` defaults."""

    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        default = match.group("default")
        env = os.environ.get(name)
        if env:
            return env
        if default is not None:
            return default
        raise CustomError(
            f"Required environment variable {name!r} referenced by params.yaml is unset or empty"
        )

    return _PLACEHOLDER.sub(replace, value)


def _resolve(node: Any) -> Any:
    """Walk the parsed YAML tree resolving placeholders in every string leaf."""
    if isinstance(node, dict):
        return {key: _resolve(child) for key, child in node.items()}
    if isinstance(node, list):
        return [_resolve(child) for child in node]
    if isinstance(node, str):
        return _resolve_placeholders(node)
    return node


@lru_cache(maxsize=1)
def load_config(path: str | Path = PARAMS_PATH) -> dict[str, Any]:
    """Parse ``params.yaml`` once per process, with ``${ENV}`` resolved."""
    config_path = Path(path)
    if not config_path.is_file():
        raise CustomError(f"Config file not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:  # pragma: no cover - malformed YAML is a dev error
        raise CustomError(error) from error
    if not isinstance(raw, dict):
        raise CustomError(f"Config file {config_path} did not parse to a mapping")
    return cast(dict[str, Any], _resolve(raw))


# Derived paths. Every path in the codebase comes from here.
def _path(key: str, config: dict[str, Any] | None = None) -> Path:
    cfg_ = config if config is not None else load_config()
    return ROOT_DIR / str(cfg_["data_paths"][key])


def raw_csv_path(config: dict[str, Any] | None = None) -> Path:
    return _path("raw_csv", config)


def processed_features_path(config: dict[str, Any] | None = None) -> Path:
    return _path("processed_features", config)


def portfolio_scores_path(config: dict[str, Any] | None = None) -> Path:
    return _path("portfolio_scores", config)


def model_artifact_dir(config: dict[str, Any] | None = None) -> Path:
    return _path("model_artifact_dir", config)


def eval_reports_dir(config: dict[str, Any] | None = None) -> Path:
    return _path("eval_reports_dir", config)


DATA_DIR = ROOT_DIR / "data"


# Validation
def _validate_pillars(config: dict[str, Any], errors: list[str]) -> None:
    pillars = config["pillars"]
    fhs_weights = pillars.get("fhs_weights", {})
    total = sum(float(v) for v in fhs_weights.values())
    if abs(total - 1.0) > _WEIGHT_TOLERANCE:
        errors.append(f"pillars.fhs_weights must sum to 1.0, got {total!r}")

    standing = config.get("categorical_standing", {})
    for name, spec in pillars.items():
        if name == "fhs_weights":
            continue
        if name not in fhs_weights:
            errors.append(f"pillar {name!r} has no entry in pillars.fhs_weights")
        drivers = spec.get("drivers", {})
        if not drivers:
            errors.append(f"pillar {name!r} declares no drivers")
            continue
        driver_total = sum(float(d["weight"]) for d in drivers.values())
        if abs(driver_total - 1.0) > _WEIGHT_TOLERANCE:
            errors.append(f"pillar {name!r} driver weights must sum to 1.0, got {driver_total!r}")
        band = spec.get("band")
        if not (isinstance(band, list) and len(band) == 2):
            errors.append(f"pillar {name!r} must declare band: [lo, hi]")
        else:
            lo, hi = float(band[0]), float(band[1])
            # The band is the whole range a pillar can report, so a band inside
            # 0-100 is what makes the score unclampable and the decomposition
            # exact — see src.scoring.pillars.
            if not 0.0 <= lo < hi <= 100.0:
                errors.append(
                    f"pillar {name!r} band must satisfy 0 <= lo < hi <= 100, got {band!r}"
                )
        for driver_name, driver in drivers.items():
            where = f"pillars.{name}.drivers.{driver_name}"
            if float(driver["worst"]) == float(driver["best"]):
                errors.append(f"{where} has a zero-width ramp (worst == best)")
            source = str(driver["source"])
            if source.endswith("_standing") and source not in standing:
                errors.append(f"{where}.source {source!r} has no categorical_standing map")


def _validate_policy(config: dict[str, Any], errors: list[str]) -> None:
    policy = config["policy"]
    bands = policy["pd_bands"]
    low_max, medium_max = float(bands["low_max"]), float(bands["medium_max"])
    if not 0.0 < low_max < medium_max < 1.0:
        errors.append(
            f"policy.pd_bands must satisfy 0 < low_max < medium_max < 1, "
            f"got low_max={low_max}, medium_max={medium_max}"
        )
    multiples = policy["limits"]["band_multiple"]
    for band, value in multiples.items():
        if not 0.0 <= float(value) < 1.0:
            errors.append(f"policy.limits.band_multiple.{band} must be in [0, 1), got {value!r}")
    for band in ("low", "medium", "high"):
        if band not in multiples:
            errors.append(f"policy.limits.band_multiple is missing band {band!r}")
    if float(policy["limits"]["min_limit_inr"]) > float(policy["limits"]["max_limit_inr"]):
        errors.append("policy.limits.min_limit_inr exceeds max_limit_inr")


def _validate_grades(config: dict[str, Any], errors: list[str]) -> None:
    mins = [float(g["min"]) for g in config["grades"]]
    if mins != sorted(mins, reverse=True):
        errors.append("grades must be listed in descending order of `min`")
    if mins and mins[-1] != 0.0:
        errors.append("the last grade band must have min: 0.0 so every score lands somewhere")


def _validate_schema(config: dict[str, Any], errors: list[str]) -> None:
    schema = config["schema"]
    features = schema["feature_columns"]
    labels = set(schema["label_columns"])
    if len(features) != len(set(features)):
        errors.append("schema.feature_columns contains duplicates")
    leaked = labels.intersection(features)
    if leaked:
        errors.append(f"LEAKAGE: label columns present in feature_columns: {sorted(leaked)}")
    if schema["id_column"] in features:
        errors.append("schema.id_column must not be a feature")
    missing = set(schema["pillar_label_map"]) - set(config["pillars"]["fhs_weights"])
    if missing:
        errors.append(f"schema.pillar_label_map names unknown pillars: {sorted(missing)}")


def validate_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Assert the config is internally consistent, or raise with *every* problem.

    Collecting all errors before raising matters: a misconfigured deployment is
    fixed in one pass instead of one boot per typo.
    """
    cfg_ = config if config is not None else load_config()
    errors: list[str] = [
        f"missing required section {section!r}"
        for section in _REQUIRED_SECTIONS
        if section not in cfg_
    ]
    if errors:
        raise CustomError("Invalid params.yaml: " + "; ".join(errors))

    _validate_schema(cfg_, errors)
    _validate_pillars(cfg_, errors)
    _validate_policy(cfg_, errors)
    _validate_grades(cfg_, errors)

    if not str(cfg_["mlflow_config"].get("tracking_uri", "")):
        errors.append("mlflow_config.tracking_uri is empty")
    if int(cfg_["api"]["max_batch_size"]) < 1:
        errors.append("api.max_batch_size must be at least 1")

    thresholds = cfg_["evaluation"]["thresholds"]
    if not 0.0 < float(thresholds["eligibility_auc_min"]) <= 1.0:
        errors.append("evaluation.thresholds.eligibility_auc_min must be in (0, 1]")
    # Named explicitly so a params.yaml still carrying the old ``pillar_r2_min``
    # fails at boot rather than silently ungating the rubric.
    if "pillar_adj_r2_min" not in thresholds:
        errors.append(
            "evaluation.thresholds.pillar_adj_r2_min is missing "
            "(the gate moved from plain R² to adjusted R²)"
        )
    elif not 0.0 < float(thresholds["pillar_adj_r2_min"]) <= 1.0:
        errors.append("evaluation.thresholds.pillar_adj_r2_min must be in (0, 1]")

    if errors:
        raise CustomError("Invalid params.yaml: " + "; ".join(errors))
    return cfg_
