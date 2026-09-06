"""The registry artifact: one object that can score a raw MSME record.

The MLflow model is deliberately *not* a bare XGBoost booster. A booster alone
cannot produce a Health Card - it has no winsorisation limits, no rubric, no
policy engine. Registering only the booster would mean the serving
code and the training code each own half the pipeline, and the halves would drift.

So the registered ``pyfunc`` carries the whole :class:`ScoringBundle` and calls the
same :mod:`src.scoring.scoring_orchestration` path the API does. ``code_paths=["src"]``
ships the source alongside it, so a card scored from the registry a year from now
uses the rubric that was in force when the model was trained.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from mlflow.pyfunc import PythonModel, PythonModelContext

from src.scoring.scoring_orchestration import ScoringBundle, score_many
from src.utils.config import load_config, model_artifact_dir
from src.utils.custom_error import CustomError
from src.utils.logger import get_logger

logger = get_logger(__name__)

BUNDLE_FILENAME = "scoring_bundle.joblib"
CONFIG_FILENAME = "params_snapshot.json"
ARTIFACT_KEY_BUNDLE = "scoring_bundle"
ARTIFACT_KEY_CONFIG = "params_snapshot"


def save_bundle(bundle: ScoringBundle, directory: str | Path) -> Path:
    """Persist a bundle, minus the SHAP explainer (rebuilt on load).

    The explainer is derived state — pickling it would double the artifact size and
    pin a shap version into the registry for no benefit.
    """
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    path = target / BUNDLE_FILENAME
    shap_explainer = bundle.shap_explainer
    bundle.shap_explainer = None
    try:
        joblib.dump(bundle, path, compress=3)
    finally:
        bundle.shap_explainer = shap_explainer
    logger.info("Scoring bundle saved", path=str(path), size_bytes=path.stat().st_size)
    return path


def load_bundle(
    path: str | Path, config: dict[str, Any] | None = None, *, with_shap: bool = True
) -> ScoringBundle:
    """Load a bundle and rebuild the SHAP explainer."""
    source = Path(path)
    if source.is_dir():
        source = source / BUNDLE_FILENAME
    if not source.is_file():
        raise CustomError(f"scoring bundle not found: {source}")
    bundle: ScoringBundle = joblib.load(source)
    if with_shap and bundle.pd_model is not None:
        from src.scoring.explainer import ShapExplainer

        bundle.shap_explainer = ShapExplainer(bundle.pd_model, config)
    logger.info(
        "Scoring bundle loaded",
        path=str(source),
        model_version=bundle.model_version,
        has_ml=bundle.has_ml,
    )
    return bundle


# mlflow ships no type information, so PythonModel resolves to Any and mypy
# cannot verify the subclass. The base class is still real at runtime.
class HealthCardModelWrapper(PythonModel):  # type: ignore[misc]
    """MLflow ``pyfunc`` wrapping the complete Health Card pipeline.

    ``predict`` accepts raw MSME records — dataset-named columns, exactly what the
    ingestion APIs receive — and returns one row per firm with the card summary,
    plus the full card as JSON when explanations are requested.
    """

    def __init__(self, bundle: ScoringBundle | None = None) -> None:
        self._bundle = bundle
        self._config: dict[str, Any] | None = None

    def load_context(self, context: PythonModelContext) -> None:
        self._bundle = load_bundle(context.artifacts[ARTIFACT_KEY_BUNDLE])
        snapshot = context.artifacts.get(ARTIFACT_KEY_CONFIG)
        if snapshot:
            self._config = json.loads(Path(snapshot).read_text(encoding="utf-8"))
        else:  # pragma: no cover - only when logged without a snapshot
            self._config = load_config()

    @property
    def config(self) -> dict[str, Any]:
        if self._config is None:
            self._config = load_config()
        return self._config

    def predict(
        self,
        context: PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        _ = context
        if self._bundle is None:  # pragma: no cover - defensive
            raise CustomError("HealthCardModelWrapper used before load_context")
        options = params or {}
        include_explanation = bool(options.get("include_explanation", False))
        frame = pd.DataFrame(model_input).copy()
        cards = score_many(
            frame, self._bundle, self.config, include_explanation=include_explanation
        )
        records = []
        for card in cards:
            row = dict(card.summary)
            row["grade"] = card.grade
            row["probability_of_default"] = card.credit.probability_of_default
            row["tenor_months"] = card.credit.tenor_months
            row["indicative_rate_pct"] = card.credit.indicative_rate_pct
            row["flag_count"] = len(card.risk_flags)
            row["thin_file"] = card.thin_file
            row["policy_version"] = card.policy_version
            row["feature_snapshot_hash"] = card.feature_snapshot_hash
            if include_explanation:
                row["card_json"] = card.model_dump_json()
            records.append(row)
        return pd.DataFrame.from_records(records)


class ModelLoader:
    """Resolve a scoring bundle from the local artifact directory.

    Deliberately local-only: the training run writes ``./model_artifact``, and the
    API loads from there. So the API never makes a network call to the tracking
    server — not on the request path and not at boot — and an MLflow outage cannot
    take scoring down.

    Pulling a specific registry version at boot is the natural production
    addition; see ``docs/BEYOND_SCOPE.md``.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config if config is not None else load_config()

    def local_path(self) -> Path:
        return model_artifact_dir(self.config)

    def from_local(self, *, with_shap: bool = True) -> ScoringBundle | None:
        path = self.local_path() / BUNDLE_FILENAME
        if not path.is_file():
            logger.warning("No local model artifact", path=str(path))
            return None
        return load_bundle(path, self.config, with_shap=with_shap)

    def load(self, *, with_shap: bool = True) -> ScoringBundle:
        """Best available bundle. Never returns None — falls back to an ML-free bundle.

        An ML-free bundle still produces a complete card via the rubric and the
        fitted PD fallback, which is what lets the dashboard come up on a fresh
        checkout instead of failing closed with a blank page.
        """
        local = self.from_local(with_shap=with_shap)
        if local is not None:
            return local
        logger.warning(
            "Scoring without ML models - rubric and fitted PD fallback only",
            hint="run python -m src.model.model_orchestration",
        )
        return self._rubric_only_bundle()

    def _rubric_only_bundle(self) -> ScoringBundle:
        from src.data.data_orchestration import WINSOR_LIMITS_FILENAME
        from src.utils import read_json
        from src.utils.config import processed_features_path

        limits: dict[str, dict[str, float]] = {}
        limits_path = processed_features_path(self.config).parent / WINSOR_LIMITS_FILENAME
        if limits_path.is_file():
            limits = read_json(limits_path)
        return ScoringBundle(winsor_limits=limits)
