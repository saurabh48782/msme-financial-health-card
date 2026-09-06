"""Train -> evaluate -> log to MLflow -> register -> alias staging.

Run as ``python -m src.model.model_orchestration``.

One parent run per training job with a nested run per head, because the PD
regressor and the eligibility classifier are separately promotable ideas that
happen to ship together. Every artifact needed to defend a decision is logged:
the fairness table, the reliability curve, the feature allowlist, the pillar
weights, and a snapshot of ``params.yaml`` as it was at training time.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # no display on a CI runner or in a container
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.data_orchestration import WINSOR_LIMITS_FILENAME  # noqa: E402
from src.model.components import (  # noqa: E402
    build_feature_matrix,
    evaluate_classification,
    evaluate_probability,
    feature_importance,
    model_feature_names,
    predict_eligibility,
    predict_pd,
    stratified_split,
    train_eligibility_classifier,
    train_pd_regressor,
)
from src.model.fairness import slice_report, to_frame, worst_deviation  # noqa: E402
from src.model.reliability import reliability_curve  # noqa: E402
from src.model.utilities import (  # noqa: E402
    ARTIFACT_KEY_BUNDLE,
    ARTIFACT_KEY_CONFIG,
    BUNDLE_FILENAME,
    HealthCardModelWrapper,
    save_bundle,
)
from src.scoring.health_score import compute_health_score  # noqa: E402
from src.scoring.pillar_calibration import (  # noqa: E402
    adjusted_r_squared,
    fails_floor,
    r_squared,
)
from src.scoring.pillars import compute_pillars, pillar_keys  # noqa: E402
from src.scoring.scoring_orchestration import ScoringBundle  # noqa: E402
from src.utils import read_dataframe, read_json, write_json  # noqa: E402
from src.utils.config import (  # noqa: E402
    ROOT_DIR,
    load_config,
    model_artifact_dir,
    processed_features_path,
    validate_config,
)
from src.utils.custom_error import CustomError  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.tracing import trace_stage  # noqa: E402

logger = get_logger(__name__)


def _log_flat_metrics(metrics: dict[str, Any], prefix: str = "") -> None:
    """MLflow takes scalars only; flatten one level and drop anything else."""
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            mlflow.log_metric(f"{prefix}{key}", float(value))


def _reliability_plot(curve: list[dict[str, float]], path: Path) -> None:
    figure, axes = plt.subplots(figsize=(5, 5))
    predicted = [row["mean_predicted"] for row in curve]
    actual = [row["mean_actual"] for row in curve]
    axes.plot(
        [0, max(predicted + actual + [0.01])],
        [0, max(predicted + actual + [0.01])],
        linestyle="--",
        linewidth=1,
        label="perfect calibration",
    )
    axes.plot(predicted, actual, marker="o", label="model")
    axes.set_xlabel("Mean predicted PD")
    axes.set_ylabel("Mean realised PD")
    axes.set_title("PD reliability")
    axes.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _shap_summary_plot(
    bundle: ScoringBundle, sample: pd.DataFrame, path: Path, config: dict[str, Any]
) -> None:
    """Global SHAP beeswarm. Best-effort — a plotting failure must not fail training."""
    try:
        import shap

        if bundle.pd_model is None:
            return
        matrix, _, _ = build_feature_matrix(sample, config, features=bundle.pd_model.features)
        explainer = shap.TreeExplainer(bundle.pd_model.estimator)
        values = explainer.shap_values(matrix)
        plt.figure()
        shap.summary_plot(
            values,
            matrix,
            max_display=int(config["explainability"]["shap_max_display"]),
            show=False,
        )
        plt.tight_layout()
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close("all")
    except Exception as error:  # noqa: BLE001 - a plot is not worth failing a run
        logger.warning("SHAP summary plot skipped", error=str(error))


def _export_requirements(config: dict[str, Any]) -> list[str] | None:
    """Read the pip-style requirements ``scripts/train.sh`` exports from uv.lock.

    MLflow's ``log_model`` wants a pip requirement list and cannot read ``uv.lock``,
    so the lockfile stays the source of truth and this is a rendering of it.
    """
    path = ROOT_DIR / str(config["mlflow_config"].get("requirements_file", ""))
    if not path.is_file():
        logger.warning(
            "No exported requirements file — MLflow will infer them",
            expected=str(path),
            hint="uv export --no-dev --no-hashes --format requirements-txt",
        )
        return None
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith(("#", "-e", "--"))
    ]
    return lines or None


def train(
    frame: pd.DataFrame, config: dict[str, Any] | None = None
) -> tuple[ScoringBundle, dict[str, Any]]:
    """Train both heads and evaluate on the holdout."""
    cfg = config if config is not None else load_config()
    train_frame, test_frame = stratified_split(frame, cfg)

    with trace_stage("train.pd", rows=len(train_frame)):
        pd_model = train_pd_regressor(train_frame, cfg)
    with trace_stage("train.eligibility", rows=len(train_frame)):
        eligibility_model = train_eligibility_classifier(train_frame, cfg)

    # Reliability is measured on the holdout, never on training predictions: the
    # point of the curve is to expose overfit, which in-sample output hides.
    test_pd = predict_pd(pd_model, test_frame, cfg)

    test_eligibility = predict_eligibility(eligibility_model, test_frame, cfg)
    truth_eligible = test_frame["Credit_Eligible"].astype("string").eq("Yes").astype(int)

    pd_metrics = evaluate_probability(
        test_frame["Probability_of_Default"], test_pd, n_features=len(pd_model.features)
    )
    eligibility_metrics = evaluate_classification(truth_eligible, test_eligibility)

    curve = reliability_curve(
        test_frame["Probability_of_Default"].to_numpy(),
        test_pd,
        bins=int(cfg["evaluation"]["calibration_bins"]),
    )

    scored = test_frame.assign(predicted_pd=test_pd, predicted_eligibility=test_eligibility)
    fairness = slice_report(scored, config=cfg)

    # The rubric is evaluated on the same holdout so the model card carries one
    # consistent set of numbers rather than two runs' worth.
    pillars = compute_pillars(test_frame, cfg)
    fhs = compute_health_score(pillars, cfg)
    pillar_metrics: dict[str, dict[str, float]] = {}
    for key, column in cfg["schema"]["pillar_label_map"].items():
        if column not in test_frame.columns:
            continue
        actual = test_frame[column].to_numpy(dtype="float64")
        predicted = pillars[f"{key}_score"].to_numpy(dtype="float64")
        # A pillar costs one degree of freedom per driver: the weights it applies
        # here were fitted, so quoting its fit unadjusted would flatter a rubric
        # that simply has more drivers than its neighbours.
        drivers = len(cfg["pillars"][key]["drivers"])
        pillar_metrics[key] = {
            "r2": round(r_squared(actual, predicted), 6),
            "adj_r2": round(adjusted_r_squared(actual, predicted, drivers), 6),
            "n_features": float(drivers),
            "mae": round(float(np.abs(actual - predicted).mean()), 6),
        }
    fhs_actual = test_frame["Financial_Health_Score"].to_numpy(dtype="float64")
    fhs_predicted = fhs.to_numpy(dtype="float64")
    fhs_features = len(pillar_keys(cfg))
    fhs_metrics = {
        "r2": round(r_squared(fhs_actual, fhs_predicted), 6),
        "adj_r2": round(adjusted_r_squared(fhs_actual, fhs_predicted, fhs_features), 6),
        "n_features": float(fhs_features),
        "mae": round(float(np.abs(fhs_actual - fhs_predicted).mean()), 6),
    }

    limits_path = processed_features_path(cfg).parent / WINSOR_LIMITS_FILENAME
    bundle = ScoringBundle(
        winsor_limits=read_json(limits_path) if limits_path.is_file() else {},
        pd_model=pd_model,
        eligibility_model=eligibility_model,
    )

    report: dict[str, Any] = {
        "rows_train": len(train_frame),
        "rows_test": len(test_frame),
        "features": model_feature_names(frame, cfg),
        "pd_metrics": pd_metrics,
        "eligibility_metrics": eligibility_metrics,
        "pillar_metrics": pillar_metrics,
        "fhs_metrics": fhs_metrics,
        "reliability_curve": curve,
        "fairness": fairness,
        "pd_feature_importance": feature_importance(pd_model, top_n=25),
        "eligibility_feature_importance": feature_importance(eligibility_model, top_n=25),
    }
    logger.info(
        "Training evaluated",
        pd_adj_r2=pd_metrics["adj_r2"],
        pd_mae=pd_metrics["mae"],
        eligibility_auc=eligibility_metrics["roc_auc"],
        fhs_mae=fhs_metrics["mae"],
        worst_fairness_deviation=worst_deviation(fairness),
    )
    return bundle, report


def check_thresholds(report: dict[str, Any], config: dict[str, Any]) -> list[str]:
    """Compare the run against ``evaluation.thresholds``. Returns the breaches."""
    thresholds = config["evaluation"]["thresholds"]
    breaches: list[str] = []
    if report["pd_metrics"]["mae"] > float(thresholds["pd_mae_max"]):
        breaches.append(f"pd_mae {report['pd_metrics']['mae']:.4f} > {thresholds['pd_mae_max']}")
    if report["eligibility_metrics"]["roc_auc"] < float(thresholds["eligibility_auc_min"]):
        breaches.append(
            f"eligibility_auc {report['eligibility_metrics']['roc_auc']:.4f} "
            f"< {thresholds['eligibility_auc_min']}"
        )
    for key, metrics in report["pillar_metrics"].items():
        floor = float(thresholds["pillar_adj_r2_min"])
        if fails_floor(metrics["adj_r2"], floor):
            breaches.append(f"pillar {key} adj_r2 {metrics['adj_r2']:.4f} < {floor}")
    if report["fhs_metrics"]["mae"] > float(thresholds["fhs_mae_max"]):
        breaches.append(f"fhs_mae {report['fhs_metrics']['mae']:.4f} > {thresholds['fhs_mae_max']}")
    deviation = worst_deviation(report["fairness"], "pd_mae")
    # The fairness threshold is in FHS points; PD MAE deviation is a fraction, so
    # it is compared on the same 0-100 scale the threshold is written in.
    if deviation * 100.0 > float(thresholds["fairness_max_deviation_points"]):
        breaches.append(
            f"fairness pd_mae deviation {deviation * 100:.3f} pts "
            f"> {thresholds['fairness_max_deviation_points']}"
        )
    return breaches


def input_example(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """One real record for the MLflow signature.

    Signature inference calls ``predict`` on this frame, so it has to be a row the
    scorer can actually take: a synthetic all-zeros row makes XGBoost reject 0.0 as
    a category and the model registers without a signature. Rows carrying a
    structural null are skipped so every column types cleanly.
    """
    columns = [config["schema"]["id_column"], *config["schema"]["feature_columns"]]
    present = [c for c in dict.fromkeys(columns) if c in frame.columns]
    complete = frame[present].dropna()
    source = complete if not complete.empty else frame[present]
    return source.head(1).reset_index(drop=True)


def log_to_mlflow(
    bundle: ScoringBundle,
    report: dict[str, Any],
    config: dict[str, Any],
    frame: pd.DataFrame,
) -> str | None:
    """Log params, metrics and artifacts; register the model; alias it staging."""
    mlflow_config = config["mlflow_config"]
    mlflow.set_tracking_uri(str(mlflow_config["tracking_uri"]))
    mlflow.set_experiment(str(mlflow_config["experiment_name"]))

    registered_version: str | None = None
    with (
        mlflow.start_run(run_name="healthcard-training") as run,
        tempfile.TemporaryDirectory() as tmp,
    ):
        staging = Path(tmp)
        mlflow.log_params(
            {
                "rows_train": report["rows_train"],
                "rows_test": report["rows_test"],
                "n_features": len(report["features"]),
                "policy_version": config["policy_version"],
                "rubric_version": config["rubric_version"],
                "test_size": config["model_params"]["test_size"],
            }
        )
        mlflow.log_dict(config["model_params"], "hyperparameters.json")
        mlflow.log_dict({"features": report["features"]}, "feature_allowlist.json")
        mlflow.log_dict(
            {k: v["drivers"] for k, v in config["pillars"].items() if k != "fhs_weights"},
            "pillar_weights.json",
        )
        mlflow.log_dict(report["fairness"], "fairness.json")
        mlflow.log_dict(report["reliability_curve"], "reliability_curve.json")
        mlflow.log_artifact(str(ROOT_DIR / "params.yaml"), artifact_path="config")

        with mlflow.start_run(run_name="pd_regressor", nested=True):
            _log_flat_metrics(report["pd_metrics"], "test_")
        with mlflow.start_run(run_name="eligibility_classifier", nested=True):
            _log_flat_metrics(report["eligibility_metrics"], "test_")
        with mlflow.start_run(run_name="pillar_rubric", nested=True):
            for key, metrics in report["pillar_metrics"].items():
                _log_flat_metrics(metrics, f"pillar_{key}_")
            _log_flat_metrics(report["fhs_metrics"], "fhs_")

        _log_flat_metrics(report["pd_metrics"], "pd_")
        _log_flat_metrics(report["eligibility_metrics"], "eligibility_")
        _log_flat_metrics(report["fhs_metrics"], "fhs_")
        # Per-segment fairness as first-class metrics so they are chartable in the
        # MLflow UI and comparable across runs, not buried in a JSON artifact.
        for column, levels in report["fairness"]["slices"].items():
            for level, metrics in levels.items():
                safe = str(level).replace(" ", "_").replace("/", "_").replace("&", "and")
                for metric in ("pd_mae", "pd_bias", "approval_rate_predicted", "eligibility_auc"):
                    if metric in metrics:
                        mlflow.log_metric(f"slice_{column}_{safe}_{metric}", metrics[metric])

        fairness_csv = staging / "fairness.csv"
        to_frame(report["fairness"]).to_csv(fairness_csv, index=False)
        mlflow.log_artifact(str(fairness_csv), artifact_path="reports")

        reliability_png = staging / "reliability_curve.png"
        _reliability_plot(report["reliability_curve"], reliability_png)
        mlflow.log_artifact(str(reliability_png), artifact_path="plots")

        importance_json = staging / "feature_importance.json"
        write_json(
            {
                "pd": report["pd_feature_importance"],
                "eligibility": report["eligibility_feature_importance"],
            },
            importance_json,
        )
        mlflow.log_artifact(str(importance_json), artifact_path="reports")

        bundle_dir = staging / "bundle"
        save_bundle(bundle, bundle_dir)
        snapshot = staging / "params_snapshot.json"
        write_json(config, snapshot)

        example = input_example(frame, config)

        model_info = mlflow.pyfunc.log_model(
            name="healthcard_model",
            python_model=HealthCardModelWrapper(),
            artifacts={
                ARTIFACT_KEY_BUNDLE: str(bundle_dir / BUNDLE_FILENAME),
                ARTIFACT_KEY_CONFIG: str(snapshot),
            },
            code_paths=[str(ROOT_DIR / "src")],
            pip_requirements=_export_requirements(config),
            input_example=example,
            registered_model_name=str(mlflow_config["registered_model_name"]),
        )

        # Prefer the version MLflow reports for the model it just logged. Inferring
        # it from a search result is a hazard: search_model_versions makes no
        # ordering guarantee, so versions[0] is not reliably the newest.
        registered_version = getattr(model_info, "registered_model_version", None)
        if registered_version is None:
            client = mlflow.MlflowClient()
            versions = client.search_model_versions(
                f"name='{mlflow_config['registered_model_name']}' and run_id='{run.info.run_id}'"
            )
            registered_version = str(max(int(v.version) for v in versions)) if versions else None
        if registered_version is not None:
            registered_version = str(registered_version)
            logger.info("Model registered", version=registered_version)

        _shap_summary_plot(
            bundle, staging_sample(report, config), staging / "shap_summary.png", config
        )
        shap_png = staging / "shap_summary.png"
        if shap_png.is_file():
            mlflow.log_artifact(str(shap_png), artifact_path="plots")

    return registered_version


_SHAP_SAMPLE: dict[str, pd.DataFrame] = {}


def staging_sample(report: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    """The frame the beeswarm is drawn on, stashed by :func:`main`."""
    _ = report, config
    return _SHAP_SAMPLE.get("frame", pd.DataFrame())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None, help="processed features CSV")
    parser.add_argument("--no-mlflow", action="store_true", help="train and save locally only")
    parser.add_argument(
        "--fail-on-threshold",
        action="store_true",
        help="exit non-zero when a metric breaches evaluation.thresholds",
    )
    args = parser.parse_args(argv)

    logger.info("--- Starting model training ---")
    try:
        cfg = validate_config(load_config())
        source = args.input or processed_features_path(cfg)
        if not Path(source).is_file():
            raise CustomError(
                f"processed dataset not found at {source} — "
                f"run python -m src.data.data_orchestration first"
            )
        frame = read_dataframe(source)

        bundle, report = train(frame, cfg)
        _SHAP_SAMPLE["frame"] = frame.sample(
            n=min(int(cfg["explainability"]["shap_background_samples"]), len(frame)),
            random_state=int(cfg["model_params"]["random_state"]),
        )

        breaches = check_thresholds(report, cfg)
        if breaches:
            logger.warning("Metric thresholds breached", breaches=breaches)
        else:
            logger.info("All metric thresholds met")

        if not args.no_mlflow:
            version = log_to_mlflow(bundle, report, cfg, frame)
            bundle.model_version = str(version) if version is not None else None

        target = model_artifact_dir(cfg)
        save_bundle(bundle, target)
        write_json(report, target / "training_report.json")
        logger.info("Training artifacts written", path=str(target))

        print(
            json.dumps(
                {
                    "pd": report["pd_metrics"],
                    "eligibility": report["eligibility_metrics"],
                    "fhs": report["fhs_metrics"],
                    "pillars": report["pillar_metrics"],
                    "breaches": breaches,
                },
                indent=2,
            )
        )

        return 1 if (breaches and args.fail_on_threshold) else 0
    except Exception:
        logger.error("Model training failed", exc_info=True)
        return 1
    finally:
        logger.info("--- Model training Finished ---")


if __name__ == "__main__":
    raise SystemExit(main())
