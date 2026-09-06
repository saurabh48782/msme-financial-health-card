"""Portfolio analytics and model metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, status

from src.api.deps import ConfigDep, PortfolioDep
from src.schemas.portfolio import MetricsReport, PortfolioSummary, SliceMetrics
from src.utils import read_json
from src.utils.config import model_artifact_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["portfolio"])

TRAINING_REPORT = "training_report.json"


@router.get(
    "/portfolio/summary",
    response_model=PortfolioSummary,
    summary="Distributions, cohort cuts and approval-rate lift by segment",
)
async def summary(portfolio: PortfolioDep) -> PortfolioSummary:
    return await portfolio.summary()


def load_training_report(config: dict[str, Any]) -> dict[str, Any] | None:
    path: Path = model_artifact_dir(config) / TRAINING_REPORT
    if not path.is_file():
        return None
    return read_json(path)  # type: ignore[no-any-return]


@router.get(
    "/models/metrics",
    response_model=MetricsReport,
    summary="Holdout metrics, calibration curve and fairness slices for the live model",
)
async def metrics(config: ConfigDep) -> MetricsReport:
    report = load_training_report(config)
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No training report available — run python -m src.model.model_orchestration",
        )
    fairness: list[SliceMetrics] = [
        SliceMetrics(slice=column, level=level, rows=int(values.get("rows", 0)), metrics=values)
        for column, levels in report.get("fairness", {}).get("slices", {}).items()
        for level, values in levels.items()
    ]
    return MetricsReport(
        model_version=None,
        policy_version=str(config["policy_version"]),
        rubric_version=str(config["rubric_version"]),
        pd_metrics=report.get("pd_metrics", {}),
        eligibility_metrics=report.get("eligibility_metrics", {}),
        pillar_metrics=report.get("pillar_metrics", {}),
        fhs_metrics=report.get("fhs_metrics", {}),
        calibration_curve=report.get("reliability_curve", []),
        fairness=fairness,
        thresholds=config["evaluation"]["thresholds"],
    )


@router.get("/models/versions", summary="Registry versions and their aliases")
async def versions(config: ConfigDep) -> list[dict[str, Any]]:
    """Best-effort: a tracking-server outage returns an empty list, not a 500.

    The registry is metadata. The API scores from a local artifact and must not
    start failing requests because MLflow is down.
    """
    try:
        import mlflow

        mlflow_config = config["mlflow_config"]
        mlflow.set_tracking_uri(str(mlflow_config["tracking_uri"]))
        client = mlflow.MlflowClient()
        name = str(mlflow_config["registered_model_name"])
        found = []
        for version in client.search_model_versions(f"name='{name}'"):
            aliases = list(getattr(version, "aliases", []) or [])
            found.append(
                {
                    "name": name,
                    "version": str(version.version),
                    "aliases": aliases,
                    "run_id": version.run_id,
                    "created_at": str(version.creation_timestamp),
                }
            )
        return sorted(found, key=lambda v: int(v["version"]), reverse=True)
    except Exception as error:  # noqa: BLE001 - registry is optional metadata
        logger.warning("Model registry unavailable", error=str(error))
        return []
