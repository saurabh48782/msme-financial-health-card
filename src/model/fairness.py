"""Slice metrics by segment, location, industry and constitution.

This is the project's central claim under test. The dataset deliberately shows
thin-file firms are not inherently worse — mean FHS is 74.3 / 74.5 / 75.4 across
NTC / NTB / Existing-to-Credit. If our model reintroduces a segment penalty, the
system has recreated the exclusion it was built to remove, and no amount of AUC
compensates.

So every slice's metrics are computed, logged to MLflow, rendered on the metrics
dashboard, and asserted by a marker-gated regression test. The claim is a failing
test, not a paragraph.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

MIN_SLICE_ROWS = 50


def _slice_metrics(
    actual_pd: NDArray[np.float64],
    predicted_pd: NDArray[np.float64],
    actual_eligible: NDArray[np.float64],
    predicted_eligible: NDArray[np.float64],
    decision: NDArray[Any] | None = None,
) -> dict[str, float]:
    """Metrics for one cohort.

    ``predicted_eligible`` is a *score* and drives AUC; ``decision`` is the policy
    engine's actual approve/decline and drives the approval rate. Conflating them
    produces a nonsense approval rate: thresholding a PD-derived score at 0.5
    "approves" 99% of a book the policy engine approves 73% of.
    """
    approval = decision if decision is not None else (predicted_eligible >= 0.5)
    metrics: dict[str, float] = {
        "rows": float(len(actual_pd)),
        "pd_mae": round(float(np.abs(actual_pd - predicted_pd).mean()), 6),
        "pd_mean_actual": round(float(actual_pd.mean()), 6),
        "pd_mean_predicted": round(float(predicted_pd.mean()), 6),
        "pd_bias": round(float(predicted_pd.mean() - actual_pd.mean()), 6),
        "approval_rate_actual": round(float(actual_eligible.mean()), 6),
        "approval_rate_predicted": round(float(np.asarray(approval, dtype="float64").mean()), 6),
    }
    # AUC is undefined on a single-class slice; absent beats a misleading 0.5.
    if 0 < actual_eligible.sum() < len(actual_eligible):
        metrics["eligibility_auc"] = round(
            float(roc_auc_score(actual_eligible, predicted_eligible)), 6
        )
    return metrics


def slice_report(
    frame: pd.DataFrame,
    *,
    actual_pd: str = "Probability_of_Default",
    predicted_pd: str = "predicted_pd",
    actual_eligible: str = "Credit_Eligible",
    predicted_eligible: str = "predicted_eligibility",
    decision: str | None = "eligible",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Metrics overall and per level of every configured slice column."""
    cfg = config if config is not None else load_config()
    columns = [c for c in cfg["model_params"]["fairness_slice_columns"] if c in frame.columns]

    truth_pd = pd.to_numeric(frame[actual_pd], errors="coerce").to_numpy(dtype="float64")
    pred_pd = np.asarray(frame[predicted_pd], dtype="float64")
    # The eligibility label arrives as "Yes"/"No" from the dataset and as a bool
    # from the store; both are normalised to a float indicator here.
    eligible_series = frame[actual_eligible]
    is_text = eligible_series.dtype == object or str(eligible_series.dtype) == "string"
    indicator = eligible_series.astype("string").eq("Yes") if is_text else eligible_series
    truth_eligible: NDArray[np.float64] = np.asarray(indicator, dtype="float64")
    pred_eligible = np.asarray(frame[predicted_eligible], dtype="float64")
    decisions: NDArray[np.float64] | None = (
        np.asarray(frame[decision].astype(bool).to_numpy(), dtype="float64")
        if decision is not None and decision in frame.columns
        else None
    )

    overall = _slice_metrics(truth_pd, pred_pd, truth_eligible, pred_eligible, decisions)
    report: dict[str, Any] = {"overall": overall, "slices": {}, "max_deviation": {}}

    for column in columns:
        levels: dict[str, dict[str, float]] = {}
        for level, group in frame.groupby(column, observed=True):
            if len(group) < MIN_SLICE_ROWS:
                continue
            index = group.index
            positions = frame.index.get_indexer(index)
            levels[str(level)] = _slice_metrics(
                truth_pd[positions],
                pred_pd[positions],
                truth_eligible[positions],
                pred_eligible[positions],
                decisions[positions] if decisions is not None else None,
            )
        report["slices"][column] = levels
        if levels:
            # Deviation is measured on *residual* error, not on the outcome rate:
            # NTC firms legitimately differ from Existing-to-Credit ones, but the
            # model's error must not.
            deviations = {
                metric: round(max(abs(v[metric] - overall[metric]) for v in levels.values()), 6)
                for metric in ("pd_mae", "pd_bias")
                if all(metric in v for v in levels.values())
            }
            report["max_deviation"][column] = deviations

    logger.info(
        "Fairness slices computed",
        columns=columns,
        max_deviation=report["max_deviation"],
    )
    return report


def to_frame(report: dict[str, Any]) -> pd.DataFrame:
    """Flatten a slice report into one row per (column, level) for CSV/HTML."""
    records: list[dict[str, Any]] = [{"slice": "overall", "level": "all", **report["overall"]}]
    for column, levels in report["slices"].items():
        for level, metrics in levels.items():
            records.append({"slice": column, "level": level, **metrics})
    return pd.DataFrame.from_records(records)


def worst_deviation(report: dict[str, Any], metric: str = "pd_mae") -> float:
    """Largest deviation of ``metric`` from the portfolio value, across all slices."""
    values = [
        deviations[metric]
        for deviations in report.get("max_deviation", {}).values()
        if metric in deviations
    ]
    return max(values) if values else 0.0
