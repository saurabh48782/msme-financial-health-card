"""Cohort framing for the offline harness.

There is exactly one slicing implementation — :mod:`src.model.fairness` — so the
number in a report and the number logged to MLflow cannot disagree. This module
only reshapes the harness's scored frame into the columns that implementation
expects, and adds the two worst-deviation summaries the report quotes.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.model.fairness import slice_report, worst_deviation
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build(scored: pd.DataFrame, labels: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Slice metrics across every configured cohort column."""
    frame = scored.copy()
    frame["Probability_of_Default"] = labels["Probability_of_Default"].to_numpy()
    frame["Credit_Eligible"] = labels["Credit_Eligible"].to_numpy()
    frame["predicted_pd"] = pd.to_numeric(frame["probability_of_default"], errors="coerce")
    frame["predicted_eligibility"] = 1.0 - frame["predicted_pd"]

    for column in config["model_params"]["fairness_slice_columns"]:
        if column not in frame.columns and column in labels.columns:
            frame[column] = labels[column].to_numpy()

    report = slice_report(frame, config=config)
    report["worst_pd_mae_deviation"] = worst_deviation(report, "pd_mae")
    report["worst_pd_bias_deviation"] = worst_deviation(report, "pd_bias")
    return report
