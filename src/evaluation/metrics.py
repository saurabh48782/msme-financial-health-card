"""Metric computation for the offline harness.

Deliberately separate from ``src/model``: those metrics describe a training run,
these describe the *system* as deployed — rubric, models, policy and limit sizing
together, measured against the labels we have.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.model.components import evaluate_classification, evaluate_probability
from src.model.reliability import expected_calibration_error, reliability_curve
from src.scoring.pillar_calibration import adjusted_r_squared, r_squared
from src.scoring.pillars import pillar_keys
from src.utils.logger import get_logger

logger = get_logger(__name__)


def pillar_metrics(
    scored: pd.DataFrame, labels: pd.DataFrame, config: dict[str, Any]
) -> dict[str, dict[str, float]]:
    """Adjusted R² and MAE of each pillar against the dataset's own pillar column.

    Each pillar is charged its own driver count, so a six-driver rubric cannot
    out-score a three-driver one purely on width.
    """
    mapping = config["schema"]["pillar_label_map"]
    out: dict[str, dict[str, float]] = {}
    for key in pillar_keys(config):
        column = mapping.get(key)
        if column is None or column not in labels.columns:
            continue
        actual = pd.to_numeric(labels[column], errors="coerce").to_numpy(dtype="float64")
        predicted = pd.to_numeric(scored[f"{key}_score"], errors="coerce").to_numpy(dtype="float64")
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        drivers = len(config["pillars"][key]["drivers"])
        out[key] = {
            "r2": round(r_squared(actual[mask], predicted[mask]), 6),
            "adj_r2": round(adjusted_r_squared(actual[mask], predicted[mask], drivers), 6),
            "n_features": float(drivers),
            "mae": round(float(np.abs(actual[mask] - predicted[mask]).mean()), 6),
            "rows": float(mask.sum()),
        }
    return out


def health_score_metrics(
    scored: pd.DataFrame, labels: pd.DataFrame, config: dict[str, Any]
) -> dict[str, float]:
    """Fit of the composite score. Charged one predictor per pillar it blends."""
    actual = pd.to_numeric(labels["Financial_Health_Score"], errors="coerce").to_numpy("float64")
    predicted = pd.to_numeric(scored["financial_health_score"], errors="coerce").to_numpy("float64")
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    features = len(pillar_keys(config))
    return {
        "r2": round(r_squared(actual[mask], predicted[mask]), 6),
        "adj_r2": round(adjusted_r_squared(actual[mask], predicted[mask], features), 6),
        "n_features": float(features),
        "mae": round(float(np.abs(actual[mask] - predicted[mask]).mean()), 6),
        "rows": float(mask.sum()),
    }


def pd_metrics(
    scored: pd.DataFrame, labels: pd.DataFrame, config: dict[str, Any], *, n_features: int
) -> dict[str, Any]:
    """PD fit and calibration. ``n_features`` is the served model's feature count.

    It comes from the loaded bundle rather than the config so the penalty tracks
    the model that actually produced these predictions, not the allowlist as it
    stands today — a rubric-only bundle predicts from no fitted features at all.
    """
    actual = pd.to_numeric(labels["Probability_of_Default"], errors="coerce").to_numpy("float64")
    predicted = pd.to_numeric(scored["probability_of_default"], errors="coerce").to_numpy("float64")
    bins = int(config["evaluation"]["calibration_bins"])
    metrics = evaluate_probability(pd.Series(actual), predicted, n_features=n_features)
    metrics["ece"] = expected_calibration_error(actual, predicted, bins=bins)
    return metrics


def eligibility_metrics(scored: pd.DataFrame, labels: pd.DataFrame) -> dict[str, float]:
    """Discrimination of our PD against the dataset's eligibility label.

    Scored as ``1 - PD`` because the policy gate is a PD cut: the question is
    whether the risk estimate separates eligible from ineligible firms, not
    whether a second classifier agrees with the first.
    """
    truth = labels["Credit_Eligible"].astype("string").eq("Yes").astype(int)
    scores = 1.0 - pd.to_numeric(scored["probability_of_default"], errors="coerce").to_numpy(
        "float64"
    )
    return evaluate_classification(truth, scores)


def decision_agreement(scored: pd.DataFrame, labels: pd.DataFrame) -> dict[str, float]:
    """How often the policy engine reproduces the dataset's own decisions."""
    truth_eligible = labels["Credit_Eligible"].astype("string").eq("Yes")
    ours = scored["eligible"].astype(bool)
    band_agreement = float(
        (
            scored["risk_band"].astype("string") == labels["Credit_Risk_Category"].astype("string")
        ).mean()
    )
    return {
        "eligibility_agreement": round(
            float((ours.to_numpy() == truth_eligible.to_numpy()).mean()), 6
        ),
        "risk_band_agreement": round(band_agreement, 6),
        "approval_rate_ours": round(float(ours.mean()), 6),
        "approval_rate_dataset": round(float(truth_eligible.mean()), 6),
    }


def limit_metrics(scored: pd.DataFrame, labels: pd.DataFrame) -> dict[str, float]:
    """Limit accuracy on firms both we and the dataset approve.

    Restricted to jointly-approved firms because a limit comparison on a firm one
    side declines is a comparison against zero, which measures the eligibility gate
    twice instead of measuring the sizing rule once.
    """
    both = (
        scored["eligible"].astype(bool).to_numpy()
        & labels["Credit_Eligible"].astype("string").eq("Yes").to_numpy()
    )
    if not both.any():
        return {"rows": 0.0}
    ours = pd.to_numeric(scored["credit_limit_inr"], errors="coerce").to_numpy("float64")[both]
    theirs = pd.to_numeric(labels["Recommended_Credit_Limit_INR"], errors="coerce").to_numpy(
        "float64"
    )[both]
    safe = np.where(theirs == 0, np.nan, theirs)
    ape = np.abs(ours - safe) / safe
    return {
        "rows": float(both.sum()),
        "mape": round(float(np.nanmean(ape)), 6),
        "median_ape": round(float(np.nanmedian(ape)), 6),
        "p90_ape": round(float(np.nanquantile(ape, 0.9)), 6),
    }


def calibration_curve(
    scored: pd.DataFrame, labels: pd.DataFrame, config: dict[str, Any]
) -> list[dict[str, float]]:
    return reliability_curve(
        pd.to_numeric(labels["Probability_of_Default"], errors="coerce").to_numpy("float64"),
        pd.to_numeric(scored["probability_of_default"], errors="coerce").to_numpy("float64"),
        bins=int(config["evaluation"]["calibration_bins"]),
    )
