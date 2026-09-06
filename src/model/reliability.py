"""Is the PD trustworthy as a number, not just as a ranking?

A PD is not a ranking, it is a price. If the model says 8% and the realised rate
is 12%, every loan in that bucket is underpriced regardless of how good the AUC
is. So discrimination is not enough and the report carries a reliability curve -
predicted PD against realised PD, in equal-count bins - plus the expected
calibration error that summarises it.

The PD head is an XGBoost **regressor** on a continuous target, already trained to
minimise squared error against the quantity it reports, so this module measures
calibration rather than correcting it. The measured ECE (see the model card) says
there is nothing material to correct, and the bias is asserted per cohort by the
eval gate.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def reliability_curve(
    actual: NDArray[np.float64], predicted: NDArray[np.float64], *, bins: int = 10
) -> list[dict[str, float]]:
    """Equal-count bins of predicted PD against the mean realised PD.

    Quantile bins rather than equal-width: PD is heavily skewed towards zero, so
    equal-width bins would put 90% of the portfolio in the first bucket and report
    calibration on a handful of firms in the rest.
    """
    actual = np.asarray(actual, dtype="float64")
    predicted = np.clip(np.asarray(predicted, dtype="float64"), 0.0, 1.0)
    if len(predicted) == 0:
        return []
    edges = np.unique(np.quantile(predicted, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 2:
        return [
            {
                "bin": 0,
                "count": float(len(predicted)),
                "mean_predicted": round(float(predicted.mean()), 6),
                "mean_actual": round(float(actual.mean()), 6),
                "gap": round(float(predicted.mean() - actual.mean()), 6),
            }
        ]
    indices = np.clip(np.digitize(predicted, edges[1:-1], right=True), 0, len(edges) - 2)
    curve: list[dict[str, float]] = []
    for index in range(len(edges) - 1):
        mask = indices == index
        if not mask.any():
            continue
        mean_predicted = float(predicted[mask].mean())
        mean_actual = float(actual[mask].mean())
        curve.append(
            {
                "bin": float(index),
                "count": float(mask.sum()),
                "lower": round(float(edges[index]), 6),
                "upper": round(float(edges[index + 1]), 6),
                "mean_predicted": round(mean_predicted, 6),
                "mean_actual": round(mean_actual, 6),
                "gap": round(mean_predicted - mean_actual, 6),
            }
        )
    return curve


def expected_calibration_error(
    actual: NDArray[np.float64], predicted: NDArray[np.float64], *, bins: int = 10
) -> float:
    """Count-weighted mean absolute gap across the reliability bins."""
    curve = reliability_curve(actual, predicted, bins=bins)
    total = sum(row["count"] for row in curve) or 1.0
    return round(sum(abs(row["gap"]) * row["count"] for row in curve) / total, 6)
