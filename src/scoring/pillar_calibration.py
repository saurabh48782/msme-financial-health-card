"""Fit the pillar rubrics against the dataset's own pillar columns.

Why this module exists: a rubric assembled from domain anchors ranks firms
correctly but lands on its own scale — measured raw, the six rubrics correlate
0.78-0.95 with the dataset's pillars yet sit 12-25 points below them. Fitting
makes the transparent rubric *provably* track ground truth instead of asserting it.

The fit is a single non-negative least squares of the true pillar on the driver
ramp scores, which yields the weights and the scale at once::

    coef     <- LinearRegression(positive=True)
    weights  = coef / sum(coef)          # then floored, see below
    lo, hi   <- refit on the weighted rubric, then clipped into 0-100

``lo`` and ``hi`` are the pillar's ``band`` in ``params.yaml``: the score of a firm
at every worst anchor and at every best. The refit produces them as an intercept
and a slope, but since the rubric weights sum to 1 an affine map of a weighted
mean *is* a weighted mean over the mapped range — so the scale is a property of
the pillar, not a transform applied after it. Clipping the band into 0-100 at fit
time is what lets the runtime score without a clamp; the metrics below are
computed from the clipped band, so what is reported is what the rubric produces.

Non-negativity is not a technicality — it guarantees no driver can be perverse
(better GST filing can never lower a Compliance score), which is a fairness and
audit property, not just a fitting trick.

**The floor is the one piece of judgement in the fit.** A pure fit drives four
drivers to zero — among them the cash buffer and customer concentration — because
this synthetic generator happens not to use them. A driver being unused by a
generator is not evidence it is irrelevant to real MSME credit, and a rubric that
silently drops the cash buffer would be worthless in production. So each weight is
held at a floor of ``WEIGHT_FLOOR`` and the vector renormalised::

    w = normalise(max(fitted, WEIGHT_FLOOR))

Renormalising after the clamp means the realised floor is slightly under the
nominal one (~0.028 for 0.03). That is deliberate: iterating to honour it exactly
would be precision theatre, because the third decimal of a rubric weight is noise
next to the choice of anchors.

Output is a report, not a config overwrite. A rubric weight change is a reviewed
commit to ``params.yaml``, not something a training job does behind your back —
so this prints a paste-ready YAML block and writes a JSON artifact.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression

from src.scoring.pillars import SCORE_MAX, SCORE_MIN, driver_scores, pillar_keys
from src.utils import read_dataframe, write_json
from src.utils.config import DATA_DIR, load_config, processed_features_path
from src.utils.custom_error import CustomError
from src.utils.logger import get_logger

logger = get_logger(__name__)

WEIGHT_FLOOR = 0.03
# A degenerate fit must still leave a usable band rather than a single point.
_MIN_BAND_WIDTH = 1.0
CALIBRATION_ARTIFACT = DATA_DIR / "artifacts" / "pillar_calibration.json"


def r_squared(actual: NDArray[np.float64], predicted: NDArray[np.float64]) -> float:
    """Plain coefficient of determination. Reported, but never gated on.

    Kept because it is the number every reader recognises and because the gap
    between it and :func:`adjusted_r_squared` is itself the diagnostic: a wide
    gap means the fit is buying its score with predictors rather than signal.
    """
    actual = np.asarray(actual, dtype="float64")
    predicted = np.asarray(predicted, dtype="float64")
    total = float(((actual - actual.mean()) ** 2).sum())
    if total == 0.0:
        return 0.0
    return 1.0 - float(((actual - predicted) ** 2).sum()) / total


def adjusted_r_squared(
    actual: NDArray[np.float64], predicted: NDArray[np.float64], n_features: int
) -> float:
    """R² charged a degree of freedom for every predictor behind the fit.

    Plain R² is monotone in predictor count — bolt a column of noise onto the
    design matrix and it still rises — so it cannot answer the only question a
    gate cares about: did that driver earn its place? The adjustment prices each
    predictor against the sample that has to support it::

        adj = 1 - (1 - R²) * (n - 1) / (n - p - 1)

    ``n_features`` is the count of *fitted* predictors that produced
    ``predicted``, not the width of the frame it was computed from. A rubric
    whose weights were fitted from k drivers costs k, even though the final
    prediction is an affine function of a single weighted sum.

    Returns NaN when the sample cannot pay for the predictors (``n <= p + 1``),
    where the quantity is undefined. Every threshold gate routes NaN through
    :func:`fails_floor`, so an unsupportable fit reads as a breach, not a pass.
    """
    if n_features < 0:
        raise CustomError(f"n_features must be non-negative, got {n_features}")
    actual = np.asarray(actual, dtype="float64")
    rows = int(actual.size)
    if rows - n_features - 1 <= 0:
        return float("nan")
    plain = r_squared(actual, predicted)
    return 1.0 - (1.0 - plain) * (rows - 1) / (rows - n_features - 1)


def fails_floor(value: float, floor: float) -> bool:
    """True when ``value`` misses ``floor`` — NaN included.

    Written as a helper rather than ``value < floor`` at each gate because a NaN
    adjusted R² silently *passes* a naive ``<`` comparison, which is the one
    failure mode a gate must not have.
    """
    return math.isnan(value) or value < floor


def _floored(weights: NDArray[np.float64], floor: float, *, places: int = 4) -> NDArray[np.float64]:
    """Clamp every weight to ``floor``, renormalise, and round to sum to *exactly* 1.

    See the module docstring for why the realised floor lands just under the
    nominal one and why that is fine.

    The rounding is not cosmetic. ``validate_config`` asserts driver weights sum to
    1.0, and rounding a normalised vector to 4 places routinely lands on 0.9999 or
    1.0001 — so the residual is absorbed by the largest weight, where it is
    proportionally smallest.
    """
    if floor * len(weights) >= 1.0:
        raise CustomError(
            f"weight floor {floor} is impossible for {len(weights)} drivers "
            f"(needs {floor * len(weights):.2f} of the available 1.0)"
        )
    clamped = np.maximum(weights, floor)
    floored: NDArray[np.float64] = np.round(clamped / clamped.sum(), places)
    dominant = int(np.argmax(floored))
    floored[dominant] = round(floored[dominant] + (1.0 - floored.sum()), places)
    return floored


def calibrate_pillar(
    frame: pd.DataFrame,
    pillar_key: str,
    config: dict[str, Any] | None = None,
    *,
    floor: float = WEIGHT_FLOOR,
) -> dict[str, Any]:
    """Fit weights and the affine transform for one pillar against its label column."""
    cfg = config if config is not None else load_config()
    label_column = cfg["schema"]["pillar_label_map"][pillar_key]
    if label_column not in frame.columns:
        raise CustomError(
            f"cannot calibrate {pillar_key!r}: label column {label_column!r} is absent"
        )

    spec: dict[str, dict[str, Any]] = cfg["pillars"][pillar_key]["drivers"]
    names = list(spec)

    scores = driver_scores(frame, pillar_key, cfg)
    target = pd.to_numeric(frame[label_column], errors="coerce")
    # Fit on rows where every driver is observed: a partially observed row would
    # otherwise attribute a structurally missing driver's signal to its neighbours.
    mask = scores.notna().all(axis=1) & target.notna()
    if int(mask.sum()) < 100:
        raise CustomError(
            f"cannot calibrate {pillar_key!r}: only {int(mask.sum())} fully observed rows"
        )

    fit = LinearRegression(positive=True).fit(scores.loc[mask, names], target[mask])
    coef = np.asarray(fit.coef_, dtype="float64")
    fitted_slope = float(coef.sum())
    if fitted_slope <= 0.0:
        raise CustomError(f"degenerate fit for {pillar_key!r}: all coefficients are zero")
    fitted_weights = coef / fitted_slope
    weights = _floored(fitted_weights, floor)

    # Refit the scale on the floored rubric, since the floor moved it, and read it
    # back off as the band the pillar reports on.
    raw = (scores.loc[mask, names].to_numpy() * weights).sum(axis=1)
    affine = LinearRegression().fit(raw.reshape(-1, 1), target[mask])
    slope, intercept = float(affine.coef_[0]), float(affine.intercept_)
    lo = min(max(intercept, SCORE_MIN), SCORE_MAX)
    hi = min(max(intercept + SCORE_MAX * slope, lo + _MIN_BAND_WIDTH), SCORE_MAX)
    predicted = lo + (hi - lo) * raw / SCORE_MAX
    observed = target[mask].to_numpy(dtype="float64")

    result = {
        "pillar": pillar_key,
        "label_column": label_column,
        "rows_fitted": int(mask.sum()),
        "weight_floor": floor,
        "band": [round(lo, 4), round(hi, 4)],
        "weights": {name: float(w) for name, w in zip(names, weights, strict=True)},
        "fitted_weights": {
            name: round(float(w), 4) for name, w in zip(names, fitted_weights, strict=True)
        },
        "metrics": {
            # Every R² here is charged len(names) predictors: the band refit takes
            # one input, but that input is a weighted sum whose weights were
            # themselves fitted from all the drivers on this same data.
            "adj_r2": round(adjusted_r_squared(observed, predicted, len(names)), 4),
            "r2": round(r_squared(observed, predicted), 4),
            "mae": round(float(np.abs(observed - predicted).mean()), 4),
            # The unfloored NNLS fit: the ceiling the weight floor is paid out of.
            "pure_fit_adj_r2": round(
                adjusted_r_squared(observed, fit.predict(scores.loc[mask, names]), len(names)), 4
            ),
            "n_features": float(len(names)),
        },
    }
    logger.info(
        "Pillar calibrated",
        pillar=pillar_key,
        adj_r2=result["metrics"]["adj_r2"],
        r2=result["metrics"]["r2"],
        mae=result["metrics"]["mae"],
        rows=result["rows_fitted"],
    )
    return result


def calibrate_all(
    frame: pd.DataFrame,
    config: dict[str, Any] | None = None,
    *,
    floor: float = WEIGHT_FLOOR,
) -> dict[str, Any]:
    """Calibrate every pillar that has a label column."""
    cfg = config if config is not None else load_config()
    results = {
        key: calibrate_pillar(frame, key, cfg, floor=floor)
        for key in pillar_keys(cfg)
        if key in cfg["schema"]["pillar_label_map"]
    }
    threshold = float(cfg["evaluation"]["thresholds"]["pillar_adj_r2_min"])
    failing = [k for k, r in results.items() if fails_floor(r["metrics"]["adj_r2"], threshold)]
    return {
        "weight_floor": floor,
        "pillar_adj_r2_min": threshold,
        "pillars": results,
        "below_threshold": failing,
    }


def to_yaml_block(report: dict[str, Any], config: dict[str, Any] | None = None) -> str:
    """Render the fitted numbers as a paste-ready ``params.yaml`` fragment.

    Complete driver lines, not just the weights, so review is a diff of the block
    rather than a hunt for the numbers that moved.
    """
    cfg = config if config is not None else load_config()
    lines: list[str] = ["# --- fitted by src.scoring.pillar_calibration ---"]
    for key, result in report["pillars"].items():
        spec: dict[str, dict[str, Any]] = cfg["pillars"][key]["drivers"]
        lo, hi = result["band"]
        lines.append(f"  {key}:")
        lines.append(f'    label: "{cfg["pillars"][key]["label"]}"')
        lines.append(f"    band: [{lo}, {hi}]")
        lines.append("    drivers:")
        for name, weight in result["weights"].items():
            driver = spec[name]
            parts = [
                f"source: {driver['source']}",
                f"weight: {weight}",
                f"worst: {driver['worst']}",
                f"best: {driver['best']}",
            ]
            if "label" in driver:
                parts.append(f'label: "{driver["label"]}"')
            if driver.get("optional"):
                parts.append("optional: true")
            lines.append(f"      {name}: {{{', '.join(parts)}}}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit the pillar rubrics against the dataset.")
    parser.add_argument("--input", type=Path, default=None, help="processed features CSV")
    parser.add_argument("--floor", type=float, default=WEIGHT_FLOOR)
    parser.add_argument("--output", type=Path, default=CALIBRATION_ARTIFACT)
    args = parser.parse_args(argv)

    logger.info("--- Starting pillar calibration ---")
    try:
        cfg = load_config()
        source = args.input or processed_features_path(cfg)
        frame = read_dataframe(source)
        report = calibrate_all(frame, cfg, floor=args.floor)
        write_json(report, args.output)

        print(to_yaml_block(report, cfg))
        print(f"\n# Adjusted R^2 by pillar (threshold {report['pillar_adj_r2_min']:.2f}):")
        for key, result in report["pillars"].items():
            metrics = result["metrics"]
            mark = "FAIL" if fails_floor(metrics["adj_r2"], report["pillar_adj_r2_min"]) else "ok "
            print(
                f"#   [{mark}] {key:22s} adj_r2={metrics['adj_r2']:.4f} "
                f"(r2={metrics['r2']:.4f}, p={int(metrics['n_features'])}) "
                f"mae={metrics['mae']:.3f} "
                f"(unfloored ceiling {metrics['pure_fit_adj_r2']:.4f})"
            )
        if report["below_threshold"]:
            logger.error(
                "Pillars below the adjusted R2 threshold", pillars=report["below_threshold"]
            )
            return 1
        return 0
    except Exception:
        logger.error("Pillar calibration failed", exc_info=True)
        return 1
    finally:
        logger.info("--- Pillar calibration Finished ---")


if __name__ == "__main__":
    raise SystemExit(main())
