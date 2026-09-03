"""Winsorisation, dtype coercion and categorical ordinal mapping.

Winsorisation limits are *fitted once* on the training portfolio and persisted, so
a single MSME scored in real time is clipped against the same bounds as the batch
that trained the models.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def fit_winsor_limits(
    frame: pd.DataFrame, config: dict[str, Any] | None = None
) -> dict[str, dict[str, float]]:
    """Compute per-column clip bounds from the configured quantiles."""
    cfg = config if config is not None else load_config()
    spec = cfg["preprocessing"]["winsorise"]
    lower_q, upper_q = float(spec["lower_quantile"]), float(spec["upper_quantile"])
    limits: dict[str, dict[str, float]] = {}
    for column in spec["columns"]:
        if column not in frame.columns:
            continue
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if series.empty:
            continue
        limits[column] = {
            "lower": float(series.quantile(lower_q)),
            "upper": float(series.quantile(upper_q)),
        }
    logger.info(
        "Winsorisation limits fitted", columns=len(limits), lower_q=lower_q, upper_q=upper_q
    )
    return limits


def apply_winsor_limits(frame: pd.DataFrame, limits: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Clip each column to its fitted bounds. Returns a copy."""
    out = frame.copy()
    for column, bounds in limits.items():
        if column not in out.columns:
            continue
        out[column] = pd.to_numeric(out[column], errors="coerce").clip(
            lower=bounds["lower"], upper=bounds["upper"]
        )
    return out


def coerce_dtypes(frame: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Numeric columns to float, categoricals to string-backed category."""
    cfg = config if config is not None else load_config()
    schema = cfg["schema"]
    categorical = set(schema["categorical_columns"])
    out = frame.copy()
    for column in schema["feature_columns"]:
        if column not in out.columns:
            continue
        if column in categorical:
            out[column] = out[column].astype("string").astype("category")
        else:
            out[column] = pd.to_numeric(out[column], errors="coerce").astype("float64")
    return out


def map_categorical_standing(
    frame: pd.DataFrame, config: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Add the ``*_standing`` ordinal columns the rubrics score against.

    An unmapped level yields NaN rather than 0.0 on purpose: 0.0 would silently
    treat an unrecognised business type as the worst possible one, whereas NaN
    makes the driver unavailable and lets the rubric redistribute its weight.
    """
    cfg = config if config is not None else load_config()
    standing: dict[str, dict[str, float]] = cfg.get("categorical_standing", {})
    source_columns = {
        "business_type_standing": "Business_Type",
        "location_standing": "Location_Category",
        "gst_return_frequency_standing": "GST_Return_Frequency",
    }
    out = frame.copy()
    for target, source in source_columns.items():
        if target not in standing or source not in out.columns:
            continue
        mapping = {str(k): float(v) for k, v in standing[target].items()}
        out[target] = (
            out[source].astype("string").map(mapping).astype("float64")  # type: ignore[arg-type]
        )
        unmapped = int(out[target].isna().sum() - out[source].isna().sum())
        if unmapped > 0:
            logger.warning(
                "Unmapped categorical level", column=source, standing=target, rows=unmapped
            )
    return out


def preprocess(
    frame: pd.DataFrame,
    config: dict[str, Any] | None = None,
    *,
    winsor_limits: dict[str, dict[str, float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Coerce -> winsorise -> map standings. Returns (frame, limits actually used).

    Pass ``winsor_limits`` at inference time; omit it to fit on ``frame``.
    """
    cfg = config if config is not None else load_config()
    out = coerce_dtypes(frame, cfg)
    limits = winsor_limits if winsor_limits is not None else fit_winsor_limits(out, cfg)
    out = apply_winsor_limits(out, limits)
    out = map_categorical_standing(out, cfg)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out, limits
