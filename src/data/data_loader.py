"""Load the raw MSME dataset with an explicit dtype schema.

The loader is also the leakage firewall's first gate: :func:`split_features_labels`
hands back the ID column plus the 33 allowlisted inputs and *nothing else*, so no
downstream stage can accidentally see a pillar score or ``Probability_of_Default``.
Labels come back in a separate frame, fetched deliberately by name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import read_dataframe
from src.utils.config import load_config, raw_csv_path
from src.utils.custom_error import CustomError
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _dtypes(config: dict[str, Any]) -> dict[str, str]:
    schema = config["schema"]
    return {name: "category" for name in schema["categorical_columns"]}


def load_raw(
    path: str | Path | None = None,
    config: dict[str, Any] | None = None,
    *,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Read the raw dataset, categoricals typed, ``MSME_ID`` as a plain column."""
    cfg = config if config is not None else load_config()
    source = Path(path) if path is not None else raw_csv_path(cfg)
    frame = read_dataframe(source, dtype=_dtypes(cfg), nrows=nrows)
    logger.info("Raw dataset loaded", source=str(source), rows=len(frame), columns=frame.shape[1])
    return frame


def split_features_labels(
    frame: pd.DataFrame, config: dict[str, Any] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a raw frame into (id + allowlisted features, id + labels).

    Raises when a declared column is absent - a silently missing feature would
    become a silently NaN driver, which is far worse than a failed load.
    """
    cfg = config if config is not None else load_config()
    schema = cfg["schema"]
    id_column = schema["id_column"]
    features: list[str] = list(schema["feature_columns"])
    labels: list[str] = list(schema["label_columns"])

    missing = [c for c in (id_column, *features) if c not in frame.columns]
    if missing:
        raise CustomError(f"Raw dataset is missing required columns: {missing}")

    feature_frame = frame.loc[:, [id_column, *features]].copy()
    present_labels = [c for c in labels if c in frame.columns]
    label_frame = frame.loc[:, [id_column, *present_labels]].copy()
    if len(present_labels) != len(labels):
        logger.warning(
            "Label columns absent - inference-only dataset",
            missing=sorted(set(labels) - set(present_labels)),
        )
    return feature_frame, label_frame
