"""Small IO helpers. Local paths and ``s3://`` URIs are handled identically so a
pipeline stage never branches on where its data lives.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.custom_error import CustomError


def _is_remote(path: str | Path) -> bool:
    return str(path).startswith(("s3://", "gs://", "az://"))


def _ensure_parent(path: str | Path) -> None:
    if not _is_remote(path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_dataframe(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read a CSV or parquet file by extension."""
    suffix = str(path).lower()
    try:
        if suffix.endswith(".parquet"):
            parquet: pd.DataFrame = pd.read_parquet(path, **kwargs)
            return parquet
        if suffix.endswith((".csv", ".csv.gz")):
            csv: pd.DataFrame = pd.read_csv(path, **kwargs)
            return csv
    except (OSError, ValueError) as error:
        raise CustomError(error) from error
    raise CustomError(f"Unsupported data file extension: {path}")


def write_dataframe(frame: pd.DataFrame, path: str | Path, **kwargs: Any) -> None:
    """Write a DataFrame as CSV or parquet, creating parent directories."""
    _ensure_parent(path)
    suffix = str(path).lower()
    try:
        if suffix.endswith(".parquet"):
            frame.to_parquet(path, index=False, **kwargs)
            return
        if suffix.endswith(".csv"):
            frame.to_csv(path, index=False, **kwargs)
            return
    except (OSError, ValueError) as error:
        raise CustomError(error) from error
    raise CustomError(f"Unsupported data file extension: {path}")


def read_json(path: str | Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CustomError(error) from error


def write_json(payload: Any, path: str | Path, *, indent: int = 2) -> None:
    _ensure_parent(path)
    try:
        Path(path).write_text(
            json.dumps(payload, indent=indent, default=str, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise CustomError(error) from error
