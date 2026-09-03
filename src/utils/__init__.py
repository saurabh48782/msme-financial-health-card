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
    """Read a CSV file."""
    if not str(path).lower().endswith((".csv", ".csv.gz")):
        raise CustomError(f"Unsupported data file extension: {path}")
    try:
        csv: pd.DataFrame = pd.read_csv(path, **kwargs)
    except (OSError, ValueError) as error:
        raise CustomError(error) from error
    return csv


def write_dataframe(frame: pd.DataFrame, path: str | Path, **kwargs: Any) -> None:
    """Write a DataFrame as CSV, creating parent directories."""
    if not str(path).lower().endswith(".csv"):
        raise CustomError(f"Unsupported data file extension: {path}")
    _ensure_parent(path)
    try:
        frame.to_csv(path, index=False, **kwargs)
    except (OSError, ValueError) as error:
        raise CustomError(error) from error


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
