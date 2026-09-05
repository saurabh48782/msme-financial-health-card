"""Shared fixtures.

Two rules the whole suite depends on:

* **No test touches the network, MLflow or a trained model.** Every boundary has
  a stub in ``tests/stubs.py``.
* **Environment is set before any config is loaded**, because ``load_config`` is
  ``lru_cache``d — a test that set an env var afterwards would silently see the
  previous value.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

os.environ.setdefault("LOG_TARGET", "stdout")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///:memory:")

from src.utils.config import ROOT_DIR, load_config  # noqa: E402


@pytest.fixture(scope="session")
def config() -> dict[str, Any]:
    """The real ``params.yaml``. Tests assert against shipped config, not a copy."""
    return load_config()


@pytest.fixture(scope="session")
def raw_frame() -> pd.DataFrame:
    """A slice of the real dataset, or a synthetic stand-in when it is absent.

    The dataset is DVC-tracked and not in git, so CI must not depend on it. The
    synthetic builder reproduces the same columns and the same structural-NULL
    relationship, which is what these tests actually assert on.
    """
    path = ROOT_DIR / "data" / "raw" / "msme_synthetic_50k.csv"
    if path.is_file():
        return pd.read_csv(path, nrows=600)
    from tests.stubs import synthetic_raw_frame

    return synthetic_raw_frame(600)


@pytest.fixture(scope="session")
def featured_frame(raw_frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Raw rows put through the real feature pipeline."""
    from src.data.data_loader import split_features_labels
    from src.data.data_orchestration import build_features

    features, labels = split_features_labels(raw_frame, config)
    featured, _ = build_features(features, config)
    return featured.merge(labels, on=config["schema"]["id_column"], how="left")


@pytest.fixture
def sample_row(featured_frame: pd.DataFrame) -> pd.Series:
    return featured_frame.iloc[0]


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    (tmp_path / "processed").mkdir(parents=True, exist_ok=True)
    (tmp_path / "artifacts").mkdir(parents=True, exist_ok=True)
    return tmp_path
