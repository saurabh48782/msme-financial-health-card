"""Raw CSV -> processed features CSV + winsorisation artifact.

Run as ``python -m src.data.data_orchestration``. Idempotent: rerunning it
overwrites the artifacts from the same raw input.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.data_loader import load_raw, split_features_labels
from src.data.data_preprocessor import preprocess
from src.data.data_validator import validate
from src.data.feature_engineer import add_derived_features
from src.utils import write_dataframe, write_json
from src.utils.config import load_config, processed_features_path, validate_config
from src.utils.logger import get_logger
from src.utils.tracing import trace_stage

logger = get_logger(__name__)

WINSOR_LIMITS_FILENAME = "winsor_limits.json"


def build_features(
    frame: pd.DataFrame,
    config: dict[str, Any] | None = None,
    *,
    winsor_limits: dict[str, dict[str, float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """The shared feature path, used by training, batch scoring and the API alike.

    Passing ``winsor_limits`` reuses the fitted limits (inference); omitting them
    fits on ``frame`` (training). Keeping one function for both is what stops
    train/serve skew.
    """
    cfg = config if config is not None else load_config()

    with trace_stage("preprocess", rows=len(frame)) as span:
        processed, limits = preprocess(frame, cfg, winsor_limits=winsor_limits)
        span["winsorised_columns"] = len(limits)

    with trace_stage("derive_features", rows=len(processed)) as span:
        featured = add_derived_features(processed, cfg)
        span["columns"] = featured.shape[1]

    return featured, limits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None, help="raw CSV (default: params.yaml)")
    parser.add_argument("--nrows", type=int, default=None, help="load only the first N rows")
    args = parser.parse_args(argv)

    logger.info("--- Starting data orchestration ---")
    try:
        cfg = validate_config(load_config())

        raw = load_raw(args.input, cfg, nrows=args.nrows)
        validate(raw, cfg).raise_for_errors()

        features, labels = split_features_labels(raw, cfg)
        featured, limits = build_features(features, cfg)

        id_column = cfg["schema"]["id_column"]
        combined = featured.merge(labels, on=id_column, how="left", validate="one_to_one")

        target = processed_features_path(cfg)
        write_dataframe(combined, target)
        write_json(limits, target.parent / WINSOR_LIMITS_FILENAME)

        logger.info(
            "Processed dataset written",
            path=str(target),
            rows=len(combined),
            columns=combined.shape[1],
        )
        return 0
    except Exception:
        logger.error("Data orchestration failed", exc_info=True)
        return 1
    finally:
        logger.info("--- Data orchestration Finished ---")


if __name__ == "__main__":
    raise SystemExit(main())
