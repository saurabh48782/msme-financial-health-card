"""Score a whole portfolio to CSV.

``python -m src.model.batch_inference --input-filename data/raw/msme_synthetic_50k.csv``

This is the path behind the portfolio dashboard and the monitoring view. It uses
the vectorised :func:`score_portfolio`, not a loop over ``score_one``: 50,000
firms through the per-card path would take minutes and produce identical numbers.
The unit tests assert the two agree, which is what makes the shortcut safe.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.data_loader import load_raw, split_features_labels
from src.model.utilities import ModelLoader
from src.scoring.scoring_orchestration import score_portfolio
from src.utils import write_dataframe
from src.utils.config import load_config, portfolio_scores_path, validate_config
from src.utils.custom_error import CustomError
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run(
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
    *,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Score every row of ``input_path`` and write the result."""
    cfg = config if config is not None else load_config()
    raw = load_raw(input_path, cfg, nrows=nrows)
    features, labels = split_features_labels(raw, cfg)

    bundle = ModelLoader(cfg).load(with_shap=False)
    scored = score_portfolio(features, bundle, cfg)

    # Carry the labels alongside so the portfolio view can show measured agreement
    # rather than only our own numbers.
    id_column = cfg["schema"]["id_column"]
    if labels.shape[1] > 1:
        scored = scored.merge(labels, on=id_column, how="left", validate="one_to_one")

    target = Path(output_path) if output_path is not None else portfolio_scores_path(cfg)
    write_dataframe(scored, target)
    logger.info("Portfolio scores written", path=str(target), rows=len(scored))
    return scored


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-filename", type=Path, default=None)
    parser.add_argument("--output-filename", type=Path, default=None)
    parser.add_argument("--nrows", type=int, default=None)
    args = parser.parse_args(argv)

    logger.info("--- Starting batch inference ---")
    try:
        cfg = validate_config(load_config())
        scored = run(
            args.input_filename,
            args.output_filename,
            cfg,
            nrows=args.nrows,
        )
        if scored.empty:
            raise CustomError("batch inference produced no rows")
        print(
            f"scored {len(scored):,} firms | approval rate "
            f"{scored['eligible'].mean():.1%} | mean FHS "
            f"{scored['financial_health_score'].mean():.1f}"
        )
        return 0
    except Exception:
        logger.error("Batch inference failed", exc_info=True)
        return 1
    finally:
        logger.info("--- Batch inference Finished ---")


if __name__ == "__main__":
    raise SystemExit(main())
