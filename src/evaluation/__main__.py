"""Offline evaluation CLI.

    python -m src.evaluation calibrate   # refit the pillar rubrics and report adjusted R²
    python -m src.evaluation backtest    # PD reliability, discrimination, limit sanity
    python -m src.evaluation fairness    # slice tables by segment / location / industry
    python -m src.evaluation report      # the full report, JSON + markdown, timestamped

Deliberately not an HTTP surface: nothing on the request path imports this
package, so a heavy evaluation can never be triggered by a user request.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation import report as report_module
from src.utils import read_dataframe
from src.utils.config import load_config, validate_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, default=None, help="raw CSV (default: params.yaml)")
    parser.add_argument("--nrows", type=int, default=None, help="evaluate only the first N rows")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.evaluation", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    _add_common(sub.add_parser("calibrate", help="refit the pillar rubrics against the dataset"))
    _add_common(sub.add_parser("backtest", help="PD reliability, discrimination, limit sanity"))
    _add_common(sub.add_parser("fairness", help="slice tables by cohort"))

    full = sub.add_parser("report", help="full report, written to data/eval/reports/")
    full.add_argument("--no-save", action="store_true")
    _add_common(full)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logger.info("--- Starting evaluation ---", command=args.command)
    try:
        config = validate_config(load_config())

        if args.command == "calibrate":
            from src.scoring.pillar_calibration import calibrate_all, fails_floor, to_yaml_block
            from src.utils.config import processed_features_path

            frame = read_dataframe(args.input or processed_features_path(config))
            result = calibrate_all(frame, config)
            print(to_yaml_block(result))
            floor = result["pillar_adj_r2_min"]
            for key, values in result["pillars"].items():
                mark = "FAIL" if fails_floor(values["metrics"]["adj_r2"], floor) else "ok "
                print(
                    f"# [{mark}] {key:22s} adj_r2={values['metrics']['adj_r2']:.4f} "
                    f"(r2={values['metrics']['r2']:.4f})"
                )
            return 1 if result["below_threshold"] else 0

        built = report_module.build_report(config, input_path=args.input, nrows=args.nrows)

        if args.command == "backtest":
            print(
                json.dumps(
                    {
                        "health_score": built["health_score"],
                        "probability_of_default": built["probability_of_default"],
                        "eligibility": built["eligibility"],
                        "decisions": built["decisions"],
                        "limits": built["limits"],
                        "breaches": built["breaches"],
                    },
                    indent=2,
                )
            )
        elif args.command == "fairness":
            from src.model.fairness import to_frame

            print(to_frame(built["fairness"]).to_string(index=False))
            print(
                f"\nworst PD-error deviation: "
                f"{built['fairness']['worst_pd_mae_deviation'] * 100:.4f} points"
            )
        else:
            if not args.no_save:
                report_module.save(built, config)
            print(report_module.to_markdown(built))

        if built["breaches"]:
            logger.error("Threshold breaches", breaches=built["breaches"])
            return 1
        return 0
    except Exception:
        logger.error("Evaluation failed", exc_info=True)
        return 1
    finally:
        logger.info("--- Evaluation Finished ---")


if __name__ == "__main__":
    raise SystemExit(main())
