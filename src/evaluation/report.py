"""Assemble, persist and gate an evaluation report."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.data_loader import load_raw, split_features_labels
from src.evaluation import fairness, metrics
from src.model.utilities import ModelLoader
from src.scoring.pillar_calibration import fails_floor
from src.scoring.scoring_orchestration import score_portfolio
from src.utils import write_json
from src.utils.config import eval_reports_dir, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _sample(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows = int(config["evaluation"].get("sample_rows", 0) or 0)
    if rows and rows < len(frame):
        return frame.sample(n=rows, random_state=int(config["model_params"]["random_state"]))
    return frame


def build_report(
    config: dict[str, Any] | None = None,
    *,
    input_path: Path | None = None,
    nrows: int | None = None,
) -> dict[str, Any]:
    """Score the portfolio end to end and measure everything we claim."""
    cfg = config if config is not None else load_config()

    raw = _sample(load_raw(input_path, cfg, nrows=nrows), cfg)
    features, labels = split_features_labels(raw, cfg)

    bundle = ModelLoader(cfg).load(with_shap=False)
    scored = score_portfolio(features, bundle, cfg).reset_index(drop=True)
    labels = labels.reset_index(drop=True)

    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "rows": int(len(scored)),
        "model_version": bundle.model_version,
        "policy_version": str(cfg["policy_version"]),
        "rubric_version": str(cfg["rubric_version"]),
        "has_ml_model": bundle.has_ml,
        "pillars": metrics.pillar_metrics(scored, labels, cfg),
        "health_score": metrics.health_score_metrics(scored, labels, cfg),
        "probability_of_default": metrics.pd_metrics(
            scored,
            labels,
            cfg,
            n_features=0 if bundle.pd_model is None else len(bundle.pd_model.features),
        ),
        "eligibility": metrics.eligibility_metrics(scored, labels),
        "decisions": metrics.decision_agreement(scored, labels),
        "limits": metrics.limit_metrics(scored, labels),
        "calibration_curve": metrics.calibration_curve(scored, labels, cfg),
        "fairness": fairness.build(scored, labels, cfg),
        "thresholds": cfg["evaluation"]["thresholds"],
    }
    report["breaches"] = check_thresholds(report, cfg)
    logger.info(
        "Evaluation report built",
        rows=report["rows"],
        breaches=report["breaches"],
        fhs_mae=report["health_score"]["mae"],
        pd_mae=report["probability_of_default"]["mae"],
    )
    return report


def check_thresholds(report: dict[str, Any], config: dict[str, Any]) -> list[str]:
    """Compare a report against ``evaluation.thresholds``; returns the breaches.

    This is the function the marker-gated regression test asserts on, which is how
    the segment-neutrality claim becomes a failing test rather than a paragraph.
    """
    thresholds = config["evaluation"]["thresholds"]
    breaches: list[str] = []

    pd_mae = report["probability_of_default"]["mae"]
    if pd_mae > float(thresholds["pd_mae_max"]):
        breaches.append(f"pd_mae {pd_mae:.5f} > {thresholds['pd_mae_max']}")

    auc = report["eligibility"]["roc_auc"]
    if auc < float(thresholds["eligibility_auc_min"]):
        breaches.append(f"eligibility_auc {auc:.5f} < {thresholds['eligibility_auc_min']}")

    pillar_floor = float(thresholds["pillar_adj_r2_min"])
    for key, values in report["pillars"].items():
        if fails_floor(values["adj_r2"], pillar_floor):
            breaches.append(f"pillar {key} adj_r2 {values['adj_r2']:.4f} < {pillar_floor}")

    fhs_mae = report["health_score"]["mae"]
    if fhs_mae > float(thresholds["fhs_mae_max"]):
        breaches.append(f"fhs_mae {fhs_mae:.4f} > {thresholds['fhs_mae_max']}")

    mape = report["limits"].get("mape")
    if mape is not None and mape > float(thresholds["limit_mape_max"]):
        breaches.append(f"limit_mape {mape:.4f} > {thresholds['limit_mape_max']}")

    # The fairness threshold is written in score points; PD-error deviation is a
    # fraction, so it is scaled onto the same 0-100 scale before comparison.
    deviation = float(report["fairness"].get("worst_pd_mae_deviation", 0.0)) * 100.0
    if deviation > float(thresholds["fairness_max_deviation_points"]):
        breaches.append(
            f"fairness pd_mae deviation {deviation:.4f} pts "
            f"> {thresholds['fairness_max_deviation_points']}"
        )
    return breaches


def to_markdown(report: dict[str, Any]) -> str:
    """A human-readable summary, suitable for pasting into the model card."""
    lines = [
        "# MSME Financial Health Card — evaluation report",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Rows scored: **{report['rows']:,}**",
        f"- Model version: `{report['model_version'] or 'rubric-only'}`"
        f" · policy `{report['policy_version']}` · rubric `{report['rubric_version']}`",
        "",
        "## Threshold gates",
        "",
    ]
    if report["breaches"]:
        lines += ["| Breach |", "|---|", *[f"| {b} |" for b in report["breaches"]], ""]
    else:
        lines += ["All configured thresholds met.", ""]

    health = report["health_score"]
    pd_block = report["probability_of_default"]
    lines += [
        "## Headline metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Financial Health Score adjusted R² | {health['adj_r2']:.4f} |",
        f"| Financial Health Score R² (unadjusted) | {health['r2']:.4f} |",
        f"| Financial Health Score MAE | {health['mae']:.3f} points |",
        f"| PD adjusted R² | {pd_block['adj_r2']:.4f} |",
        f"| PD R² (unadjusted) | {pd_block['r2']:.4f} |",
        f"| PD MAE | {pd_block['mae']:.5f} |",
        f"| PD expected calibration error | {pd_block['ece']:.5f} |",
        f"| Eligibility ROC-AUC | {report['eligibility']['roc_auc']:.4f} |",
        f"| Eligibility KS | {report['eligibility']['ks']:.4f} |",
        f"| Risk-band agreement | {report['decisions']['risk_band_agreement']:.4f} |",
        f"| Eligibility agreement | {report['decisions']['eligibility_agreement']:.4f} |",
        f"| Credit limit MAPE | {report['limits'].get('mape', float('nan')):.4f} |",
        "",
        "## Pillar rubric fit",
        "",
        "| Pillar | Adjusted R² | R² | Drivers | MAE |",
        "|---|---|---|---|---|",
    ]
    for key, values in report["pillars"].items():
        lines.append(
            f"| {key} | {values['adj_r2']:.4f} | {values['r2']:.4f} | "
            f"{int(values['n_features'])} | {values['mae']:.3f} |"
        )

    lines += [
        "",
        "## Fairness slices",
        "",
        "| Slice | Level | Firms | PD MAE | PD bias | Approval (ours) | Approval (dataset) |",
        "|---|---|---|---|---|---|---|",
    ]
    for column, levels in report["fairness"]["slices"].items():
        for level, values in levels.items():
            lines.append(
                f"| {column} | {level} | {int(values['rows']):,} | {values['pd_mae']:.5f} | "
                f"{values['pd_bias']:+.5f} | {values['approval_rate_predicted']:.3f} | "
                f"{values['approval_rate_actual']:.3f} |"
            )
    worst = float(report["fairness"].get("worst_pd_mae_deviation", 0.0)) * 100.0
    lines += [
        "",
        f"Largest PD-error deviation across all slices: **{worst:.4f} points** "
        f"(limit {report['thresholds']['fairness_max_deviation_points']}).",
        "",
    ]
    return "\n".join(lines)


def save(report: dict[str, Any], config: dict[str, Any] | None = None) -> tuple[Path, Path]:
    """Write a timestamped JSON + markdown pair. Reports accumulate as a history."""
    cfg = config if config is not None else load_config()
    directory = eval_reports_dir(cfg)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = report["generated_at"].replace(":", "-")
    json_path = directory / f"report-{stamp}.json"
    markdown_path = directory / f"report-{stamp}.md"
    write_json(report, json_path)
    markdown_path.write_text(to_markdown(report), encoding="utf-8")

    # A stable filename so the model card and CI can reference the newest report
    # without globbing.
    write_json(report, directory / "latest.json")
    (directory / "latest.md").write_text(to_markdown(report), encoding="utf-8")
    logger.info("Evaluation report saved", json=str(json_path), markdown=str(markdown_path))
    return json_path, markdown_path


def load_latest(config: dict[str, Any] | None = None) -> dict[str, Any] | None:
    cfg = config if config is not None else load_config()
    path = eval_reports_dir(cfg) / "latest.json"
    if not path.is_file():
        return None
    from src.utils import read_json

    return read_json(path)  # type: ignore[no-any-return]
