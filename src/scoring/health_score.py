"""The Financial Health Score: a weighted aggregate of the six pillars.

The weights are not invented. Regressing the dataset's ``Financial_Health_Score``
on its six pillar columns gives R^2 = 0.831 with coefficients
(Compliance .278, Cashflow .243, Payment .230, Consistency .105, Stability .087,
Growth .079); those, renormalised to sum to 1.0, are ``pillars.fhs_weights``.

That 0.831 is also the honest ceiling on this task: roughly 17% of the label is
generator noise no rubric can recover. It is why the evaluation gate targets an
FHS MAE of 2.5 points rather than zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.scoring.pillars import (
    SCORE_MAX,
    SCORE_MIN,
    PillarResult,
    compute_pillars,
    pillar_keys,
)
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Grade:
    grade: str
    label: str


@dataclass(frozen=True)
class PillarShare:
    """A pillar's share of the Financial Health Score, in FHS points."""

    key: str
    label: str
    score: float
    weight: float
    contribution: float


@dataclass
class HealthScore:
    value: float
    grade: str
    grade_label: str
    pillar_shares: list[PillarShare]

    def decomposition_error(self) -> float:
        return abs(self.value - sum(s.contribution for s in self.pillar_shares))


def grade_for(score: float, config: dict[str, Any] | None = None) -> Grade:
    """Map a 0-100 score onto its configured grade band (evaluated top-down)."""
    cfg = config if config is not None else load_config()
    for band in cfg["grades"]:
        if score >= float(band["min"]):
            return Grade(grade=str(band["grade"]), label=str(band["label"]))
    last = cfg["grades"][-1]
    return Grade(grade=str(last["grade"]), label=str(last["label"]))


def compute_health_score(
    pillar_scores: pd.DataFrame, config: dict[str, Any] | None = None
) -> pd.Series:
    """Vectorised FHS from a frame of ``<pillar_key>_score`` columns.

    Any pillar that could not be scored has its weight redistributed over the
    rest, mirroring the driver-level rule one layer down.
    """
    cfg = config if config is not None else load_config()
    weights = cfg["pillars"]["fhs_weights"]
    keys = pillar_keys(cfg)

    scores = pillar_scores[[f"{k}_score" for k in keys]]
    nominal = scores.notna().astype("float64") * [float(weights[k]) for k in keys]
    total = nominal.sum(axis=1)
    effective = nominal.div(total.where(total > 0), axis=0).fillna(0.0)
    value = (scores.fillna(0.0).to_numpy() * effective.to_numpy()).sum(axis=1)
    return pd.Series(value, index=scores.index, name="financial_health_score").clip(
        SCORE_MIN, SCORE_MAX
    )


def score_frame(frame: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Feature frame -> six pillar scores + FHS + grade. The batch scoring path."""
    cfg = config if config is not None else load_config()
    pillars = compute_pillars(frame, cfg)
    pillars["financial_health_score"] = compute_health_score(pillars, cfg)
    grades = pillars["financial_health_score"].map(lambda v: grade_for(float(v), cfg))
    pillars["grade"] = [g.grade for g in grades]
    pillars["grade_label"] = [g.label for g in grades]
    logger.info(
        "Health scores computed",
        rows=len(pillars),
        fhs_mean=round(float(pillars["financial_health_score"].mean()), 3),
        grade_mix=pillars["grade"].value_counts().to_dict(),
    )
    return pillars


def explain_health_score(
    pillar_results: dict[str, PillarResult], config: dict[str, Any] | None = None
) -> HealthScore:
    """Assemble the FHS from decomposed pillars, keeping the arithmetic exact."""
    cfg = config if config is not None else load_config()
    weights = cfg["pillars"]["fhs_weights"]

    available = {k: r for k, r in pillar_results.items() if r.score is not None}
    total_weight = sum(float(weights[k]) for k in available if k in weights)
    if total_weight <= 0:
        raise ValueError("no pillar carries FHS weight — check pillars.fhs_weights")

    shares: list[PillarShare] = []
    for key in pillar_keys(cfg):
        result = available.get(key)
        if result is None:
            continue
        weight = float(weights[key]) / total_weight
        shares.append(
            PillarShare(
                key=key,
                label=result.label,
                score=round(result.score, 4),
                weight=round(weight, 6),
                contribution=round(result.score * weight, 4),
            )
        )

    value = round(min(max(sum(s.contribution for s in shares), SCORE_MIN), SCORE_MAX), 4)
    grade = grade_for(value, cfg)
    return HealthScore(
        value=value, grade=grade.grade, grade_label=grade.label, pillar_shares=shares
    )
