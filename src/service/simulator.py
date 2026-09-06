"""What-if: perturb inputs, re-score, report the delta.

This turns a decline into advice. "You were declined" is a dead end; "raising GST
filing timeliness from 71% to 95% moves you from High to Medium risk and unlocks
a Rs 2.4 lakh limit" is something a firm can act on — and it is exactly what the
exact pillar decomposition makes computable rather than guessed.

Overrides are validated against the feature allowlist, so a simulation cannot
inject an arbitrary column into the scoring frame.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.schemas.credit import SimulationDelta, SimulationResult
from src.schemas.healthcard import HealthCard
from src.scoring.scoring_orchestration import ScoringBundle, score_one
from src.utils.config import load_config
from src.utils.custom_error import CustomError
from src.utils.logger import get_logger

logger = get_logger(__name__)

_TRACKED_METRICS = (
    "financial_health_score",
    "probability_of_default",
    "credit_limit_inr",
    "tenor_months",
    "indicative_rate_pct",
)
_MATERIAL_DELTA = 1e-6


class Simulator:
    def __init__(self, bundle: ScoringBundle, config: dict[str, Any] | None = None) -> None:
        self.bundle = bundle
        self.config = config if config is not None else load_config()
        self.allowed = set(self.config["schema"]["feature_columns"])
        self.max_overrides = int(self.config["api"]["simulate_max_overrides"])

    def validate_overrides(self, overrides: dict[str, Any]) -> dict[str, Any]:
        """Reject unknown columns loudly rather than silently ignoring them."""
        if len(overrides) > self.max_overrides:
            raise CustomError(
                f"too many overrides: {len(overrides)} (maximum {self.max_overrides})"
            )
        unknown = sorted(set(overrides) - self.allowed)
        if unknown:
            raise CustomError(
                f"unknown feature(s) in overrides: {unknown}. "
                f"Use dataset column names from schema.feature_columns."
            )
        return overrides

    def simulate(
        self, profile: dict[str, Any], overrides: dict[str, Any], msme_id: str
    ) -> SimulationResult:
        self.validate_overrides(overrides)
        id_column = self.config["schema"]["id_column"]

        base_row = dict(profile)
        base_row.setdefault(id_column, msme_id)
        after_row = {**base_row, **overrides}

        before = score_one(pd.Series(base_row), self.bundle, self.config)
        after = score_one(pd.Series(after_row), self.bundle, self.config)

        deltas = [
            SimulationDelta(
                metric=metric,
                before=round(_metric_of(before, metric), 6),
                after=round(_metric_of(after, metric), 6),
                delta=round(_metric_of(after, metric) - _metric_of(before, metric), 6),
            )
            for metric in _TRACKED_METRICS
        ]
        unchanged = all(abs(delta.delta) < _MATERIAL_DELTA for delta in deltas)

        logger.info(
            "Simulation run",
            msme_id=msme_id,
            overrides=sorted(overrides),
            fhs_delta=deltas[0].delta,
            band_before=before.credit.risk_band,
            band_after=after.credit.risk_band,
        )
        return SimulationResult(
            msme_id=msme_id,
            overrides=overrides,
            deltas=deltas,
            before_grade=before.grade,
            after_grade=after.grade,
            before_pillars=before.pillars,
            after_pillars=after.pillars,
            before_credit=before.credit,
            after_credit=after.credit,
            unchanged=unchanged,
        )


def _metric_of(card: HealthCard, metric: str) -> float:
    if metric == "financial_health_score":
        return float(card.financial_health_score)
    return float(getattr(card.credit, metric))
