"""Credit decision contracts beyond the card itself: simulation and audit views."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.healthcard import CreditRecommendation, PillarScore


class PolicyDecisionOut(BaseModel):
    """A standalone, auditable record of one credit decision."""

    model_config = ConfigDict(extra="forbid")

    msme_id: str
    decision: CreditRecommendation
    knockout_codes: list[str] = Field(default_factory=list)
    evaluated_at: datetime
    policy_version: str


class SimulationDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: str = Field(examples=["financial_health_score"])
    before: float
    after: float
    delta: float


class SimulationResult(BaseModel):
    """Before/after for a what-if, so an officer can advise rather than only decline."""

    model_config = ConfigDict(extra="forbid")

    msme_id: str
    overrides: dict[str, float | int | str] = Field(default_factory=dict)
    deltas: list[SimulationDelta] = Field(default_factory=list)
    before_grade: str
    after_grade: str
    before_pillars: list[PillarScore] = Field(default_factory=list)
    after_pillars: list[PillarScore] = Field(default_factory=list)
    before_credit: CreditRecommendation
    after_credit: CreditRecommendation
    unchanged: bool = Field(
        default=False, description="True when the overrides moved nothing material"
    )
