"""Health Card response contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Severity = Literal["low", "medium", "high", "critical"]


class DriverBreakdown(BaseModel):
    """One rubric driver's exact contribution, with the value behind it."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(examples=["gst_filing_timeliness"])
    label: str = Field(examples=["GST filing timeliness"])
    value: float | None = Field(default=None, examples=[88.4])
    formatted_value: str | None = Field(default=None, examples=["88.4%"])
    driver_score: float = Field(ge=0.0, le=100.0, examples=[66.9])
    weight: float = Field(ge=0.0, le=1.0, examples=[0.6855])
    contribution: float = Field(examples=[17.53])
    available: bool = Field(
        default=True,
        description="False when the input is structurally absent; its weight was redistributed",
    )


class PillarScore(BaseModel):
    """One of the six sub-scores. ``baseline + sum(drivers) == score``."""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(examples=["compliance"])
    label: str = Field(examples=["Compliance"])
    score: float = Field(ge=0.0, le=100.0, examples=[85.86])
    weight: float = Field(ge=0.0, le=1.0, description="Share of the Financial Health Score")
    contribution: float = Field(description="Points this pillar adds to the FHS")
    baseline: float = Field(description="Fitted peer-market baseline before driver effects")
    thin_file: bool = Field(default=False, description="Scored without a formal repayment record")
    drivers: list[DriverBreakdown] = Field(default_factory=list)


class RiskFlagOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(examples=["gst_bank_mismatch"])
    severity: Severity = Field(examples=["medium"])
    message: str = Field(examples=["Declared GST sales differ materially from bank credits"])
    metric: str = Field(examples=["gst_bank_reconciliation_gap"])
    value: float | None = Field(default=None, examples=[0.27])
    threshold: float = Field(examples=[0.2])


class ReasonCodeOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["pillar", "model"] = Field(examples=["pillar"])
    code: str = Field(examples=["compliance.gst_filing_timeliness"])
    pillar: str | None = Field(default=None, examples=["compliance"])
    text: str = Field(examples=["Weak GST filing timeliness (71.0%) cost 6.4 points on Compliance"])
    impact: float = Field(examples=[6.4])
    unit: Literal["points", "pp_default_probability"] = Field(examples=["points"])
    direction: Literal["positive", "negative"] = Field(examples=["negative"])
    formatted_value: str | None = Field(default=None, examples=["71.0%"])


class ShapContribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature: str = Field(examples=["Overdraft_Usage_Ratio"])
    label: str = Field(examples=["Overdraft usage"])
    value: float | None = Field(default=None, examples=[0.41])
    shap: float = Field(examples=[0.0183])
    shap_pp: float = Field(examples=[1.83], description="Percentage points of default probability")


class ExplanationOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason_codes: list[ReasonCodeOut] = Field(default_factory=list)
    shap_base_value: float | None = Field(default=None)
    shap_contributions: list[ShapContribution] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class CreditRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    eligible: bool = Field(examples=[True])
    risk_band: Literal["Low", "Medium", "High"] = Field(examples=["Medium"])
    probability_of_default: float = Field(ge=0.0, le=1.0, examples=[0.087])
    credit_limit_inr: float = Field(ge=0.0, examples=[388_000.0])
    tenor_months: int = Field(ge=0, examples=[24])
    indicative_rate_pct: float = Field(ge=0.0, examples=[13.5])
    decline_reasons: list[str] = Field(default_factory=list)
    model_eligibility_probability: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "The eligibility classifier's own probability. Kept alongside the policy "
            "decision so a disagreement (policy declines, model is confident) can be "
            "routed to manual review instead of vanishing."
        ),
    )
    limit_basis: dict[str, float] = Field(
        default_factory=dict, description="Every number used to size the limit"
    )
    notes: list[str] = Field(default_factory=list)
    policy_version: str = Field(examples=["policy-1.0.0"])


class HealthCard(BaseModel):
    """The deliverable: one MSME's complete, explainable Financial Health Card."""

    model_config = ConfigDict(extra="forbid")

    msme_id: str = Field(examples=["MSME0000001"])
    financial_health_score: float = Field(ge=0.0, le=100.0, examples=[76.14])
    grade: str = Field(examples=["A"])
    grade_label: str = Field(examples=["Strong"])
    customer_segment: str | None = Field(default=None, examples=["NTC"])
    industry: str | None = Field(default=None, examples=["Retail"])
    location_category: str | None = Field(default=None, examples=["Tier 2"])
    pillars: list[PillarScore] = Field(default_factory=list)
    credit: CreditRecommendation
    risk_flags: list[RiskFlagOut] = Field(default_factory=list)
    explanation: ExplanationOut | None = Field(default=None)
    thin_file: bool = Field(
        default=False, description="No formal credit history; scored on alternative data alone"
    )

    # --- provenance: what a digital-lending audit asks for -------------------
    model_version: str | None = Field(default=None, examples=["3"])
    policy_version: str = Field(examples=["policy-1.0.0"])
    rubric_version: str = Field(examples=["rubric-1.0.0"])
    feature_snapshot_hash: str = Field(
        examples=["7f3c1e9a2b4d5f60"],
        description="Hash of the exact inputs used, so a card is reproducible",
    )
    scored_at: datetime
    version: int = Field(
        default=1,
        ge=1,
        description=(
            "Card version. Always 1 in this deployment: cards are scored on demand "
            "rather than persisted, so there is no version history to walk."
        ),
    )
    consent_artifact_id: str | None = Field(
        default=None,
        description=(
            "The Account Aggregator consent this card was produced under, so a "
            "decision can be traced to the authorisation for it. Carried through "
            "the scoring path but unpopulated here — the synthetic dataset has no "
            "consent layer."
        ),
    )

    @property
    def summary(self) -> dict[str, Any]:
        """Compact form for list views and log lines."""
        return {
            "msme_id": self.msme_id,
            "fhs": self.financial_health_score,
            "grade": self.grade,
            "risk_band": self.credit.risk_band,
            "eligible": self.credit.eligible,
            "limit_inr": self.credit.credit_limit_inr,
        }


class HealthCardSummary(BaseModel):
    """Row shape for the paginated card list."""

    model_config = ConfigDict(extra="forbid")

    msme_id: str
    financial_health_score: float
    grade: str
    risk_band: str
    eligible: bool
    credit_limit_inr: float
    probability_of_default: float
    customer_segment: str | None = None
    industry: str | None = None
    location_category: str | None = None
    flag_count: int = 0
    thin_file: bool = False
    version: int = 1
    scored_at: datetime | None = None


class HealthCardPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[HealthCardSummary] = Field(default_factory=list)
    total: int = Field(ge=0)
    page: int = Field(ge=1)
    page_size: int = Field(ge=1)

    @property
    def pages(self) -> int:
        return max(1, -(-self.total // self.page_size))
