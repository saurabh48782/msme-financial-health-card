"""Portfolio analytics contracts — the inclusion story, in numbers."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Bucket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    count: int = Field(ge=0)
    share: float = Field(ge=0.0, le=1.0)


class SegmentLift(BaseModel):
    """Approval rate per segment — the New-to-Credit inclusion claim, measured."""

    model_config = ConfigDict(extra="forbid")

    segment: str = Field(examples=["NTC"])
    firms: int = Field(ge=0)
    approval_rate: float = Field(ge=0.0, le=1.0)
    mean_health_score: float
    mean_probability_of_default: float
    median_credit_limit_inr: float
    thin_file_share: float = Field(ge=0.0, le=1.0)


class SliceMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slice: str = Field(examples=["Customer_Segment"])
    level: str = Field(examples=["NTC"])
    rows: int = Field(ge=0)
    metrics: dict[str, float] = Field(default_factory=dict)


class PortfolioSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    firms: int = Field(ge=0)
    scored_at: str | None = None
    mean_health_score: float
    median_health_score: float
    approval_rate: float = Field(ge=0.0, le=1.0)
    total_sanctioned_inr: float = Field(ge=0.0)
    mean_probability_of_default: float
    thin_file_share: float = Field(ge=0.0, le=1.0)
    health_score_histogram: list[Bucket] = Field(default_factory=list)
    grade_mix: list[Bucket] = Field(default_factory=list)
    risk_band_mix: list[Bucket] = Field(default_factory=list)
    segment_lift: list[SegmentLift] = Field(default_factory=list)
    industry_mix: list[Bucket] = Field(default_factory=list)
    location_mix: list[Bucket] = Field(default_factory=list)
    flag_mix: list[Bucket] = Field(default_factory=list)
    decline_reason_mix: list[Bucket] = Field(default_factory=list)


class ModelVersionInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    aliases: list[str] = Field(default_factory=list)
    run_id: str | None = None
    created_at: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    is_current: bool = False


class MetricsReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_version: str | None = None
    policy_version: str
    rubric_version: str
    pd_metrics: dict[str, float] = Field(default_factory=dict)
    eligibility_metrics: dict[str, float] = Field(default_factory=dict)
    pillar_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    fhs_metrics: dict[str, float] = Field(default_factory=dict)
    calibration_curve: list[dict[str, float]] = Field(default_factory=list)
    fairness: list[SliceMetrics] = Field(default_factory=list)
    thresholds: dict[str, float] = Field(default_factory=dict)
    generated_at: str | None = None
