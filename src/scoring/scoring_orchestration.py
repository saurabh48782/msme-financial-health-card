"""Assemble a complete Health Card: features -> pillars -> ML -> anomaly -> policy -> explain.

This module is the single scoring path. The API, the batch portfolio scorer, the
MLflow ``pyfunc`` wrapper and the what-if simulator all call in here, which is the
only way to guarantee a firm gets the same card in real time as it did in the batch.

Every stage is wrapped in ``traced_stage`` and a slow card names the layer that was slow.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.data.data_orchestration import build_features
from src.schemas.healthcard import (
    CreditRecommendation,
    DriverBreakdown,
    ExplanationOut,
    HealthCard,
    PillarScore,
    RiskFlagOut,
)
from src.scoring.anomaly import flags_for_row
from src.scoring.explainer import ShapExplainer, build_explanation, format_value
from src.scoring.health_score import explain_health_score, grade_for
from src.scoring.pillars import explain_pillars, pillar_keys
from src.scoring.policy import decide
from src.utils.config import load_config
from src.utils.custom_error import CustomError
from src.utils.logger import get_logger
from src.utils.tracing import trace_stage

if TYPE_CHECKING:  # pragma: no cover
    from src.model.components import TrainedModel

logger = get_logger(__name__)


@dataclass
class ScoringBundle:
    """Every fitted artifact the scoring path needs, loaded once.

    Held together in one object because they are only valid *as a set*: a PD model
    trained on one winsorisation and served with another is silently wrong.
    """

    winsor_limits: dict[str, dict[str, float]] = field(default_factory=dict)
    pd_model: TrainedModel | None = None
    eligibility_model: TrainedModel | None = None
    shap_explainer: ShapExplainer | None = None
    model_version: str | None = None

    @property
    def has_ml(self) -> bool:
        return self.pd_model is not None


def feature_snapshot_hash(row: dict[str, Any], config: dict[str, Any] | None = None) -> str:
    """Stable hash of the inputs that produced a card.

    Only the allowlisted inputs are hashed, in sorted order, with floats rounded —
    so a card is reproducible and a changed input is detectable, while harmless
    float noise does not invent a new snapshot on every re-score.
    """
    cfg = config if config is not None else load_config()
    columns = [cfg["schema"]["id_column"], *cfg["schema"]["feature_columns"]]
    payload: dict[str, Any] = {}
    for column in columns:
        value = row.get(column)
        if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
            numeric = float(value)
            payload[column] = None if math.isnan(numeric) else round(numeric, 6)
        elif value is None or (isinstance(value, float) and pd.isna(value)):
            payload[column] = None
        else:
            payload[column] = str(value)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.blake2b(encoded, digest_size=8).hexdigest()


def prepare_features(
    records: pd.DataFrame,
    bundle: ScoringBundle,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Raw MSME rows -> the engineered feature frame, reusing the fitted artifacts."""
    cfg = config if config is not None else load_config()
    featured, _ = build_features(records, cfg, winsor_limits=bundle.winsor_limits or None)
    return featured


def _predict_pd(
    featured: pd.DataFrame, bundle: ScoringBundle, config: dict[str, Any]
) -> NDArray[np.float64]:
    """Calibrated PD from the model, or a monotone fallback from the FHS.

    The fallback matters: the dashboard and the policy engine must work before a
    model has ever been trained, otherwise nothing can be demonstrated end to end
    on a fresh checkout. Its coefficients are fitted and documented under
    ``model_params.pd_fallback``, and a card scored this way carries
    ``model_version = null`` so it can never be mistaken for a model prediction.
    """
    from src.model.components import predict_pd

    if bundle.pd_model is not None:
        raw = predict_pd(bundle.pd_model, featured, config)
        return np.clip(raw, 0.0, 1.0)

    from src.scoring.health_score import compute_health_score
    from src.scoring.pillars import compute_pillars

    spec = config["model_params"]["pd_fallback"]
    fhs = compute_health_score(compute_pillars(featured, config), config).to_numpy()
    estimate = np.exp(float(spec["slope"]) * fhs + float(spec["intercept"]))
    return np.clip(estimate, float(spec["pd_min"]), float(spec["pd_max"]))


def score_one(
    record: pd.Series | dict[str, Any],
    bundle: ScoringBundle,
    config: dict[str, Any] | None = None,
    *,
    include_explanation: bool = True,
    version: int = 1,
    consent_artifact_id: str | None = None,
    featured_row: pd.Series | None = None,
) -> HealthCard:
    """Score one MSME. ``featured_row`` skips feature building when already done."""
    cfg = config if config is not None else load_config()
    raw = dict(record)
    id_column = cfg["schema"]["id_column"]
    msme_id = str(raw.get(id_column) or "unknown")

    with trace_stage("features", msme_id=msme_id):
        if featured_row is None:
            featured = prepare_features(pd.DataFrame([raw]), bundle, cfg)
        else:
            featured = pd.DataFrame([dict(featured_row)])
        row = featured.iloc[0]

    with trace_stage("pillars", msme_id=msme_id) as span:
        pillar_results = explain_pillars(row, cfg)
        health = explain_health_score(pillar_results, cfg)
        span["fhs"] = health.value

    with trace_stage("ml_risk", msme_id=msme_id) as span:
        pd_value = float(_predict_pd(featured, bundle, cfg)[0])
        eligibility_probability: float | None = None
        if bundle.eligibility_model is not None:
            from src.model.components import predict_eligibility

            eligibility_probability = float(
                predict_eligibility(bundle.eligibility_model, featured, cfg)[0]
            )
        span["pd"] = round(pd_value, 6)

    with trace_stage("anomaly", msme_id=msme_id) as span:
        flags = flags_for_row(row)
        span["flags"] = len(flags)

    with trace_stage("policy", msme_id=msme_id) as span:
        cashflow = pillar_results["cashflow"].score if "cashflow" in pillar_results else 0.0
        decision = decide(row, pd_value, cashflow, flags, cfg)
        span["eligible"] = decision.eligible
        span["limit"] = decision.credit_limit_inr

    explanation: ExplanationOut | None = None
    if include_explanation:
        with trace_stage("explain", msme_id=msme_id) as span:
            shap_contributions: list[dict[str, Any]] = []
            base_value: float | None = None
            if bundle.shap_explainer is not None and bundle.pd_model is not None:
                from src.model.components import build_feature_matrix

                matrix, _, _ = build_feature_matrix(
                    featured, cfg, features=bundle.pd_model.features
                )
                shap_contributions, base_value = bundle.shap_explainer.explain_row(matrix)
            built = build_explanation(pillar_results, shap_contributions, base_value, cfg)
            explanation = ExplanationOut(
                reason_codes=[
                    {
                        "source": code.source,
                        "code": code.code,
                        "pillar": code.pillar,
                        "text": code.text,
                        "impact": code.impact,
                        "unit": code.unit,
                        "direction": code.direction,
                        "formatted_value": code.formatted_value,
                    }  # type: ignore[list-item]
                    for code in built.reason_codes
                ],
                shap_base_value=built.shap_base_value,
                shap_contributions=built.shap_contributions,  # type: ignore[arg-type]
                notes=built.notes,
            )
            span["reason_codes"] = len(explanation.reason_codes)

    shares = {share.key: share for share in health.pillar_shares}
    pillars = [
        PillarScore(
            key=key,
            label=result.label,
            score=result.score,
            weight=shares[key].weight if key in shares else 0.0,
            contribution=shares[key].contribution if key in shares else 0.0,
            baseline=result.baseline,
            thin_file=result.thin_file,
            drivers=[
                DriverBreakdown(
                    name=driver.name,
                    label=driver.label,
                    value=driver.value,
                    formatted_value=format_value(driver.source, driver.value),
                    driver_score=driver.driver_score,
                    weight=driver.weight,
                    contribution=driver.contribution,
                    available=driver.available,
                )
                for driver in result.contributions
            ],
        )
        for key, result in pillar_results.items()
    ]

    credit = CreditRecommendation(
        eligible=decision.eligible,
        risk_band=decision.risk_band,  # type: ignore[arg-type]
        probability_of_default=decision.probability_of_default,
        credit_limit_inr=decision.credit_limit_inr,
        tenor_months=decision.tenor_months,
        indicative_rate_pct=decision.indicative_rate_pct,
        decline_reasons=decision.decline_reasons,
        model_eligibility_probability=(
            round(eligibility_probability, 6) if eligibility_probability is not None else None
        ),
        limit_basis=decision.limit_basis,
        notes=decision.notes,
        policy_version=decision.policy_version,
    )

    card = HealthCard(
        msme_id=msme_id,
        financial_health_score=health.value,
        grade=health.grade,
        grade_label=health.grade_label,
        customer_segment=_as_str(raw.get("Customer_Segment")),
        industry=_as_str(raw.get("Industry")),
        location_category=_as_str(raw.get("Location_Category")),
        pillars=pillars,
        credit=credit,
        risk_flags=[
            RiskFlagOut(
                code=flag.code,
                severity=flag.severity,  # type: ignore[arg-type]
                message=flag.message,
                metric=flag.metric,
                value=flag.value,
                threshold=flag.threshold,
            )
            for flag in flags
        ],
        explanation=explanation,
        thin_file=not bool(row.get("has_repayment_history", False)),
        model_version=str(bundle.model_version) if bundle.model_version is not None else None,
        policy_version=str(cfg["policy_version"]),
        rubric_version=str(cfg["rubric_version"]),
        feature_snapshot_hash=feature_snapshot_hash(raw, cfg),
        scored_at=datetime.now(UTC),
        version=version,
        consent_artifact_id=consent_artifact_id,
    )
    logger.info("Card scored", **card.summary)
    return card


def _as_str(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(value)


def score_many(
    records: pd.DataFrame,
    bundle: ScoringBundle,
    config: dict[str, Any] | None = None,
    *,
    include_explanation: bool = False,
) -> list[HealthCard]:
    """Score a batch, building features once for the whole frame.

    Used by ``/api/v1/score`` for bulk requests. Explanations are off by default
    because a SHAP call per row dominates the cost of a 500-record batch.
    """
    cfg = config if config is not None else load_config()
    if records.empty:
        return []
    featured = prepare_features(records, bundle, cfg)
    if len(featured) != len(records):
        raise CustomError(
            f"feature pipeline changed row count: {len(records)} in, {len(featured)} out"
        )
    return [
        score_one(
            records.iloc[position],
            bundle,
            cfg,
            include_explanation=include_explanation,
            featured_row=featured.iloc[position],
        )
        for position in range(len(records))
    ]


def score_portfolio(
    records: pd.DataFrame,
    bundle: ScoringBundle,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Fully vectorised portfolio scoring — the 50k batch path.

    Returns one row per firm with scores, PD, band, decision and flag counts. This
    is the path behind the portfolio dashboard, and it deliberately does *not*
    produce explanations: a beeswarm over 50,000 firms is a training-time artifact,
    not a request-time one.
    """
    cfg = config if config is not None else load_config()
    from src.scoring.anomaly import detect
    from src.scoring.health_score import score_frame
    from src.scoring.policy import decide_frame

    id_column = cfg["schema"]["id_column"]

    with trace_stage("portfolio.features", rows=len(records)):
        featured = prepare_features(records, bundle, cfg)

    with trace_stage("portfolio.pillars", rows=len(featured)):
        scores = score_frame(featured, cfg)

    with trace_stage("portfolio.ml", rows=len(featured)) as span:
        pd_values = pd.Series(_predict_pd(featured, bundle, cfg), index=featured.index)
        span["mean_pd"] = round(float(pd_values.mean()), 6)

    with trace_stage("portfolio.anomaly", rows=len(featured)):
        anomalies = detect(featured)

    with trace_stage("portfolio.policy", rows=len(featured)):
        merged = featured.join(anomalies[["max_severity"]])
        decisions = decide_frame(merged, pd_values, scores["cashflow_score"], cfg)

    out = pd.concat(
        [
            featured[[id_column]].reset_index(drop=True),
            scores.reset_index(drop=True),
            decisions.reset_index(drop=True),
            # Every per-rule flag column travels too, not just the count: the
            # anomaly dashboard needs to say *which* rules fire across the book,
            # and a count alone cannot answer that.
            anomalies.reset_index(drop=True),
        ],
        axis=1,
    )
    out["thin_file"] = ~featured["has_repayment_history"].to_numpy()
    out["model_version"] = bundle.model_version
    out["rubric_version"] = str(cfg["rubric_version"])
    out["scored_at"] = datetime.now(UTC).isoformat()
    for column in ("Customer_Segment", "Industry", "Location_Category", "Annual_Turnover_INR"):
        if column in featured.columns:
            out[column] = featured[column].to_numpy()

    out["feature_snapshot_hash"] = [
        feature_snapshot_hash({str(k): v for k, v in row.items()}, cfg)
        for row in records.to_dict(orient="records")
    ]
    out["grade"] = out["financial_health_score"].map(lambda v: grade_for(float(v), cfg).grade)
    logger.info(
        "Portfolio scored",
        rows=len(out),
        approval_rate=round(float(out["eligible"].mean()), 4),
        mean_fhs=round(float(out["financial_health_score"].mean()), 3),
    )
    return out


def pillar_score_columns(config: dict[str, Any] | None = None) -> list[str]:
    cfg = config if config is not None else load_config()
    return [f"{key}_score" for key in pillar_keys(cfg)]
