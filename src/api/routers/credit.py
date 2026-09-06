"""The credit recommendation on its own, for callers that only need the decision."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, status

from src.api.deps import ServiceDep
from src.schemas.credit import PolicyDecisionOut
from src.schemas.healthcard import CreditRecommendation
from src.service.healthcard_service import MSMENotFoundError
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["credit"])


@router.get(
    "/credit-recommendation/{msme_id}",
    response_model=PolicyDecisionOut,
    summary="Credit decision with its knock-out reasons",
)
async def credit_recommendation(msme_id: str, service: ServiceDep) -> PolicyDecisionOut:
    try:
        card = await service.get_card(msme_id)
    except MSMENotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error

    decision: CreditRecommendation = card.credit
    return PolicyDecisionOut(
        msme_id=card.msme_id,
        decision=decision,
        knockout_codes=[reason for reason in decision.decline_reasons],
        evaluated_at=datetime.now(UTC),
        policy_version=card.policy_version,
    )
