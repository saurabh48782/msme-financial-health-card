"""SHAP drivers, the pillar waterfall and plain-English reason codes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from src.api.deps import ServiceDep
from src.schemas.healthcard import ExplanationOut, PillarScore
from src.service.healthcard_service import MSMENotFoundError
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["explainability"])


@router.get(
    "/explain/{msme_id}",
    response_model=ExplanationOut,
    summary="Why this MSME scored what it scored",
)
async def explain(msme_id: str, service: ServiceDep) -> ExplanationOut:
    try:
        card = await service.get_card(msme_id)
    except MSMENotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    if card.explanation is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No explanation is available for this card",
        )
    return card.explanation


@router.get(
    "/explain/{msme_id}/pillars",
    response_model=list[PillarScore],
    summary="Exact pillar decomposition — contributions sum to the score",
)
async def explain_pillars(msme_id: str, service: ServiceDep) -> list[PillarScore]:
    try:
        card = await service.get_card(msme_id)
    except MSMENotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    return card.pillars
