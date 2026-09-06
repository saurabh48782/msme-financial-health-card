"""Health card retrieval.

There is no ``/refresh`` endpoint and no version history: cards are not stored,
so ``GET`` already re-scores from the current model, policy and rubric versions
on every call. An endpoint that re-did what ``GET`` does anyway would be surface
area pretending to be a capability.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Query, status

from src.api.deps import ConfigDep, ServiceDep
from src.schemas.healthcard import HealthCard, HealthCardPage
from src.service.healthcard_service import MSMENotFoundError
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["health card"])


@router.get("/healthcards", response_model=HealthCardPage, summary="List scored MSMEs")
async def list_cards(
    service: ServiceDep,
    config: ConfigDep,
    page: int = Query(1, ge=1),
    page_size: int | None = Query(None, ge=1, le=200),
    grade: str | None = None,
    risk_band: str | None = None,
    eligible: bool | None = None,
    customer_segment: str | None = None,
    industry: str | None = None,
    location_category: str | None = None,
) -> HealthCardPage:
    filters = {
        "grade": grade,
        "risk_band": risk_band,
        "eligible": eligible,
        "customer_segment": customer_segment,
        "industry": industry,
        "location_category": location_category,
    }
    return await service.list_cards(
        page=page,
        page_size=page_size or int(config["api"]["page_size"]),
        filters={k: v for k, v in filters.items() if v is not None},
    )


@router.get("/healthcard/{msme_id}", response_model=HealthCard, summary="Get a Health Card")
async def get_card(msme_id: str, service: ServiceDep) -> HealthCard:
    try:
        # Shielded: a client that disconnects mid-score should not leave the
        # scoring stack cancelled half-way through a card.
        return await asyncio.shield(service.get_card(msme_id))
    except MSMENotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
