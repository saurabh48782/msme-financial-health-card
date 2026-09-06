"""Stateless scoring. Nothing is persisted; the caller owns the payload."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, status

from src.api.deps import ConfigDep, ServiceDep
from src.schemas.healthcard import HealthCard
from src.schemas.msme import ScoreRequest
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["scoring"])


@router.post(
    "/score",
    response_model=list[HealthCard],
    summary="Score one or more MSMEs without persisting anything",
)
async def score(request: ScoreRequest, service: ServiceDep, config: ConfigDep) -> list[HealthCard]:
    limit = int(config["api"]["max_batch_size"])
    if len(request.records) > limit:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Batch of {len(request.records)} exceeds the maximum of {limit}",
        )
    # Scoring is CPU-bound (trees, SHAP, pandas). Off the event loop it goes, or a
    # single 500-record batch stalls every other request in the worker.
    cards = await asyncio.to_thread(
        service.score_records,
        request.records,
        include_explanation=request.include_explanation,
    )
    logger.info("Batch scored", records=len(request.records), cards=len(cards))
    return cards
