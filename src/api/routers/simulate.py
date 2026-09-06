"""What-if simulation: turn a decline into advice."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, status

from src.api.deps import ServiceDep, SimulatorDep
from src.schemas.credit import SimulationResult
from src.schemas.msme import SimulationRequest
from src.utils.custom_error import CustomError
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["simulation"])


@router.post(
    "/simulate/{msme_id}",
    response_model=SimulationResult,
    summary="Re-score with overridden inputs and report the before/after delta",
)
async def simulate(
    msme_id: str,
    request: SimulationRequest,
    service: ServiceDep,
    simulator: SimulatorDep,
) -> SimulationResult:
    profile = await service.store.get_profile(msme_id)
    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"no MSME profile stored for {msme_id}"
        )
    try:
        # Two full scoring passes; keep them off the event loop.
        return await asyncio.to_thread(
            simulator.simulate, profile, dict(request.overrides), msme_id
        )
    except CustomError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
        ) from error
