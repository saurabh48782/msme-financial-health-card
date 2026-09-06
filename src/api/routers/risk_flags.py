"""Risk flags for one MSME and the portfolio-wide anomaly queue."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, status

from src.api.deps import PortfolioDep, ServiceDep
from src.schemas.healthcard import RiskFlagOut
from src.service.healthcard_service import MSMENotFoundError
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["risk"])


@router.get(
    "/risk-flags/{msme_id}", response_model=list[RiskFlagOut], summary="Risk flags for one MSME"
)
async def risk_flags(msme_id: str, service: ServiceDep) -> list[RiskFlagOut]:
    try:
        card = await service.get_card(msme_id)
    except MSMENotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    return card.risk_flags


@router.get("/anomalies", summary="Portfolio anomaly queue, most severe first")
async def anomalies(
    portfolio: PortfolioDep, limit: int = Query(100, ge=1, le=1000)
) -> list[dict[str, Any]]:
    frame = await portfolio.anomaly_queue(limit)
    if frame.empty:
        return []
    columns = [
        c
        for c in (
            "MSME_ID",
            "msme_id",
            "financial_health_score",
            "risk_band",
            "max_severity",
            "flag_count",
            "eligible",
            "Customer_Segment",
            "Industry",
        )
        if c in frame.columns
    ]
    return [{str(k): v for k, v in row.items()} for row in frame[columns].to_dict(orient="records")]
