"""Scoring - the layer the routers actually call.

Nothing here imports FastAPI. The routers translate HTTP into these calls and
these results back into HTTP, which is what lets the same logic serve the JSON
API, the HTML views and the batch scorer without three implementations of same thing.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.data_access.csv_store import CsvCardStore
from src.schemas.healthcard import HealthCard, HealthCardPage
from src.schemas.msme import MSMEFeatures
from src.scoring.scoring_orchestration import ScoringBundle, score_many, score_one
from src.utils.config import load_config
from src.utils.custom_error import CustomError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MSMENotFoundError(CustomError):
    """No stored profile to score from."""


class HealthCardService:
    """Orchestrates scoring for one deployment."""

    def __init__(
        self,
        store: CsvCardStore,
        bundle: ScoringBundle,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.store = store
        self.bundle = bundle
        self.config = config if config is not None else load_config()
        self.id_column = self.config["schema"]["id_column"]

    # -- stateless scoring ---------------------------------------------------
    def score_records(
        self,
        records: list[MSMEFeatures],
        *,
        include_explanation: bool = True,
    ) -> list[HealthCard]:
        """Score a batch without touching the store.

        A single record takes the per-card path so it gets a SHAP explanation; a
        batch skips explanations by default because a SHAP call per row dominates
        the cost of 500 records.
        """
        if not records:
            return []
        frame = pd.DataFrame([record.to_row() for record in records])
        if len(records) == 1:
            return [
                score_one(
                    frame.iloc[0],
                    self.bundle,
                    self.config,
                    include_explanation=include_explanation,
                )
            ]
        return score_many(frame, self.bundle, self.config, include_explanation=include_explanation)

    # -- one stored firm -----------------------------------------------------
    async def get_card(self, msme_id: str) -> HealthCard:
        """Score one of the stored firms on demand.

        The on-demand path is what makes the read-only deployment useful: every
        one of the 50,000 firms is drillable without a write-capable database,
        and a card is always produced by the current model and policy version
        rather than served from a stale row.
        """
        profile = await self.store.get_profile(msme_id)
        if profile is None:
            raise MSMENotFoundError(f"no MSME profile stored for {msme_id}")
        profile.setdefault(self.id_column, msme_id)
        return score_one(
            pd.Series(profile),
            self.bundle,
            self.config,
            include_explanation=True,
        )

    async def list_cards(
        self, *, page: int = 1, page_size: int = 25, filters: dict[str, Any] | None = None
    ) -> HealthCardPage:
        return await self.store.list_cards(page=page, page_size=page_size, filters=filters)

    async def health(self) -> bool:
        return await self.store.health()
