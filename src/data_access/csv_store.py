"""The card store: read-only, backed by the batch-scoring artifacts.

Why this shape: the whole submission - dashboard, portfolio analytics, a
drillable card for any of 50,000 firms - works on a fresh clone with ``uv sync``
and two pipeline commands. The portfolio listing comes from
``portfolio_scores.csv``; a single card is scored on demand from the raw feature
row, so nothing has to be persisted for every firm to be inspectable.

Cards are therefore not stored, and there is no version history. What makes a
decision reproducible instead is that every card carries the ``model_version``,
``policy_version``, ``rubric_version`` and ``feature_snapshot_hash`` it was
produced under - re-running those inputs reproduces the card exactly. A
write-through versioned store is the obvious production addition; see
``docs/BEYOND_SCOPE.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.schemas.healthcard import HealthCardPage, HealthCardSummary
from src.utils.config import load_config, portfolio_scores_path, raw_csv_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SORT_COLUMN = "financial_health_score"


class CsvCardStore:
    """Portfolio listing from the batch scores CSV; profiles from the raw dataset."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        scores: pd.DataFrame | None = None,
        profiles: pd.DataFrame | None = None,
    ) -> None:
        self.config = config if config is not None else load_config()
        self.id_column = self.config["schema"]["id_column"]
        self._scores = scores if scores is not None else self._load_scores()
        self._profiles = profiles if profiles is not None else self._load_profiles()
        if self._profiles is not None and self.id_column in self._profiles.columns:
            self._profiles = self._profiles.set_index(self.id_column, drop=False)

    # -- loading -------------------------------------------------------------
    def _load_scores(self) -> pd.DataFrame:
        path = portfolio_scores_path(self.config)
        if not Path(path).is_file():
            logger.warning(
                "No portfolio scores CSV — the portfolio views will be empty",
                path=str(path),
                hint="run python -m src.model.batch_inference",
            )
            return pd.DataFrame()
        frame = pd.read_csv(path)
        logger.info("Portfolio scores loaded", path=str(path), rows=len(frame))
        return frame

    def _load_profiles(self) -> pd.DataFrame | None:
        path = raw_csv_path(self.config)
        if not Path(path).is_file():
            logger.warning("No raw dataset — on-demand card scoring is unavailable")
            return None
        columns = [self.id_column, *self.config["schema"]["feature_columns"]]
        frame = pd.read_csv(path, usecols=columns)
        logger.info("MSME profiles loaded", path=str(path), rows=len(frame))
        return frame

    # -- reads ---------------------------------------------------------------
    async def get_profile(self, msme_id: str) -> dict[str, Any] | None:
        if self._profiles is None or msme_id not in self._profiles.index:
            return None
        row = self._profiles.loc[msme_id]
        if isinstance(row, pd.DataFrame):  # duplicate ids: take the first
            row = row.iloc[0]
        return {str(k): (None if pd.isna(v) else v) for k, v in row.items()}

    async def list_cards(
        self, *, page: int = 1, page_size: int = 25, filters: dict[str, Any] | None = None
    ) -> HealthCardPage:
        frame = self._filtered(filters)
        total = len(frame)
        if _SORT_COLUMN in frame.columns:
            frame = frame.sort_values(_SORT_COLUMN, ascending=False)
        start = (max(page, 1) - 1) * page_size
        window = frame.iloc[start : start + page_size]
        return HealthCardPage(
            items=[self._summary(row) for _, row in window.iterrows()],
            total=total,
            page=max(page, 1),
            page_size=page_size,
        )

    def _filtered(self, filters: dict[str, Any] | None) -> pd.DataFrame:
        frame = self._scores
        if frame.empty or not filters:
            return frame
        mask = pd.Series(True, index=frame.index)
        for key, value in filters.items():
            if value is None:
                continue
            column = {
                "customer_segment": "Customer_Segment",
                "industry": "Industry",
                "location_category": "Location_Category",
            }.get(key, key)
            if column not in frame.columns:
                continue
            mask &= (
                frame[column].astype("string") == str(value)
                if not isinstance(value, bool)
                else frame[column].astype(bool) == value
            )
        return frame[mask]

    def _summary(self, row: pd.Series) -> HealthCardSummary:
        return HealthCardSummary(
            msme_id=str(row[self.id_column]),
            financial_health_score=float(row.get("financial_health_score", 0.0)),
            grade=str(row.get("grade", "")),
            risk_band=str(row.get("risk_band", "")),
            eligible=bool(row.get("eligible", False)),
            credit_limit_inr=float(row.get("credit_limit_inr", 0.0)),
            probability_of_default=float(row.get("probability_of_default", 0.0)),
            customer_segment=_optional(row.get("Customer_Segment")),
            industry=_optional(row.get("Industry")),
            location_category=_optional(row.get("Location_Category")),
            flag_count=int(row.get("flag_count", 0) or 0),
            thin_file=bool(row.get("thin_file", False)),
            version=1,
        )

    async def portfolio_frame(self) -> pd.DataFrame:
        return self._scores

    async def health(self) -> bool:
        return not self._scores.empty


def _optional(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(value)
