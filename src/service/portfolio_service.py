"""Portfolio analytics - the inclusion story, computed.

Consumes whatever frame the store hands back rather than querying it directly,
so the aggregation is a pure function of the scored portfolio and is testable
without any store at all.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data_access.csv_store import CsvCardStore
from src.schemas.portfolio import Bucket, PortfolioSummary, SegmentLift
from src.scoring.anomaly import SEVERITY_ORDER
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

HISTOGRAM_BUCKET = 5
_TOP_LEVELS = 12
_SEGMENT_COLUMNS = ("Customer_Segment", "customer_segment")
_INDUSTRY_COLUMNS = ("Industry", "industry")
_LOCATION_COLUMNS = ("Location_Category", "location_category")


def _column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Accept either the dataset's column naming or the scored-output naming."""
    return next((c for c in candidates if c in frame.columns), None)


def _buckets(series: pd.Series, *, limit: int | None = None) -> list[Bucket]:
    counts = series.value_counts(dropna=True)
    if limit is not None:
        counts = counts.head(limit)
    total = int(series.notna().sum()) or 1
    return [
        Bucket(label=str(label), count=int(count), share=round(int(count) / total, 6))
        for label, count in counts.items()
    ]


class PortfolioService:
    def __init__(self, store: CsvCardStore, config: dict[str, Any] | None = None) -> None:
        self.store = store
        self.config = config if config is not None else load_config()

    async def summary(self) -> PortfolioSummary:
        frame: pd.DataFrame = await self.store.portfolio_frame()
        if frame is None or frame.empty:
            logger.warning("Portfolio summary requested with no scored firms")
            return PortfolioSummary(
                firms=0,
                mean_health_score=0.0,
                median_health_score=0.0,
                approval_rate=0.0,
                total_sanctioned_inr=0.0,
                mean_probability_of_default=0.0,
                thin_file_share=0.0,
            )

        fhs = pd.to_numeric(frame["financial_health_score"], errors="coerce")
        eligible = frame["eligible"].astype(bool)
        limits = pd.to_numeric(frame["credit_limit_inr"], errors="coerce").fillna(0.0)
        pd_values = pd.to_numeric(frame["probability_of_default"], errors="coerce")
        thin = frame["thin_file"].astype(bool) if "thin_file" in frame.columns else None

        histogram_bins = (np.floor(fhs / HISTOGRAM_BUCKET) * HISTOGRAM_BUCKET).astype("Int64")
        histogram = [
            Bucket(
                label=f"{int(edge)}-{int(edge) + HISTOGRAM_BUCKET}",
                count=int(count),
                share=round(int(count) / len(frame), 6),
            )
            for edge, count in sorted(histogram_bins.value_counts().items())
        ]

        segment_column = _column(frame, _SEGMENT_COLUMNS)
        lifts: list[SegmentLift] = []
        if segment_column:
            for segment, group in frame.groupby(segment_column, observed=True):
                group_eligible = group["eligible"].astype(bool)
                group_limits = pd.to_numeric(group["credit_limit_inr"], errors="coerce").fillna(0.0)
                lifts.append(
                    SegmentLift(
                        segment=str(segment),
                        firms=len(group),
                        approval_rate=round(float(group_eligible.mean()), 6),
                        mean_health_score=round(
                            float(pd.to_numeric(group["financial_health_score"]).mean()), 3
                        ),
                        mean_probability_of_default=round(
                            float(pd.to_numeric(group["probability_of_default"]).mean()), 6
                        ),
                        median_credit_limit_inr=round(
                            float(group_limits[group_eligible].median() or 0.0), 2
                        ),
                        thin_file_share=round(
                            float(group["thin_file"].astype(bool).mean())
                            if "thin_file" in group.columns
                            else 0.0,
                            6,
                        ),
                    )
                )
            lifts.sort(key=lambda lift: lift.segment)

        flag_mix: list[Bucket] = []
        flag_columns = [c for c in frame.columns if c.startswith("flag_") and c != "flag_count"]
        if flag_columns:
            total = len(frame)
            counts = {
                column.removeprefix("flag_"): int(frame[column].astype(bool).sum())
                for column in flag_columns
            }
            flag_mix = [
                Bucket(label=name, count=count, share=round(count / total, 6))
                for name, count in sorted(counts.items(), key=lambda kv: -kv[1])
                if count
            ]

        decline_mix: list[Bucket] = []
        if "knockout_codes" in frame.columns:
            declined = frame.loc[~eligible, "knockout_codes"].astype("string").fillna("")
            exploded = declined[declined != ""].str.split(",").explode()
            decline_mix = _buckets(exploded)

        industry_column = _column(frame, _INDUSTRY_COLUMNS)
        location_column = _column(frame, _LOCATION_COLUMNS)
        scored_at = None
        if "scored_at" in frame.columns and len(frame):
            scored_at = str(frame["scored_at"].max())

        summary = PortfolioSummary(
            firms=len(frame),
            scored_at=scored_at,
            mean_health_score=round(float(fhs.mean()), 3),
            median_health_score=round(float(fhs.median()), 3),
            approval_rate=round(float(eligible.mean()), 6),
            total_sanctioned_inr=round(float(limits.sum()), 2),
            mean_probability_of_default=round(float(pd_values.mean()), 6),
            thin_file_share=round(float(thin.mean()) if thin is not None else 0.0, 6),
            health_score_histogram=histogram,
            grade_mix=_buckets(frame["grade"]) if "grade" in frame.columns else [],
            risk_band_mix=_buckets(frame["risk_band"]) if "risk_band" in frame.columns else [],
            segment_lift=lifts,
            industry_mix=_buckets(frame[industry_column], limit=_TOP_LEVELS)
            if industry_column
            else [],
            location_mix=_buckets(frame[location_column]) if location_column else [],
            flag_mix=flag_mix,
            decline_reason_mix=decline_mix,
        )
        logger.info(
            "Portfolio summary built",
            firms=summary.firms,
            approval_rate=summary.approval_rate,
            segments=len(summary.segment_lift),
        )
        return summary

    async def anomaly_queue(self, limit: int = 100) -> pd.DataFrame:
        """Flagged firms, most severe first, then most flags.

        Ordered by something a reviewer can act on: the severity of the worst
        broken invariant, then how many broke. Ties on both are broken by the
        weakest health score, so the queue starts where the risk is.
        """
        frame: pd.DataFrame = await self.store.portfolio_frame()
        if frame is None or frame.empty:
            return pd.DataFrame()
        flagged = frame[frame["flag_count"] > 0] if "flag_count" in frame.columns else frame
        flagged = flagged.assign(
            severity_rank=flagged.get("max_severity", pd.Series(dtype="object"))
            .map(SEVERITY_ORDER)
            .fillna(0)
        )
        ascending = {"severity_rank": False, "flag_count": False, "financial_health_score": True}
        sort_columns = [c for c in ascending if c in flagged.columns]
        if sort_columns:
            flagged = flagged.sort_values(
                sort_columns, ascending=[ascending[c] for c in sort_columns]
            )
        return flagged.head(limit)
