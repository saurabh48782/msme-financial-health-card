"""Service orchestration: on-demand scoring, portfolio aggregation, what-if."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
import pytest

from src.data_access.csv_store import CsvCardStore
from src.service.healthcard_service import HealthCardService, MSMENotFoundError
from src.service.portfolio_service import PortfolioService
from src.service.simulator import Simulator
from src.utils.custom_error import CustomError
from tests.stubs import rubric_only_bundle


@pytest.fixture
def profile(raw_frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    columns = [config["schema"]["id_column"], *config["schema"]["feature_columns"]]
    return {k: (None if pd.isna(v) else v) for k, v in raw_frame.iloc[0][columns].items()}


@pytest.fixture
def service(profile: dict[str, Any], config: dict[str, Any]) -> HealthCardService:
    """The real store over one injected profile row — no parallel stub to drift."""
    store = CsvCardStore(config, scores=pd.DataFrame(), profiles=pd.DataFrame([profile]))
    return HealthCardService(store, rubric_only_bundle(), config)


class TestGetCard:
    async def test_a_stored_firm_is_scored_on_demand(
        self, service: HealthCardService, profile: dict[str, Any]
    ) -> None:
        """This is what makes the read-only deployment useful: every firm is
        drillable without a write-capable database, and every card is produced by
        the current model, policy and rubric rather than served from a stale row."""
        card = await service.get_card(str(profile["MSME_ID"]))
        assert card.msme_id == str(profile["MSME_ID"])
        assert 0 <= card.financial_health_score <= 100
        assert card.explanation is not None

    async def test_an_unknown_msme_raises(self, service: HealthCardService) -> None:
        with pytest.raises(MSMENotFoundError):
            await service.get_card("MSME_NOPE")

    async def test_scoring_is_deterministic(
        self, service: HealthCardService, profile: dict[str, Any]
    ) -> None:
        """No stored cards means every read re-scores, so the scoring path has to
        be reproducible or the same firm would drift between page loads."""
        msme_id = str(profile["MSME_ID"])
        first, second = await service.get_card(msme_id), await service.get_card(msme_id)
        assert first.financial_health_score == second.financial_health_score
        assert first.feature_snapshot_hash == second.feature_snapshot_hash


class TestSimulator:
    @pytest.fixture
    def simulator(self, config: dict[str, Any]) -> Simulator:
        return Simulator(rubric_only_bundle(), config)

    @pytest.mark.parametrize(
        ("build_overrides", "message"),
        [
            pytest.param(
                lambda config: {"Not_A_Column": 1}, "unknown feature", id="unknown-feature"
            ),
            pytest.param(
                lambda config: {c: 1 for c in config["schema"]["feature_columns"]},
                "too many overrides",
                id="too-many",
            ),
        ],
    )
    def test_an_invalid_override_set_is_rejected(
        self,
        simulator: Simulator,
        config: dict[str, Any],
        build_overrides: Callable[[dict[str, Any]], dict[str, Any]],
        message: str,
    ) -> None:
        with pytest.raises(CustomError, match=message):
            simulator.validate_overrides(build_overrides(config))

    def test_improving_compliance_raises_the_score(
        self, simulator: Simulator, profile: dict[str, Any]
    ) -> None:
        result = simulator.simulate(
            profile, {"GST_Filing_Timeliness_Pct": 100.0}, str(profile["MSME_ID"])
        )
        fhs = next(d for d in result.deltas if d.metric == "financial_health_score")
        assert fhs.after >= fhs.before

    def test_no_change_is_reported_as_unchanged(
        self, simulator: Simulator, profile: dict[str, Any]
    ) -> None:
        current = profile["GST_Filing_Timeliness_Pct"]
        result = simulator.simulate(
            profile, {"GST_Filing_Timeliness_Pct": current}, str(profile["MSME_ID"])
        )
        assert result.unchanged is True

    def test_before_and_after_pillars_are_both_returned(
        self, simulator: Simulator, profile: dict[str, Any]
    ) -> None:
        result = simulator.simulate(
            profile, {"Overdraft_Usage_Ratio": 0.01}, str(profile["MSME_ID"])
        )
        assert len(result.before_pillars) == len(result.after_pillars) == 6


def _scored_book(rows: int = 6) -> pd.DataFrame:
    """A batch-scoring frame in the shape ``score_portfolio`` emits."""
    return pd.DataFrame(
        [
            {
                "MSME_ID": f"MSME{index:07d}",
                "financial_health_score": 60.0 + index,
                "grade": "A" if index % 2 == 0 else "D",
                "risk_band": "Low" if index % 2 == 0 else "High",
                "eligible": index % 2 == 0,
                "credit_limit_inr": 500_000.0 if index % 2 == 0 else 0.0,
                "probability_of_default": 0.03 if index % 2 == 0 else 0.42,
                "Customer_Segment": "NTC" if index % 2 == 0 else "NTB",
                "Industry": "Retail",
                "Location_Category": "Tier 2",
                "thin_file": True,
                "flag_count": 0,
            }
            for index in range(rows)
        ]
    )


class TestPortfolioService:
    async def test_empty_store_returns_a_zero_summary(self, config: dict[str, Any]) -> None:
        """A dashboard with no data must render, not explode."""
        store = CsvCardStore(config, scores=pd.DataFrame(), profiles=pd.DataFrame())
        summary = await PortfolioService(store, config).summary()
        assert summary.firms == 0
        assert summary.approval_rate == 0.0

    async def test_summary_aggregates_the_scored_book(self, config: dict[str, Any]) -> None:
        store = CsvCardStore(config, scores=_scored_book(6), profiles=pd.DataFrame())
        summary = await PortfolioService(store, config).summary()
        assert summary.firms == 6
        assert summary.approval_rate == pytest.approx(0.5)
        assert summary.segment_lift
