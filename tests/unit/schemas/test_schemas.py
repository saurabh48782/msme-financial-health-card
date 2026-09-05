"""Contract shapes: required fields, ranges, aliases and round-trips."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import ValidationError

from src.schemas.healthcard import HealthCard
from src.schemas.msme import MSMEFeatures, ScoreRequest, SimulationRequest
from tests.stubs import make_card, synthetic_raw_frame


def _payload() -> dict:
    frame = synthetic_raw_frame(1)
    aliases = {f.alias for f in MSMEFeatures.model_fields.values() if f.alias}
    row = frame.iloc[0].to_dict()
    return {k: (None if v != v else v) for k, v in row.items() if k in aliases}  # noqa: PLR0124


class TestMSMEFeatures:
    def test_dataset_column_names_are_accepted(self) -> None:
        model = MSMEFeatures.model_validate(_payload())
        assert model.msme_id.startswith("MSME")

    def test_snake_case_names_are_also_accepted(self) -> None:
        """``populate_by_name`` lets an integrator post either spelling."""
        model = MSMEFeatures.model_validate(_payload())
        again = MSMEFeatures.model_validate(model.model_dump())
        assert again.msme_id == model.msme_id

    def test_round_trip_returns_dataset_column_names(self) -> None:
        payload = _payload()
        assert set(MSMEFeatures.model_validate(payload).to_row()) == set(payload)

    def test_unknown_field_is_rejected(self) -> None:
        """``extra="forbid"`` catches a typo at the edge instead of dropping it."""
        with pytest.raises(ValidationError):
            MSMEFeatures.model_validate({**_payload(), "Nonsense_Column": 1})

    def test_missing_required_field_is_rejected(self) -> None:
        payload = _payload()
        del payload["Annual_Turnover_INR"]
        with pytest.raises(ValidationError, match="annual_turnover_inr|Annual_Turnover_INR"):
            MSMEFeatures.model_validate(payload)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("GST_Filing_Timeliness_Pct", 150.0),
            ("Overdraft_Usage_Ratio", 1.5),
            ("Years_in_Operation", -1.0),
            ("Employee_Count", 0),
            ("Annual_Turnover_INR", -5.0),
            ("Customer_Segment", "Unknown"),
        ],
    )
    def test_out_of_range_values_are_rejected(self, field: str, value: object) -> None:
        with pytest.raises(ValidationError):
            MSMEFeatures.model_validate({**_payload(), field: value})

    def test_null_emi_rate_is_allowed(self) -> None:
        """It is structurally null for two thirds of the portfolio."""
        model = MSMEFeatures.model_validate({**_payload(), "EMI_On_Time_Rate_Pct": None})
        assert model.emi_on_time_rate_pct is None


class TestRequests:
    @pytest.mark.parametrize(
        "build",
        [
            pytest.param(lambda: ScoreRequest(records=[]), id="score-without-records"),
            pytest.param(
                lambda: SimulationRequest(overrides={}), id="simulation-without-overrides"
            ),
        ],
    )
    def test_an_empty_request_is_rejected(self, build: Callable[[], object]) -> None:
        with pytest.raises(ValidationError):
            build()


class TestHealthCard:
    def test_serialises_and_reparses(self) -> None:
        card = make_card()
        assert HealthCard.model_validate_json(card.model_dump_json()).msme_id == card.msme_id

    def test_summary_carries_the_decision(self) -> None:
        summary = make_card().summary
        assert set(summary) == {"msme_id", "fhs", "grade", "risk_band", "eligible", "limit_inr"}

    def test_score_is_bounded(self) -> None:
        with pytest.raises(ValidationError):
            make_card().model_copy(update={"financial_health_score": 150.0}).model_dump_json()
            HealthCard.model_validate({**make_card().model_dump(), "financial_health_score": 150.0})
