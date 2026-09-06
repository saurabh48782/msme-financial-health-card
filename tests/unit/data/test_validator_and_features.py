"""Dataset validation and derived-feature arithmetic."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.data.data_validator import invariant_bounds, validate
from src.data.feature_engineer import DERIVED_COLUMNS, add_derived_features
from src.utils.custom_error import CustomError
from tests.stubs import thin_file_frame


def _duplicate_the_ids(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([frame.head(3), frame.head(3)], ignore_index=True)


def _push_a_value_out_of_range(frame: pd.DataFrame) -> pd.DataFrame:
    broken = frame.head(5).copy()
    broken.loc[broken.index[0], "GST_Filing_Timeliness_Pct"] = 250.0
    return broken


def _introduce_an_unknown_category(frame: pd.DataFrame) -> pd.DataFrame:
    broken = frame.head(5).copy()
    broken["Business_Type"] = broken["Business_Type"].astype("object")
    broken.loc[broken.index[0], "Business_Type"] = "Cooperative Society"
    return broken


class TestValidator:
    def test_clean_frame_passes(self, raw_frame: pd.DataFrame, config: dict[str, Any]) -> None:
        assert validate(raw_frame, config).ok

    def test_empty_frame_fails(self, config: dict[str, Any]) -> None:
        assert not validate(pd.DataFrame(), config).ok

    @pytest.mark.parametrize(
        ("break_frame", "error"),
        [
            pytest.param(_duplicate_the_ids, "duplicate", id="duplicate-ids"),
            pytest.param(
                _push_a_value_out_of_range, "GST_Filing_Timeliness_Pct", id="out-of-range"
            ),
            # An unmapped level would score 0 on its driver, quietly penalising a firm.
            pytest.param(
                _introduce_an_unknown_category, "Business_Type", id="unknown-categorical-level"
            ),
        ],
    )
    def test_a_broken_frame_is_rejected_with_a_named_error(
        self,
        raw_frame: pd.DataFrame,
        config: dict[str, Any],
        break_frame: Callable[[pd.DataFrame], pd.DataFrame],
        error: str,
    ) -> None:
        report = validate(break_frame(raw_frame), config)
        assert not report.ok
        assert any(error in message for message in report.errors)

    def test_broken_structural_null_rule_fails(
        self, raw_frame: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        """If the EMI nullness rule stops holding, the dataset has changed shape and
        ``has_repayment_history`` silently stops meaning what it means."""
        broken = raw_frame.head(20).copy()
        no_loan = broken["Has_Existing_Loan"] == "No"
        if not no_loan.any():
            pytest.skip("sample has no thin-file firms")
        broken.loc[no_loan.idxmax(), "EMI_On_Time_Rate_Pct"] = 95.0
        report = validate(broken, config)
        assert not report.ok
        assert any("structural" in e or "nullness" in e for e in report.errors)

    def test_raise_for_errors_raises(self, config: dict[str, Any]) -> None:
        with pytest.raises(CustomError):
            validate(pd.DataFrame(), config).raise_for_errors()

    def test_label_presence_is_a_warning_not_an_error(
        self, raw_frame: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        report = validate(raw_frame, config)
        assert report.ok
        assert any("label column" in w for w in report.warnings)


class TestInvariantBounds:
    """The bounds are read off ``MSMEFeatures`` rather than restated, so the batch
    CSV path and the API cannot disagree about what a column's domain is."""

    def test_every_bounded_contract_field_is_covered(self) -> None:
        from src.schemas.msme import MSMEFeatures

        bounds = invariant_bounds()
        # Optional fields nest their constraints one FieldInfo deeper; this is the
        # case that silently went missing when the unwrapping was one level short.
        assert bounds["EMI_On_Time_Rate_Pct"] == (0.0, 100.0)
        aliases = {f.alias for f in MSMEFeatures.model_fields.values() if f.alias}
        assert set(bounds) <= aliases

    @pytest.mark.parametrize(
        ("column", "expected"),
        [
            ("GST_Filing_Timeliness_Pct", (0.0, 100.0)),
            ("Overdraft_Usage_Ratio", (0.0, 1.0)),
            ("Employee_Count", (1.0, np.inf)),
            ("Revenue_Growth_Rate_Pct", (-100.0, np.inf)),
        ],
    )
    def test_definitional_bounds_are_what_the_contract_says(
        self, column: str, expected: tuple[float, float]
    ) -> None:
        assert invariant_bounds()[column] == expected

    def test_open_ended_magnitudes_carry_no_ceiling(self) -> None:
        """No hand-picked number for "too much turnover" — that is winsorisation's job."""
        for column in (
            "Annual_Turnover_INR",
            "Employee_Count",
            "Credit_History_Months",
            "Avg_Invoice_Payment_Delay_Days",
            "UPI_Daily_Txn_Count",
        ):
            assert invariant_bounds()[column][1] == np.inf, column

    def test_a_payload_the_api_rejects_is_rejected_in_batch_too(
        self, raw_frame: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        """The point of one definition: both tiers agree on the same bad value."""
        from pydantic import ValidationError

        from src.schemas.msme import MSMEFeatures

        row = raw_frame.head(1).copy()
        row["GST_Filing_Timeliness_Pct"] = 250.0
        assert not validate(row, config).ok

        payload = {**row.iloc[0].to_dict(), "GST_Filing_Timeliness_Pct": 250.0}
        with pytest.raises(ValidationError):
            MSMEFeatures.model_validate(payload)


class TestDerivedFeatures:
    def test_every_documented_column_is_produced(self, featured_frame: pd.DataFrame) -> None:
        for column in DERIVED_COLUMNS:
            assert column in featured_frame.columns, column

    def test_no_infinities_survive(self, featured_frame: pd.DataFrame) -> None:
        numeric = featured_frame.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.to_numpy()).any()

    def test_zero_denominator_yields_a_finite_ratio(self, config: dict[str, Any]) -> None:
        """A firm with no bank debits must produce a large-but-finite ratio, not an
        inf that becomes a silent NaN three layers later."""
        frame = thin_file_frame(Monthly_Bank_Debits_INR=0.0, Monthly_GST_Purchases_INR=0.0)
        derived = add_derived_features(frame, config)
        assert np.isfinite(derived.loc[0, "credit_debit_ratio"])
        assert np.isfinite(derived.loc[0, "balance_days_of_outflow"])

    def test_dscr_is_capped_for_debt_free_firms(self, featured_frame: pd.DataFrame) -> None:
        """No debt is infinite coverage; the cap keeps it a usable feature."""
        debt_free = featured_frame[featured_frame["Monthly_Loan_EMI_INR"] <= 1.0]
        if debt_free.empty:
            pytest.skip("no debt-free firms in the sample")
        assert (debt_free["dscr_proxy"] <= 25.0).all()

    def test_triangulation_gap_is_non_negative(self, featured_frame: pd.DataFrame) -> None:
        assert (featured_frame["revenue_triangulation_gap"] >= 0).all()

    def test_row_count_is_preserved(self, raw_frame: pd.DataFrame, config: dict[str, Any]) -> None:
        assert len(add_derived_features(raw_frame, config)) == len(raw_frame)
