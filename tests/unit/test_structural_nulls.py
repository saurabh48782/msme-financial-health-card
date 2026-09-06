"""Structural NULLs: "no track record" must stay a modelled state.

``EMI_On_Time_Rate_Pct`` is NULL for the 65% of firms that have never had a loan.
Imputing it would fabricate a repayment history for exactly the New-to-Credit
firms this system exists to serve — the bias would be invisible in aggregate
metrics and devastating in individual decisions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.data.feature_engineer import add_derived_features
from src.scoring.pillars import explain_pillar
from tests.stubs import thin_file_frame


class TestRepaymentHistoryFlag:
    def test_flag_is_false_exactly_when_the_rate_is_null(
        self, featured_frame: pd.DataFrame
    ) -> None:
        expected = (
            featured_frame["Has_Existing_Loan"].astype("string").eq("Yes")
            & featured_frame["EMI_On_Time_Rate_Pct"].notna()
        )
        assert (featured_frame["has_repayment_history"] == expected).all()

    def test_the_rate_itself_is_never_imputed(self, featured_frame: pd.DataFrame) -> None:
        """The NaN must survive the feature pipeline intact."""
        no_loan = featured_frame["Has_Existing_Loan"].astype("string").eq("No")
        assert featured_frame.loc[no_loan, "EMI_On_Time_Rate_Pct"].isna().all()

    def test_derived_features_preserve_the_null(self, config: dict[str, Any]) -> None:
        derived = add_derived_features(thin_file_frame(), config)
        assert pd.isna(derived.loc[0, "EMI_On_Time_Rate_Pct"])
        assert (
            derived.loc[0, "has_repayment_history"] is np.False_
            or not derived.loc[0, "has_repayment_history"]
        )


class TestWeightRedistribution:
    def test_missing_driver_weight_is_redistributed_not_zeroed(
        self, featured_frame: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        """A thin-file firm's remaining drivers must carry the full pillar weight."""
        thin = featured_frame[~featured_frame["has_repayment_history"]]
        if thin.empty:
            return
        result = explain_pillar(thin.iloc[0], "payment_behaviour", config)
        available = [c for c in result.contributions if c.available]
        assert result.thin_file is True
        assert sum(c.weight for c in available) == pytest.approx(1.0, abs=1e-6)
        assert all(c.weight == 0.0 for c in result.contributions if not c.available)

    def test_thin_file_firm_is_not_scored_zero_on_the_pillar(
        self, featured_frame: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        """The absence of a loan must not read as bad repayment behaviour."""
        thin = featured_frame[~featured_frame["has_repayment_history"]]
        if thin.empty:
            return
        result = explain_pillar(thin.iloc[0], "payment_behaviour", config)
        assert result.score > 0.0

    def test_full_file_firm_uses_the_nominal_weights(
        self, featured_frame: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        full = featured_frame[featured_frame["has_repayment_history"]]
        if full.empty:
            return
        result = explain_pillar(full.iloc[0], "payment_behaviour", config)
        nominal = config["pillars"]["payment_behaviour"]["drivers"]["emi_on_time_rate"]["weight"]
        emi = next(c for c in result.contributions if c.name == "emi_on_time_rate")
        assert emi.available is True
        assert emi.weight == pytest.approx(float(nominal), abs=1e-6)
        assert result.thin_file is False
