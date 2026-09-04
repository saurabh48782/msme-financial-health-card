"""The policy engine: band edges, knock-outs, limit sizing and DSCR headroom.

A credit decision has to be defensible line by line, so every branch here is
pinned - especially the exact band edges, where an off-by-epsilon moves a firm
between Low and Medium risk and changes its limit by 8% of turnover.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import pandas as pd
import pytest

from src.scoring.anomaly import RiskFlag, flags_for_row
from src.scoring.health_score import score_frame
from src.scoring.policy import (
    BAND_HIGH,
    BAND_LOW,
    BAND_MEDIUM,
    annuity_principal,
    decide,
    decide_frame,
    risk_band_for,
    risk_band_series,
    round_limit,
)


@pytest.fixture
def healthy_firm() -> dict[str, Any]:
    return {
        "Annual_Turnover_INR": 6_000_000.0,
        "Years_in_Operation": 6.0,
        "GST_Filing_Timeliness_Pct": 92.0,
        "Overdraft_Usage_Ratio": 0.15,
        "Monthly_Bank_Credits_INR": 600_000.0,
        "Monthly_Bank_Debits_INR": 450_000.0,
        "Monthly_Loan_EMI_INR": 0.0,
    }


class TestRiskBands:
    @pytest.mark.parametrize(
        ("pd_value", "band"),
        [
            (0.0, BAND_LOW),
            (0.059999, BAND_LOW),
            (0.06, BAND_LOW),  # the edge belongs to the lower band
            (0.060001, BAND_MEDIUM),
            (0.15, BAND_MEDIUM),
            (0.150001, BAND_HIGH),
            (0.75, BAND_HIGH),
        ],
    )
    def test_band_edges_are_exact(self, pd_value: float, band: str, config: dict[str, Any]) -> None:
        assert risk_band_for(pd_value, config) == band

    def test_series_and_scalar_agree(self, config: dict[str, Any]) -> None:
        values = pd.Series([0.0, 0.06, 0.0601, 0.15, 0.9])
        assert list(risk_band_series(values, config)) == [risk_band_for(v, config) for v in values]


class TestKnockOuts:
    @pytest.mark.parametrize(
        ("field", "value", "code"),
        [
            ("Years_in_Operation", 0.05, "min_vintage"),
            ("Annual_Turnover_INR", 1000.0, "min_turnover"),
        ],
    )
    def test_a_firm_below_a_floor_is_declined_with_that_code(
        self,
        healthy_firm: dict[str, Any],
        config: dict[str, Any],
        field: str,
        value: float,
        code: str,
    ) -> None:
        decision = decide({**healthy_firm, field: value}, 0.03, 80.0, [], config)
        assert not decision.eligible
        assert any(k.code == code for k in decision.knockouts)

    @pytest.mark.parametrize(
        ("severity", "eligible"),
        [
            # Only `critical` severity blocks; that mapping lives in params.yaml.
            pytest.param("critical", False, id="critical-declines"),
            # An unusual firm is not a declined firm - that distinction is the point.
            pytest.param("medium", True, id="medium-does-not"),
        ],
    )
    def test_only_a_critical_flag_declines_an_otherwise_good_firm(
        self, healthy_firm: dict[str, Any], config: dict[str, Any], severity: str, eligible: bool
    ) -> None:
        flag = RiskFlag(
            code="negative_free_cashflow",
            severity=severity,
            message="Bank debits persistently exceed credits",
            metric="credit_debit_ratio",
            value=0.8,
            threshold=0.95,
        )
        decision = decide(healthy_firm, 0.02, 85.0, [flag], config)
        assert decision.eligible is eligible
        if not eligible:
            assert any(k.code.startswith("anomaly_") for k in decision.knockouts)

    def test_every_decline_carries_a_reason(
        self, healthy_firm: dict[str, Any], config: dict[str, Any]
    ) -> None:
        decision = decide(healthy_firm, 0.9, 85.0, [], config)
        assert not decision.eligible
        assert decision.notes or decision.decline_reasons


class TestLimits:
    def test_ineligible_firms_get_zero(
        self, healthy_firm: dict[str, Any], config: dict[str, Any]
    ) -> None:
        decision = decide(healthy_firm, 0.5, 85.0, [], config)
        assert decision.credit_limit_inr == 0.0
        assert decision.tenor_months == 0
        assert decision.indicative_rate_pct == 0.0

    def test_limit_scales_with_the_cashflow_pillar(
        self, healthy_firm: dict[str, Any], config: dict[str, Any]
    ) -> None:
        """Sizing on cashflow rather than turnover alone is the design decision here."""
        weak = decide(healthy_firm, 0.03, 45.0, [], config)
        strong = decide(healthy_firm, 0.03, 95.0, [], config)
        assert strong.credit_limit_inr > weak.credit_limit_inr

    def test_low_band_gets_a_larger_multiple_than_medium(
        self, healthy_firm: dict[str, Any], config: dict[str, Any]
    ) -> None:
        low = decide(healthy_firm, 0.02, 80.0, [], config)
        medium = decide(healthy_firm, 0.10, 80.0, [], config)
        assert low.credit_limit_inr > medium.credit_limit_inr
        assert low.indicative_rate_pct < medium.indicative_rate_pct
        assert low.tenor_months >= medium.tenor_months

    def test_limit_is_rounded_down(self, config: dict[str, Any]) -> None:
        """Never sanction more than the sizing supports."""
        rounding = float(config["policy"]["limits"]["rounding_inr"])
        assert round_limit(123_456.0, rounding) == 123_000.0
        assert round_limit(999.0, 1000.0) == 0.0

    def test_dscr_cap_binds_for_an_over_levered_firm(self, config: dict[str, Any]) -> None:
        """A firm already consuming its cashflow on EMIs cannot service more."""
        levered = {
            "Annual_Turnover_INR": 20_000_000.0,
            "Years_in_Operation": 8.0,
            "GST_Filing_Timeliness_Pct": 95.0,
            "Overdraft_Usage_Ratio": 0.1,
            "Monthly_Bank_Credits_INR": 1_800_000.0,
            "Monthly_Bank_Debits_INR": 1_700_000.0,
            "Monthly_Loan_EMI_INR": 400_000.0,
        }
        decision = decide(levered, 0.03, 85.0, [], config)
        sized = decision.limit_basis["sized_limit_inr"]
        capped = decision.limit_basis["dscr_capped_principal"]
        assert capped < sized
        assert any("debt-service" in note for note in decision.notes)

    def test_annuity_principal_is_positive_and_grows_with_tenor(self) -> None:
        short = annuity_principal(10_000.0, 12, 14.0)
        long = annuity_principal(10_000.0, 36, 14.0)
        assert 0 < short < long


class _Batch(NamedTuple):
    sample: pd.DataFrame
    scores: pd.DataFrame
    pd_values: pd.Series
    decisions: pd.DataFrame


def _decide_batch(
    featured_frame: pd.DataFrame,
    config: dict[str, Any],
    *,
    rows: int,
    pd_of: Callable[[int], float],
) -> _Batch:
    """A batch decision over the first ``rows`` firms, together with its inputs."""
    sample = featured_frame.head(rows).reset_index(drop=True)
    scores = score_frame(sample, config)
    pd_values = pd.Series([pd_of(i) for i in range(len(sample))], index=sample.index)
    decisions = decide_frame(sample, pd_values, scores["cashflow_score"], config)
    return _Batch(sample, scores, pd_values, decisions)


class TestVectorisedAgreesWithScalar:
    """The batch path is a performance shortcut over ``decide``. If the two ever
    disagree, a portfolio number and a card number describe the same firm
    differently — which is exactly the kind of divergence an auditor finds."""

    def test_paths_agree_on_a_real_sample(
        self, featured_frame: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        batch = _decide_batch(
            featured_frame, config, rows=120, pd_of=lambda i: 0.02 + (i % 40) * 0.006
        )

        for position in range(len(batch.sample)):
            row = batch.sample.iloc[position]
            scalar = decide(
                row,
                float(batch.pd_values.iloc[position]),
                float(batch.scores["cashflow_score"].iloc[position]),
                flags_for_row(row),
                config,
            )
            decisions = batch.decisions
            assert scalar.eligible == bool(decisions["eligible"].iloc[position]), position
            assert scalar.credit_limit_inr == pytest.approx(
                float(decisions["credit_limit_inr"].iloc[position]), abs=1.0
            ), position
            assert scalar.risk_band == decisions["risk_band"].iloc[position]

    def test_every_batch_decline_has_a_reason_code(
        self, featured_frame: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        decisions = _decide_batch(
            featured_frame, config, rows=200, pd_of=lambda i: 0.01 + (i % 60) * 0.012
        ).decisions
        declined = decisions[~decisions["eligible"]]
        assert (declined["knockout_codes"].astype(str).str.len() > 0).all()

    def test_ineligible_rows_always_have_zero_limit(
        self, featured_frame: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        decisions = _decide_batch(featured_frame, config, rows=200, pd_of=lambda _: 0.4).decisions
        assert (decisions.loc[~decisions["eligible"], "credit_limit_inr"] == 0.0).all()
