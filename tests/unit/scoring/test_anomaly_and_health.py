"""Anomaly rules, the health-score aggregate and the explainer's phrasing."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.scoring.anomaly import RULES, SEVERITY_ORDER, detect, flags_for_row
from src.scoring.explainer import build_explanation, format_value, pillar_reason_codes
from src.scoring.health_score import (
    compute_health_score,
    explain_health_score,
    grade_for,
    score_frame,
)
from src.scoring.pillars import explain_pillars


class TestAnomalyRules:
    @pytest.mark.parametrize(
        ("ratio", "fires"),
        [
            pytest.param(1.4, True, id="above-threshold"),
            pytest.param(0.6, False, id="below-threshold"),
            # Flagging on absent data would penalise exactly the thin-file firms
            # this system exists to include.
            pytest.param(np.nan, False, id="missing-metric"),
        ],
    )
    def test_a_rule_fires_only_on_an_observed_breach(self, ratio: float, fires: bool) -> None:
        frame = pd.DataFrame([{"purchases_to_sales_ratio": ratio, "credit_debit_ratio": 1.2}])
        assert bool(detect(frame)["flag_purchases_exceed_sales"].iloc[0]) is fires

    def test_flags_carry_the_evidence(self) -> None:
        flags = flags_for_row({"purchases_to_sales_ratio": 1.4})
        flag = next(f for f in flags if f.code == "purchases_exceed_sales")
        assert flag.value == 1.4
        assert flag.threshold == 1.0
        assert flag.severity in SEVERITY_ORDER
        assert flag.message

    def test_flags_are_ordered_most_severe_first(self) -> None:
        flags = flags_for_row(
            {
                "credit_debit_ratio": 0.5,  # critical
                "customer_concentration_ratio": 0.9,
                "Customer_Concentration_Ratio": 0.9,  # medium
                "balance_days_of_outflow": 1.0,  # medium
            }
        )
        ranks = [f.severity_rank for f in flags]
        assert ranks == sorted(ranks, reverse=True)

    def test_clean_firm_has_no_flags(self) -> None:
        assert (
            flags_for_row(
                {
                    "revenue_triangulation_gap": 0.02,
                    "gst_bank_reconciliation_gap": 0.01,
                    "purchases_to_sales_ratio": 0.6,
                    "payroll_to_turnover_ratio": 0.15,
                    "contraction_overdraft_signal": 0.0,
                    "Customer_Concentration_Ratio": 0.3,
                    "credit_debit_ratio": 1.25,
                    "balance_days_of_outflow": 20.0,
                }
            )
            == []
        )

    def test_max_severity_is_the_highest_flag(self) -> None:
        frame = pd.DataFrame([{"credit_debit_ratio": 0.5, "Customer_Concentration_Ratio": 0.9}])
        assert detect(frame)["max_severity"].iloc[0] == "critical"


class TestBatchDetection:
    def test_detect_reports_a_column_per_rule_plus_the_summary(
        self, featured_frame: pd.DataFrame
    ) -> None:
        result = detect(featured_frame)
        for rule in RULES:
            assert f"flag_{rule.code}" in result.columns
        assert "flag_count" in result.columns
        assert "max_severity" in result.columns
        assert result["flag_count"].min() >= 0

    def test_flag_count_matches_the_rule_columns(self, featured_frame: pd.DataFrame) -> None:
        """The count is what orders the review queue, so it must not drift from
        the columns it summarises."""
        result = detect(featured_frame)
        rule_columns = [f"flag_{rule.code}" for rule in RULES]
        assert (result["flag_count"] == result[rule_columns].sum(axis=1)).all()

    def test_batch_and_single_row_agree(self, featured_frame: pd.DataFrame) -> None:
        """A firm scored alone in real time must get the same flags as in the batch —
        no rule may depend on who else was in the frame."""
        row = featured_frame.iloc[0]
        batch = detect(featured_frame.head(1))
        single = flags_for_row(row)
        assert {f.code for f in single} == {
            rule.code for rule in RULES if bool(batch.iloc[0][f"flag_{rule.code}"])
        }


class TestHealthScore:
    def test_grade_bands_are_evaluated_top_down(self, config: dict[str, Any]) -> None:
        """Read the edges from the config: they move whenever the rubric is refit,
        and a hard-coded edge would fail for a reason that is not a bug."""
        bands = config["grades"]
        assert grade_for(100.0, config).grade == bands[0]["grade"]
        assert grade_for(0.0, config).grade == bands[-1]["grade"]
        for band, lower in zip(bands[:-1], bands[1:], strict=True):
            edge = float(band["min"])
            # The edge belongs to the higher grade.
            assert grade_for(edge, config).grade == band["grade"]
            assert grade_for(edge - 0.01, config).grade == lower["grade"]

    def test_fhs_is_the_weighted_pillar_aggregate(
        self, featured_frame: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        scored = score_frame(featured_frame.head(30), config)
        weights = config["pillars"]["fhs_weights"]
        expected = sum(scored[f"{key}_score"] * float(weight) for key, weight in weights.items())
        assert np.allclose(scored["financial_health_score"], expected, atol=1e-6)

    def test_explain_matches_the_batch_value(
        self, featured_frame: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        scored = score_frame(featured_frame.head(15), config)
        for position in range(15):
            health = explain_health_score(
                explain_pillars(featured_frame.iloc[position], config), config
            )
            assert health.value == pytest.approx(
                float(scored["financial_health_score"].iloc[position]), abs=1e-3
            )
            assert health.decomposition_error() < 1e-6

    def test_missing_pillar_weight_is_redistributed(self, config: dict[str, Any]) -> None:
        keys = list(config["pillars"]["fhs_weights"])
        frame = pd.DataFrame([{f"{k}_score": 80.0 for k in keys}])
        frame.loc[0, f"{keys[0]}_score"] = np.nan
        assert compute_health_score(frame, config).iloc[0] == pytest.approx(80.0, abs=1e-6)


class TestExplainer:
    def test_negative_reasons_are_framed_as_points_forgone(
        self, featured_frame: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        """ "You lost 9 points on GST compliance" is actionable; "your driver scored
        62" is not."""
        codes = pillar_reason_codes(explain_pillars(featured_frame.iloc[0], config), config)
        negatives = [c for c in codes if c.direction == "negative"]
        assert negatives
        assert all("cost" in c.text for c in negatives)
        assert all(c.impact > 0 for c in negatives)

    def test_positive_reasons_report_the_contribution(
        self, featured_frame: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        codes = pillar_reason_codes(explain_pillars(featured_frame.iloc[0], config), config)
        positives = [c for c in codes if c.direction == "positive" and "thin_file" not in c.code]
        assert positives
        assert all("contributed" in c.text for c in positives)

    def test_thin_file_is_stated_not_hidden(
        self, featured_frame: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        thin = featured_frame[~featured_frame["has_repayment_history"]]
        if thin.empty:
            pytest.skip("no thin-file firms in the sample")
        explanation = build_explanation(explain_pillars(thin.iloc[0], config), config=config)
        assert any("no formal credit history" in note.lower() for note in explanation.notes)

    def test_unavailable_drivers_produce_no_reason_code(
        self, featured_frame: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        thin = featured_frame[~featured_frame["has_repayment_history"]]
        if thin.empty:
            pytest.skip("no thin-file firms in the sample")
        codes = pillar_reason_codes(explain_pillars(thin.iloc[0], config), config)
        assert not any(c.code.endswith(".emi_on_time_rate") for c in codes)

    @pytest.mark.parametrize(
        ("source", "value", "expected"),
        [
            ("GST_Filing_Timeliness_Pct", 88.4, "88.4%"),
            ("Annual_Turnover_INR", 5_391_103.0, "Rs 53.91 lakh"),
            ("Annual_Turnover_INR", 254_885_957.0, "Rs 25.49 cr"),
            ("Avg_Invoice_Payment_Delay_Days", 12.4, "12 days"),
            ("Years_in_Operation", 4.3, "4.3 years"),
            ("Overdraft_Usage_Ratio", 0.241, "24%"),
            ("UPI_Daily_Txn_Count", 9.4, "9"),
        ],
    )
    def test_values_render_in_their_natural_unit(
        self, source: str, value: float, expected: str
    ) -> None:
        assert format_value(source, value) == expected

    def test_missing_value_renders_gracefully(self) -> None:
        assert format_value("Anything_INR", None) == "not available"
