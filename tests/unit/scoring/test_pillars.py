"""The pillar rubrics: monotone, bounded, and exactly decomposable.

Exact decomposition is the property that makes an adverse-action notice possible.
If contributions do not sum to the score, the explanation shown to a declined
borrower is a plausible story rather than the arithmetic that actually decided.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.scoring.pillars import (
    SCORE_MAX,
    SCORE_MIN,
    compute_pillar,
    compute_pillars,
    explain_pillar,
    explain_pillars,
    pillar_band,
    pillar_keys,
    ramp,
)


class TestRamp:
    @pytest.mark.parametrize(
        ("worst", "best", "value", "expected"),
        [
            (0.0, 10.0, 0.0, 0.0),
            (0.0, 10.0, 5.0, 0.5),
            (0.0, 10.0, 10.0, 1.0),
            (0.0, 10.0, -3.0, 0.0),
            (0.0, 10.0, 99.0, 1.0),
            # worst above best is how a lower-is-better driver is written.
            (10.0, 0.0, 0.0, 1.0),
            (10.0, 0.0, 5.0, 0.5),
            (10.0, 0.0, 10.0, 0.0),
            (10.0, 0.0, -3.0, 1.0),
            (10.0, 0.0, 99.0, 0.0),
        ],
    )
    def test_the_ramp_is_linear_between_the_anchors_and_clamped_outside(
        self, worst: float, best: float, value: float, expected: float
    ) -> None:
        assert ramp(value, worst, best)[()] == pytest.approx(expected)

    def test_nan_in_nan_out(self) -> None:
        """Availability is the caller's decision; the ramp must not invent a score."""
        assert np.isnan(ramp(np.nan, 0.0, 10.0)[()])

    def test_a_zero_width_ramp_raises(self) -> None:
        with pytest.raises(ValueError, match="zero-width"):
            ramp(1.0, 5.0, 5.0)


class TestConfiguredRubric:
    def test_driver_weights_sum_to_one(self, config: dict[str, Any]) -> None:
        for key in pillar_keys(config):
            weights = [float(d["weight"]) for d in config["pillars"][key]["drivers"].values()]
            assert sum(weights) == pytest.approx(1.0, abs=1e-6), key

    def test_fhs_weights_sum_to_one(self, config: dict[str, Any]) -> None:
        assert sum(config["pillars"]["fhs_weights"].values()) == pytest.approx(1.0, abs=1e-9)

    def test_every_band_is_fitted_and_inside_the_score_range(self, config: dict[str, Any]) -> None:
        """A band inside 0-100 is what makes the score unclampable."""
        for key in pillar_keys(config):
            lo, hi = pillar_band(key, config)
            assert SCORE_MIN <= lo < hi <= SCORE_MAX, key


class TestScores:
    def test_scores_are_bounded(self, featured_frame: pd.DataFrame, config: dict[str, Any]) -> None:
        scores = compute_pillars(featured_frame, config)
        for column in scores.columns:
            assert scores[column].between(SCORE_MIN, SCORE_MAX).all(), column

    def test_batch_and_single_row_agree(
        self, featured_frame: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        """The vectorised path and the explain path must not drift apart."""
        batch = compute_pillars(featured_frame.head(20), config)
        for position in range(20):
            row = featured_frame.iloc[position]
            for key, result in explain_pillars(row, config).items():
                assert result.score == pytest.approx(
                    float(batch.iloc[position][f"{key}_score"]), abs=1e-3
                )

    def test_decomposition_is_exact(
        self, featured_frame: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        for position in range(25):
            for result in explain_pillars(featured_frame.iloc[position], config).values():
                assert result.decomposition_error() < 1e-3, (
                    f"{result.key} contributions do not sum to its score"
                )

    def test_scores_never_leave_the_fitted_band(
        self, featured_frame: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        """The property that replaced clamping: no row can reach the 0/100 edge and
        force the contributions to be rescaled to keep adding up."""
        for key in pillar_keys(config):
            lo, hi = pillar_band(key, config)
            scores = compute_pillar(featured_frame, key, config).dropna()
            assert scores.between(lo - 1e-9, hi + 1e-9).all(), key


class TestMonotonicity:
    """Improving any driver may never lower its pillar. Non-negative fitted weights
    guarantee this structurally; the test is here because a future config edit
    could introduce a negative weight by hand."""

    @pytest.mark.parametrize(
        ("pillar", "column", "value", "improves"),
        [
            ("compliance", "GST_Filing_Timeliness_Pct", 100.0, True),
            ("payment_behaviour", "Vendor_Payment_Timeliness_Pct", 100.0, True),
            ("business_stability", "Years_in_Operation", 30.0, True),
            ("cashflow", "Cashflow_Stability_Index", 0.99, True),
            ("business_growth", "Revenue_Growth_Rate_Pct", 60.0, True),
            ("cashflow", "Overdraft_Usage_Ratio", 0.95, False),
            ("payment_behaviour", "Avg_Invoice_Payment_Delay_Days", 200.0, False),
            ("business_stability", "Employee_Attrition_Rate_Pct", 90.0, False),
        ],
    )
    def test_moving_a_driver_never_moves_the_pillar_the_other_way(
        self,
        featured_frame: pd.DataFrame,
        config: dict[str, Any],
        pillar: str,
        column: str,
        value: float,
        improves: bool,
    ) -> None:
        base = featured_frame.head(10)
        moved = base.assign(**{column: value})
        before = compute_pillar(base, pillar, config).to_numpy()
        after = compute_pillar(moved, pillar, config).to_numpy()
        if improves:
            assert (after >= before - 1e-9).all()
        else:
            assert (after <= before + 1e-9).all()


class TestUnavailableDrivers:
    def test_all_drivers_missing_yields_nan_not_zero(
        self, featured_frame: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        """A firm we cannot score must be NaN, never a confident zero."""
        sources = {str(d["source"]) for d in config["pillars"]["compliance"]["drivers"].values()}
        blanked = featured_frame.head(3).assign(**{s: np.nan for s in sources})
        assert compute_pillar(blanked, "compliance", config).isna().all()

    def test_missing_source_column_is_reported_not_fatal(
        self, featured_frame: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        dropped = featured_frame.head(3).drop(columns=["GST_Filing_Timeliness_Pct"])
        result = explain_pillar(dropped.iloc[0], "compliance", config)
        assert "gst_filing_timeliness" in result.unavailable_drivers
        assert result.score > 0.0
