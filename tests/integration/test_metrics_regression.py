"""The metric regression gate.

Marked ``eval`` and excluded by ``addopts``, because it scores the whole portfolio
and needs the DVC-tracked dataset. Run it deliberately::

    pytest -m eval

Its purpose is to turn the submission's claims into assertions. Every number this
project puts in a README — segment neutrality most of all — is checked here, so a
model or policy change that quietly breaks one fails a test instead of shipping.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.evaluation.report import build_report, check_thresholds
from src.utils.config import raw_csv_path

pytestmark = pytest.mark.eval


@pytest.fixture(scope="module")
def report(config: dict[str, Any]) -> dict[str, Any]:
    if not raw_csv_path(config).is_file():
        pytest.skip("the DVC-tracked dataset is not present")
    return build_report(config)


class TestThresholdGates:
    def test_no_configured_threshold_is_breached(
        self, report: dict[str, Any], config: dict[str, Any]
    ) -> None:
        assert check_thresholds(report, config) == []

    def test_pd_error_is_within_budget(
        self, report: dict[str, Any], config: dict[str, Any]
    ) -> None:
        limit = float(config["evaluation"]["thresholds"]["pd_mae_max"])
        assert report["probability_of_default"]["mae"] <= limit

    def test_pd_is_calibrated_not_merely_discriminating(self, report: dict[str, Any]) -> None:
        """A PD is a price. Ranking correctly while sitting 3 points high still
        misprices every loan in the book."""
        assert abs(report["probability_of_default"]["calibration_bias"]) < 0.01
        assert report["probability_of_default"]["ece"] < 0.01

    def test_eligibility_discrimination_holds(
        self, report: dict[str, Any], config: dict[str, Any]
    ) -> None:
        floor = float(config["evaluation"]["thresholds"]["eligibility_auc_min"])
        assert report["eligibility"]["roc_auc"] >= floor

    def test_every_pillar_tracks_its_ground_truth(
        self, report: dict[str, Any], config: dict[str, Any]
    ) -> None:
        floor = float(config["evaluation"]["thresholds"]["pillar_adj_r2_min"])
        assert len(report["pillars"]) == 6
        for key, values in report["pillars"].items():
            adjusted = values["adj_r2"]
            assert adjusted >= floor, f"{key} adjusted R² {adjusted} fell below {floor}"
            # The gate is the adjusted figure precisely so a pillar cannot buy its
            # way over the floor with extra drivers: adjusted never exceeds plain.
            assert adjusted <= values["r2"] + 1e-12

    def test_health_score_error_is_within_budget(
        self, report: dict[str, Any], config: dict[str, Any]
    ) -> None:
        limit = float(config["evaluation"]["thresholds"]["fhs_mae_max"])
        assert report["health_score"]["mae"] <= limit

    def test_credit_limits_track_the_reference_sizing(
        self, report: dict[str, Any], config: dict[str, Any]
    ) -> None:
        limit = float(config["evaluation"]["thresholds"]["limit_mape_max"])
        assert report["limits"]["mape"] <= limit


class TestSegmentNeutrality:
    """The inclusion claim, as a failing test rather than a paragraph."""

    def test_model_error_does_not_vary_by_cohort(
        self, report: dict[str, Any], config: dict[str, Any]
    ) -> None:
        """Cohorts may legitimately differ in creditworthiness; the model's *error*
        must not differ, or the system is transferring its own uncertainty onto one
        group of borrowers."""
        limit = float(config["evaluation"]["thresholds"]["fairness_max_deviation_points"])
        deviation = report["fairness"]["worst_pd_mae_deviation"] * 100.0
        assert deviation <= limit

    def test_thin_file_segments_are_not_systematically_declined(
        self, report: dict[str, Any]
    ) -> None:
        """New-to-Credit approval must stay close to Existing-to-Credit approval.

        This is the entire thesis: a firm without a bureau record is not a worse
        credit, it is an unmeasured one.
        """
        segments = report["fairness"]["slices"]["Customer_Segment"]
        rates = {level: values["approval_rate_predicted"] for level, values in segments.items()}
        assert {"NTC", "NTB", "Existing-to-Credit"} <= set(rates)
        spread = max(rates.values()) - min(rates.values())
        assert spread <= 0.10, f"approval rates diverge by {spread:.1%} across segments: {rates}"

    def test_our_segment_gap_is_no_wider_than_the_datasets(self, report: dict[str, Any]) -> None:
        """We must not amplify whatever cohort gap the reference decisions contain."""
        segments = report["fairness"]["slices"]["Customer_Segment"]
        ours = [v["approval_rate_predicted"] for v in segments.values()]
        theirs = [v["approval_rate_actual"] for v in segments.values()]
        assert (max(ours) - min(ours)) <= (max(theirs) - min(theirs)) + 0.02

    def test_pd_bias_is_near_zero_in_every_cohort(self, report: dict[str, Any]) -> None:
        for column, levels in report["fairness"]["slices"].items():
            for level, values in levels.items():
                assert abs(values["pd_bias"]) < 0.01, f"{column}={level} is biased"


class TestDecisionAgreement:
    def test_policy_reproduces_the_reference_decisions(self, report: dict[str, Any]) -> None:
        """The knock-out rules are our addition, so exact agreement is not the goal —
        but a large divergence would mean the engine is not reproducing the
        documented policy at all."""
        assert report["decisions"]["eligibility_agreement"] >= 0.90
        assert report["decisions"]["risk_band_agreement"] >= 0.80

    def test_approval_rate_is_plausible(self, report: dict[str, Any]) -> None:
        rate = report["decisions"]["approval_rate_ours"]
        assert 0.5 <= rate <= 0.95
