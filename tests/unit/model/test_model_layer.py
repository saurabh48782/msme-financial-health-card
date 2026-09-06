"""PD reliability reporting, fairness slicing and the metric helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.model.components import (
    evaluate_classification,
    evaluate_probability,
    ks_statistic,
    stratified_split,
)
from src.model.fairness import slice_report, to_frame, worst_deviation
from src.model.reliability import expected_calibration_error, reliability_curve


class TestReliability:
    """A PD is a price, not a ranking: a model 3 points high misprices every loan.

    So the report has to measure whether the number is trustworthy, not only
    whether the ordering is. These tests pin the measurement itself - there is
    deliberately no fitted corrective map to test; see
    :mod:`src.model.reliability` for why.
    """

    def test_a_systematic_bias_shows_up_as_calibration_error(self) -> None:
        rng = np.random.default_rng(0)
        actual = rng.uniform(0.0, 0.5, 3000)
        biased = np.clip(actual + 0.05, 0.0, 1.0)
        assert expected_calibration_error(actual, biased) > 0.04

    def test_every_bin_reports_its_gap_and_direction(self) -> None:
        rng = np.random.default_rng(2)
        actual = rng.uniform(0.0, 0.5, 2000)
        curve = reliability_curve(actual, np.clip(actual + 0.05, 0.0, 1.0))
        assert curve
        # A uniformly over-predicting model must show a positive gap everywhere,
        # which is what makes the curve actionable rather than just a number.
        assert all(row["gap"] > 0 for row in curve)

    def test_reliability_curve_uses_quantile_bins(self) -> None:
        """PD is heavily skewed toward zero; equal-width bins would put 90% of the
        portfolio in the first bucket and report calibration on a handful of firms."""
        predicted = np.concatenate([np.full(900, 0.01), np.linspace(0.1, 0.7, 100)])
        curve = reliability_curve(predicted, predicted, bins=10)
        counts = [row["count"] for row in curve]
        assert len(curve) >= 2
        assert max(counts) < len(predicted)

    def test_perfect_predictions_have_zero_calibration_error(self) -> None:
        values = np.linspace(0.01, 0.6, 400)
        assert expected_calibration_error(values, values) == pytest.approx(0.0, abs=1e-9)

    def test_empty_input_yields_an_empty_curve(self) -> None:
        assert reliability_curve(np.array([]), np.array([])) == []


class TestMetrics:
    @pytest.mark.parametrize(
        ("truth", "scores", "expected"),
        [
            pytest.param(np.zeros(10), np.random.default_rng(0).random(10), 0.0, id="single-class"),
            pytest.param(
                np.array([0] * 50 + [1] * 50),
                np.array([0.0] * 50 + [1.0] * 50),
                1.0,
                id="perfect-separation",
            ),
        ],
    )
    def test_ks_spans_no_separation_to_complete_separation(
        self, truth: np.ndarray, scores: np.ndarray, expected: float
    ) -> None:
        assert ks_statistic(truth, scores) == pytest.approx(expected)

    def test_probability_metrics_report_calibration_bias(self) -> None:
        actual = pd.Series(np.linspace(0.0, 0.5, 100))
        metrics = evaluate_probability(actual, actual.to_numpy() + 0.1, n_features=5)
        assert metrics["calibration_bias"] == pytest.approx(0.1, abs=1e-6)
        assert metrics["adj_r2"] < metrics["r2"]
        assert metrics["n_features"] == 5.0

    def test_classification_metrics_are_complete(self) -> None:
        truth = pd.Series([0, 1] * 50)
        scores = np.linspace(0.0, 1.0, 100)
        metrics = evaluate_classification(truth, scores)
        for key in ("roc_auc", "pr_auc", "ks", "precision", "recall", "f1"):
            assert key in metrics


class TestSplit:
    def test_split_preserves_the_segment_mix(
        self, featured_frame: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        """Stratifying is what makes the per-segment fairness slices meaningful."""
        train, test = stratified_split(featured_frame, config)
        column = config["model_params"]["stratify_column"]
        train_mix = train[column].value_counts(normalize=True).sort_index()
        test_mix = test[column].value_counts(normalize=True).sort_index()
        assert np.allclose(train_mix.to_numpy(), test_mix.to_numpy(), atol=0.05)

    def test_split_is_reproducible(
        self, featured_frame: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        first, _ = stratified_split(featured_frame, config)
        second, _ = stratified_split(featured_frame, config)
        assert (
            first[config["schema"]["id_column"]].tolist()
            == second[config["schema"]["id_column"]].tolist()
        )


class TestFairness:
    @pytest.fixture
    def scored(self, featured_frame: pd.DataFrame) -> pd.DataFrame:
        rng = np.random.default_rng(3)
        frame = featured_frame.copy()
        frame["predicted_pd"] = np.clip(
            frame["Probability_of_Default"] + rng.normal(0, 0.01, len(frame)), 0, 1
        )
        frame["predicted_eligibility"] = 1.0 - frame["predicted_pd"]
        return frame

    def test_report_covers_every_configured_slice(
        self, scored: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        report = slice_report(scored, config=config)
        for column in config["model_params"]["fairness_slice_columns"]:
            assert column in report["slices"]

    def test_deviation_is_measured_on_error_not_outcome(
        self, scored: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        """Cohorts may legitimately differ in approval rate; the model's *error*
        must not. Measuring deviation on outcomes would flag a real difference in
        creditworthiness as unfairness."""
        report = slice_report(scored, config=config)
        for deviations in report["max_deviation"].values():
            assert set(deviations) <= {"pd_mae", "pd_bias"}

    def test_unbiased_model_has_a_small_deviation(
        self, scored: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        assert worst_deviation(slice_report(scored, config=config), "pd_mae") < 0.05

    def test_flattened_frame_has_an_overall_row(
        self, scored: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        frame = to_frame(slice_report(scored, config=config))
        assert (frame["slice"] == "overall").sum() == 1


class TestInputExample:
    """MLflow infers the signature by scoring this frame, so it has to be a record
    the pipeline can actually take - an all-zeros row makes XGBoost reject 0.0 as a
    category and the model registers without a signature."""

    @pytest.fixture
    def example(self, featured_frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
        from src.model.model_orchestration import input_example

        return input_example(featured_frame, config)

    def test_example_is_a_single_real_record(
        self, example: pd.DataFrame, featured_frame: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        id_column = config["schema"]["id_column"]
        assert len(example) == 1
        assert list(example.columns) == [id_column, *config["schema"]["feature_columns"]]
        assert example[id_column].iloc[0] in set(featured_frame[id_column])

    def test_categoricals_carry_their_dataset_levels(
        self, example: pd.DataFrame, featured_frame: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        for column in config["schema"]["categorical_columns"]:
            assert example[column].iloc[0] in set(featured_frame[column].dropna())

    def test_structural_nulls_are_avoided_when_a_complete_row_exists(
        self, example: pd.DataFrame
    ) -> None:
        assert int(example.isna().sum().sum()) == 0

    def test_an_all_null_column_still_yields_a_row(
        self, featured_frame: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        from src.model.model_orchestration import input_example

        frame = featured_frame.copy()
        frame["EMI_On_Time_Rate_Pct"] = np.nan
        assert len(input_example(frame, config)) == 1
