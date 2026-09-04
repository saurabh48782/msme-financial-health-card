"""The rubric calibration: fitted, floored, bounded and idempotent."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.scoring.pillar_calibration import (
    WEIGHT_FLOOR,
    _floored,
    adjusted_r_squared,
    calibrate_all,
    calibrate_pillar,
    fails_floor,
    r_squared,
    to_yaml_block,
)
from src.utils.custom_error import CustomError


def _realised_floor(floor: float, drivers: int) -> float:
    """The lowest weight ``_floored`` can return, exactly.

    Clamping adds at most ``floor`` per driver to a vector that already summed to
    1, so the renormalising divisor is at most ``1 + drivers * floor`` — making
    ``floor / (1 + drivers * floor)`` the tight lower bound on any single weight.
    """
    return floor / (1.0 + drivers * floor) - 1e-9


class TestFlooredWeights:
    def test_weights_sum_to_exactly_one_after_rounding(self) -> None:
        """A normalised vector rounded to 4 places routinely lands on 1.0001, which
        validate_config rejects — so the residual is absorbed by the largest weight."""
        fitted = np.array([0.70, 0.01, 0.27, 0.01, 0.01])
        assert _floored(fitted, WEIGHT_FLOOR).sum() == pytest.approx(1.0, abs=1e-12)

    def test_floor_keeps_every_driver_influential(self) -> None:
        """A driver the synthetic generator ignored is not a driver real MSME credit
        ignores. The floor is what stops the fit deleting the cash buffer."""
        weights = _floored(np.array([1.0, 0.0]), 0.05)
        assert weights.min() >= _realised_floor(0.05, 2)

    def test_a_zero_floor_leaves_the_fit_untouched(self) -> None:
        fitted = np.array([0.7, 0.3])
        assert _floored(fitted, 0.0) == pytest.approx(fitted, abs=1e-9)

    def test_the_floor_is_what_rescues_a_zeroed_driver(self) -> None:
        """The pure fit drives four of the shipped drivers to zero. Without the
        floor the rubric would silently drop them, which is the whole reason it
        exists rather than being decoration."""
        weights = _floored(np.array([1.0, 0.0, 0.0]), WEIGHT_FLOOR)
        assert (weights > 0.0).all()
        assert weights[0] > weights[1]  # the fit still dominates


class TestCalibratePillar:
    def test_fit_meets_the_configured_threshold(
        self, featured_frame: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        result = calibrate_pillar(featured_frame, "compliance", config)
        # a 600-row sample is noisier than 50k, so the bar is the shape of the
        # result, not the production floor
        assert result["metrics"]["adj_r2"] >= 0.5
        assert result["metrics"]["adj_r2"] <= result["metrics"]["r2"]
        lo, hi = result["band"]
        assert 0.0 <= lo < hi <= 100.0

    def test_weights_sum_to_one(self, featured_frame: pd.DataFrame, config: dict[str, Any]) -> None:
        result = calibrate_pillar(featured_frame, "cashflow", config)
        assert sum(result["weights"].values()) == pytest.approx(1.0, abs=1e-9)

    def test_non_negative_weights_keep_every_driver_monotone(
        self, featured_frame: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        """Non-negativity is a fairness property, not a fitting trick: it makes it
        impossible for better GST filing to lower a Compliance score."""
        result = calibrate_pillar(featured_frame, "compliance", config)
        assert all(w >= 0.0 for w in result["weights"].values())
        assert all(w >= 0.0 for w in result["fitted_weights"].values())

    def test_calibration_is_idempotent(
        self, featured_frame: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        """The fit reads only the driver *ramps*, never the live ``weight``.

        If it read the current weight, each run would compound the last and the
        rubric would drift every time anyone ran the calibrator.
        """
        first = calibrate_pillar(featured_frame, "business_stability", config)
        second = calibrate_pillar(featured_frame, "business_stability", config)
        assert first["weights"] == second["weights"]
        assert first["band"] == second["band"]

    def test_the_fit_does_not_read_the_live_weight(
        self, featured_frame: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        """Perturbing the shipped weights must not move the fitted output — that is
        what makes calibrating twice the same as calibrating once."""
        import copy

        baseline = calibrate_pillar(featured_frame, "compliance", config)
        tweaked = copy.deepcopy(config)
        drivers = tweaked["pillars"]["compliance"]["drivers"]
        names = list(drivers)
        drivers[names[0]]["weight"] = 0.97
        for name in names[1:]:
            drivers[name]["weight"] = round(0.03 / (len(names) - 1), 4)
        assert (
            calibrate_pillar(featured_frame, "compliance", tweaked)["weights"]
            == (baseline["weights"])
        )

    @pytest.mark.parametrize(
        ("prepare", "message"),
        [
            pytest.param(
                lambda frame: frame.drop(columns=["Compliance_Score"]),
                "label column",
                id="missing-label",
            ),
            pytest.param(lambda frame: frame.head(20), "fully observed rows", id="too-few-rows"),
        ],
    )
    def test_an_unfittable_frame_raises(
        self,
        featured_frame: pd.DataFrame,
        config: dict[str, Any],
        prepare: Callable[[pd.DataFrame], pd.DataFrame],
        message: str,
    ) -> None:
        with pytest.raises(CustomError, match=message):
            calibrate_pillar(prepare(featured_frame), "compliance", config)


class TestCalibrateAll:
    def test_every_pillar_with_a_label_is_covered(
        self, featured_frame: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        report = calibrate_all(featured_frame, config)
        assert set(report["pillars"]) == set(config["schema"]["pillar_label_map"])

    def test_yaml_block_is_paste_ready(
        self, featured_frame: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        """The calibrator reports, it does not overwrite params.yaml: a rubric weight
        change is a reviewed commit, not a training-job side effect."""
        import yaml

        report = calibrate_all(featured_frame, config)
        block = to_yaml_block(report, config)
        parsed = yaml.safe_load("pillars:\n" + block)
        assert "compliance" in parsed["pillars"]
        # Complete driver lines, not just the weights: review is a diff of the
        # block, and what is pasted must be everything the runtime reads.
        compliance = parsed["pillars"]["compliance"]
        assert compliance["band"] == report["pillars"]["compliance"]["band"]
        for name, driver in compliance["drivers"].items():
            assert {"source", "weight", "worst", "best"} <= set(driver), name
            assert driver == config["pillars"]["compliance"]["drivers"][name] | {
                "weight": driver["weight"]
            }


class TestRSquared:
    @pytest.mark.parametrize(
        ("actual", "predicted", "expected"),
        [
            pytest.param(
                np.linspace(0.0, 100.0, 50), np.linspace(0.0, 100.0, 50), 1.0, id="perfect"
            ),
            pytest.param(
                np.array([1.0, 2.0, 3.0, 4.0]), np.full(4, 2.5), 0.0, id="predicting-the-mean"
            ),
            # A constant target has no variance to explain; the guard must return 0.0
            # rather than divide by it.
            pytest.param(np.full(10, 5.0), np.full(10, 5.0), 0.0, id="zero-variance-target"),
        ],
    )
    def test_r_squared_reports_the_explained_variance(
        self, actual: np.ndarray, predicted: np.ndarray, expected: float
    ) -> None:
        assert r_squared(actual, predicted) == pytest.approx(expected)


class TestAdjustedRSquared:
    """The whole point of the adjustment: predictors are not free."""

    def test_it_matches_the_closed_form(self) -> None:
        rng = np.random.default_rng(0)
        actual = rng.normal(size=200)
        predicted = actual + rng.normal(scale=0.5, size=200)
        plain = r_squared(actual, predicted)
        expected = 1.0 - (1.0 - plain) * (200 - 1) / (200 - 7 - 1)
        assert adjusted_r_squared(actual, predicted, 7) == pytest.approx(expected)

    def test_more_predictors_never_help(self) -> None:
        rng = np.random.default_rng(1)
        actual = rng.normal(size=500)
        predicted = actual + rng.normal(scale=0.4, size=500)
        scores = [adjusted_r_squared(actual, predicted, p) for p in (1, 5, 20, 100)]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] <= r_squared(actual, predicted)

    def test_a_free_fit_is_unpenalised(self) -> None:
        actual = np.linspace(0.0, 10.0, 50)
        assert adjusted_r_squared(actual, actual, 0) == pytest.approx(r_squared(actual, actual))

    def test_an_unsupportable_fit_is_nan_not_a_pass(self) -> None:
        """``n <= p + 1`` leaves nothing to adjust against; NaN must reach the gate."""
        actual = np.linspace(0.0, 1.0, 8)
        value = adjusted_r_squared(actual, actual, 8)
        assert math.isnan(value)
        assert fails_floor(value, 0.75)
        assert not fails_floor(0.8, 0.75)
        assert fails_floor(0.7, 0.75)

    def test_a_negative_predictor_count_is_rejected(self) -> None:
        with pytest.raises(CustomError):
            adjusted_r_squared(np.arange(10.0), np.arange(10.0), -1)


class TestFloorEnforcement:
    """The floor is approximate by design — but never absent, and never unbounded.

    ``_floored`` clamps to the floor and then renormalises, which divides every
    weight down again, so the realised floor sits *below* the nominal one. That is
    deliberate (iterating to honour the third decimal of a rubric weight would be
    precision theatre) but it still has to be a guarantee rather than a hope, so
    these tests assert the exact algebraic bound rather than a chosen tolerance.
    """

    @pytest.mark.parametrize("floor", [0.0, 0.01, 0.03, 0.05, 0.1])
    @pytest.mark.parametrize("drivers", [2, 4, 6])
    def test_floor_holds_to_its_algebraic_bound(self, floor: float, drivers: int) -> None:
        fitted = np.zeros(drivers)
        fitted[0] = 1.0
        weights = _floored(fitted, floor)
        assert weights.min() >= _realised_floor(floor, drivers)
        assert weights.sum() == pytest.approx(1.0, abs=1e-12)

    def test_extreme_fit_still_sums_to_one(self) -> None:
        weights = _floored(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 0.03)
        assert weights.sum() == pytest.approx(1.0, abs=1e-12)
        assert weights.min() >= _realised_floor(0.03, 6)

    def test_impossible_floor_is_rejected(self) -> None:
        """A floor of 0.3 across five drivers needs 1.5 of the available 1.0."""
        with pytest.raises(CustomError, match="impossible"):
            _floored(np.ones(5) / 5, 0.3)

    def test_all_pillars_respect_the_shipped_floor(self, config: dict[str, Any]) -> None:
        """The weights actually in params.yaml must clear the realised floor."""
        for key, spec in config["pillars"].items():
            if key == "fhs_weights":
                continue
            drivers = spec["drivers"]
            bound = _realised_floor(WEIGHT_FLOOR, len(drivers))
            for name, driver in drivers.items():
                assert float(driver["weight"]) >= bound, f"{key}.{name}"
