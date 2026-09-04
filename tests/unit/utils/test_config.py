"""Config loading, ``${ENV}`` resolution and validation."""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from src.utils.config import _resolve_placeholders, load_config, validate_config
from src.utils.custom_error import CustomError

_PLACEHOLDER_VARS = ("HC_TEST_VAR", "HC_A", "HC_B")

Mutation = Callable[[dict[str, Any]], None]


def _apply_env(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> None:
    """Pin every variable the cases use, so a leftover export cannot decide a test."""
    for name in _PLACEHOLDER_VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)


def _set(path: str, value: Any) -> Mutation:
    """A mutation that assigns ``value`` at a dotted ``path`` inside the config."""

    def mutate(config: dict[str, Any]) -> None:
        *parents, leaf = path.split(".")
        node: Any = config
        for key in parents:
            node = node[key]
        node[leaf] = value

    return mutate


def _drop_policy_section(config: dict[str, Any]) -> None:
    del config["policy"]


def _bump_driver_weight(config: dict[str, Any]) -> None:
    drivers = config["pillars"]["compliance"]["drivers"]
    drivers[next(iter(drivers))]["weight"] += 0.5


def _flatten_driver_ramp(config: dict[str, Any]) -> None:
    driver = config["pillars"]["cashflow"]["drivers"]["overdraft_usage"]
    driver["best"] = driver["worst"]


def _leak_a_label_into_the_features(config: dict[str, Any]) -> None:
    config["schema"]["feature_columns"].append("Financial_Health_Score")


def _reverse_grades(config: dict[str, Any]) -> None:
    config["grades"] = list(reversed(config["grades"]))


class TestPlaceholders:
    @pytest.mark.parametrize(
        ("env", "template", "expected"),
        [
            pytest.param({"HC_TEST_VAR": "hello"}, "${HC_TEST_VAR}", "hello", id="set"),
            pytest.param({}, "${HC_TEST_VAR:-fallback}", "fallback", id="default-when-unset"),
            pytest.param({}, "${HC_TEST_VAR:-}", "", id="empty-default"),
            pytest.param(
                {"HC_TEST_VAR": "real"}, "${HC_TEST_VAR:-fallback}", "real", id="set-beats-default"
            ),
            pytest.param({"HC_A": "a", "HC_B": "b"}, "${HC_A}/${HC_B}", "a/b", id="multiple"),
        ],
    )
    def test_placeholders_resolve(
        self,
        monkeypatch: pytest.MonkeyPatch,
        env: dict[str, str],
        template: str,
        expected: str,
    ) -> None:
        _apply_env(monkeypatch, env)
        assert _resolve_placeholders(template) == expected

    @pytest.mark.parametrize(
        "env",
        [
            pytest.param({}, id="unset"),
            # direnv exports optional vars as empty strings. Treating empty as "set"
            # would feed "" into a database DSN and fail much later, somewhere far
            # less obvious.
            pytest.param({"HC_TEST_VAR": ""}, id="empty"),
        ],
    )
    def test_a_variable_without_a_value_raises(
        self, monkeypatch: pytest.MonkeyPatch, env: dict[str, str]
    ) -> None:
        _apply_env(monkeypatch, env)
        with pytest.raises(CustomError, match="unset or empty"):
            _resolve_placeholders("${HC_TEST_VAR}")


class TestLoad:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(CustomError, match="not found"):
            load_config(tmp_path / "nope.yaml")

    def test_shipped_config_is_valid(self, config: dict[str, Any]) -> None:
        assert validate_config(copy.deepcopy(config)) is not None


class TestValidation:
    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            pytest.param(_drop_policy_section, "missing required section", id="missing-section"),
            pytest.param(
                _set("pillars.fhs_weights.compliance", 1.0),
                "fhs_weights must sum to 1.0",
                id="pillar-weights",
            ),
            pytest.param(
                _bump_driver_weight, "driver weights must sum to 1.0", id="driver-weights"
            ),
            pytest.param(_set("policy.pd_bands.low_max", 0.9), "pd_bands", id="non-monotone-bands"),
            pytest.param(
                _set("policy.limits.band_multiple.low", 1.5),
                "band_multiple",
                id="limit-multiple-above-one",
            ),
            # validate_config is the last line of the leakage firewall.
            pytest.param(_leak_a_label_into_the_features, "LEAKAGE", id="leaked-label"),
            pytest.param(_flatten_driver_ramp, "zero-width ramp", id="zero-width-ramp"),
            # A band outside 0-100 would put the score back in reach of a clamp,
            # and a clamped score no longer equals the sum of its drivers.
            pytest.param(
                _set("pillars.cashflow.band", [47.5784, 130.0]),
                "band must satisfy",
                id="band-outside-score-range",
            ),
            pytest.param(
                _set("pillars.cashflow.band", [91.0, 47.0]),
                "band must satisfy",
                id="inverted-band",
            ),
            pytest.param(_reverse_grades, "descending order", id="ascending-grades"),
        ],
    )
    def test_a_broken_config_is_rejected(
        self, config: dict[str, Any], mutate: Mutation, message: str
    ) -> None:
        broken = copy.deepcopy(config)
        mutate(broken)
        with pytest.raises(CustomError, match=message):
            validate_config(broken)

    def test_all_errors_are_reported_at_once(self, config: dict[str, Any]) -> None:
        """A misconfigured deployment should be fixed in one pass, not one boot per typo."""
        broken = copy.deepcopy(config)
        broken["pillars"]["fhs_weights"]["compliance"] += 0.3
        broken["policy"]["pd_bands"]["low_max"] = 0.9
        with pytest.raises(CustomError) as error:
            validate_config(broken)
        assert "fhs_weights" in str(error.value)
        assert "pd_bands" in str(error.value)
