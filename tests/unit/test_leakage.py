"""The leakage firewall.

This is the single most important test in the repository. The headline claim is
that alternative data alone predicts MSME credit risk. One leaked
``Financial_Health_Score`` would make that claim false while making every metric
look superb — the failure mode is invisible unless it is asserted.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from src.model.components import build_feature_matrix, forbidden_columns, model_feature_names
from src.utils.custom_error import CustomError


class TestFeatureAllowlist:
    def test_no_label_column_is_ever_a_feature(
        self, featured_frame: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        """Every supervision target must be absent from the model's inputs."""
        names = set(model_feature_names(featured_frame, config))
        leaked = names & set(config["schema"]["label_columns"])
        assert not leaked, f"label columns reached the model: {sorted(leaked)}"

    def test_our_own_pillar_scores_are_not_features(
        self, featured_frame: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        """Layer A output must not feed Layer B.

        Keeping them independent is what lets the card present the rubric and the
        ML risk view as two witnesses rather than one argument restated.
        """
        scored = featured_frame.assign(
            compliance_score=50.0, financial_health_score=70.0, grade="B"
        )
        names = set(model_feature_names(scored, config))
        assert "compliance_score" not in names
        assert "financial_health_score" not in names
        assert "grade" not in names

    def test_id_column_is_not_a_feature(
        self, featured_frame: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        """An identifier is a perfect in-sample predictor and a useless one."""
        assert config["schema"]["id_column"] not in model_feature_names(featured_frame, config)

    def test_build_feature_matrix_raises_on_forbidden_column(
        self, featured_frame: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        """An explicit request for a label column must fail loudly, not silently drop it."""
        names = model_feature_names(featured_frame, config)
        with pytest.raises(CustomError, match="LEAKAGE"):
            build_feature_matrix(
                featured_frame, config, features=[*names, "Probability_of_Default"]
            )

    def test_allowlist_is_built_by_intersection_not_subtraction(
        self, featured_frame: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        """A new upstream column must not become a feature by default.

        Subtracting known-bad columns fails open: anything unrecognised is treated
        as safe. Intersecting with an allowlist fails closed.
        """
        polluted = featured_frame.assign(Some_New_Leaky_Target=1.0)
        assert "Some_New_Leaky_Target" not in model_feature_names(polluted, config)

    def test_forbidden_set_covers_every_label(self, config: dict[str, Any]) -> None:
        forbidden = forbidden_columns(config)
        for column in config["schema"]["label_columns"]:
            assert column in forbidden


class TestFeatureMatrixContract:
    def test_missing_trained_column_raises(
        self, featured_frame: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        """Serving a frame that lacks a trained feature must fail, not impute."""
        names = model_feature_names(featured_frame, config)
        with pytest.raises(CustomError, match="missing trained column"):
            build_feature_matrix(featured_frame.drop(columns=[names[5]]), config, features=names)

    def test_column_order_is_preserved(
        self, featured_frame: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        """XGBoost is positional; a reordered frame silently scores nonsense."""
        names = model_feature_names(featured_frame, config)
        shuffled = featured_frame[list(reversed(featured_frame.columns))]
        matrix, returned, _ = build_feature_matrix(shuffled, config, features=names)
        assert list(matrix.columns) == names == returned
