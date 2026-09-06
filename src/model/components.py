"""Feature matrix assembly, XGBoost trainers and evaluators.

**The leakage firewall.** ``build_feature_matrix`` raises if any column from
``schema.label_columns`` - or any pillar score we computed
ourselves - reaches the model.

**Native NaN handling.** ``EMI_On_Time_Rate_Pct`` is NULL for 65% of firms because
they have never had a loan. XGBoost sends missing values down a learned default
branch, so "no track record" stays a modelled state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from src.scoring.pillar_calibration import adjusted_r_squared
from src.scoring.pillars import pillar_keys
from src.utils.config import load_config
from src.utils.custom_error import CustomError
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrainedModel:
    """A fitted estimator with the exact feature contract it was trained on."""

    name: str
    estimator: Any
    features: list[str]
    categorical_features: list[str]
    metrics: dict[str, float] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)


def forbidden_columns(config: dict[str, Any] | None = None) -> set[str]:
    """Everything a model must never see: labels, the id, and our own pillar outputs."""
    cfg = config if config is not None else load_config()
    schema = cfg["schema"]
    ours = {f"{key}_score" for key in pillar_keys(cfg)}
    return {
        schema["id_column"],
        *schema["label_columns"],
        *ours,
        "financial_health_score",
        "grade",
        "grade_label",
    }


def model_feature_names(frame: pd.DataFrame, config: dict[str, Any] | None = None) -> list[str]:
    """The allowlist: the raw schema inputs present in ``frame``, and nothing else.

    Built by intersection with an explicit allowlist rather than by subtracting
    known-bad columns, so a new label column added upstream cannot silently become
    a feature.
    """
    cfg = config if config is not None else load_config()
    allowed = [*cfg["schema"]["feature_columns"]]
    forbidden = forbidden_columns(cfg)
    seen: set[str] = set()
    names: list[str] = []
    for column in allowed:
        if column in frame.columns and column not in forbidden and column not in seen:
            seen.add(column)
            names.append(column)
    return names


def build_feature_matrix(
    frame: pd.DataFrame,
    config: dict[str, Any] | None = None,
    *,
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Return ``(X, feature_names, categorical_feature_names)``.

    Pass ``features`` at inference time to reproduce the training contract exactly,
    including column order - XGBoost is positional and a reordered frame silently
    scores nonsense.
    """
    cfg = config if config is not None else load_config()
    names = features if features is not None else model_feature_names(frame, cfg)

    leaked = sorted(set(names) & forbidden_columns(cfg))
    if leaked:
        raise CustomError(f"LEAKAGE: forbidden column(s) in the feature matrix: {leaked}")

    missing = [c for c in names if c not in frame.columns]
    if missing:
        raise CustomError(f"feature matrix is missing trained column(s): {missing}")

    matrix = frame.loc[:, names].copy()
    categorical = [c for c in cfg["schema"]["categorical_columns"] if c in matrix.columns]
    for column in categorical:
        matrix[column] = matrix[column].astype("string").astype("category")
    for column in matrix.columns:
        if column in categorical:
            continue
        if matrix[column].dtype == bool:
            matrix[column] = matrix[column].astype("float64")
        else:
            matrix[column] = pd.to_numeric(matrix[column], errors="coerce").astype("float64")
    matrix = matrix.replace([np.inf, -np.inf], np.nan)
    return matrix, names, categorical


def stratified_split(
    frame: pd.DataFrame, config: dict[str, Any] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train/test split stratified on ``Customer_Segment``.

    Stratifying on segment is what makes the per-segment fairness slices
    meaningful - an unstratified split can leave the test set light on NTC firms,
    which is precisely the cohort whose metrics we are claiming.
    """
    cfg = config if config is not None else load_config()
    params = cfg["model_params"]
    column = params.get("stratify_column")
    stratify = frame[column] if column and column in frame.columns else None
    train, test = train_test_split(
        frame,
        test_size=float(params["test_size"]),
        random_state=int(params["random_state"]),
        stratify=stratify,
    )
    logger.info("Split", train_rows=len(train), test_rows=len(test), stratified_on=column or "none")
    return train.reset_index(drop=True), test.reset_index(drop=True)


def evaluate_regression(
    actual: pd.Series, predicted: NDArray[np.float64], *, n_features: int
) -> dict[str, float]:
    """Fit metrics, headlined by adjusted R².

    ``n_features`` is mandatory rather than optional because plain R² rises with
    every column handed to the model, informative or not - so a fit metric
    quoted without its predictor count is not a claim anyone can check. The
    unadjusted figure stays in the dict for readers who expect it; the gates
    read ``adj_r2``.
    """
    return {
        "r2": round(float(r2_score(actual, predicted)), 6),
        "adj_r2": round(
            adjusted_r_squared(np.asarray(actual, dtype="float64"), predicted, n_features), 6
        ),
        "n_features": float(n_features),
        "mae": round(float(mean_absolute_error(actual, predicted)), 6),
        "rmse": round(float(np.sqrt(np.mean((np.asarray(actual) - predicted) ** 2))), 6),
    }


def evaluate_probability(
    actual: pd.Series, predicted: NDArray[np.float64], *, n_features: int
) -> dict[str, float]:
    """Metrics for a PD prediction: fit *and* calibration.

    ``calibration_bias`` is reported alongside adjusted R² because a PD model that
    ranks perfectly but is systematically 3 points high still misprices every loan.
    """
    clipped = np.clip(predicted, 0.0, 1.0)
    metrics = evaluate_regression(actual, clipped, n_features=n_features)
    metrics["mean_predicted"] = round(float(clipped.mean()), 6)
    metrics["mean_actual"] = round(float(np.asarray(actual, dtype="float64").mean()), 6)
    metrics["calibration_bias"] = round(metrics["mean_predicted"] - metrics["mean_actual"], 6)
    return metrics


def ks_statistic(actual: NDArray[Any], scores: NDArray[np.float64]) -> float:
    """Kolmogorov-Smirnov separation."""
    actual = np.asarray(actual).astype(bool)
    if actual.all() or not actual.any():
        return 0.0
    order = np.argsort(scores)
    positives = np.cumsum(actual[order]) / actual.sum()
    negatives = np.cumsum(~actual[order]) / (~actual).sum()
    return round(float(np.max(np.abs(positives - negatives))), 6)


def evaluate_classification(
    actual: pd.Series, probabilities: NDArray[np.float64], *, threshold: float = 0.5
) -> dict[str, float]:
    truth = np.asarray(actual).astype(bool)
    predicted = probabilities >= threshold
    true_positive = int((predicted & truth).sum())
    false_positive = int((predicted & ~truth).sum())
    false_negative = int((~predicted & truth).sum())
    precision = true_positive / (true_positive + false_positive) if predicted.any() else 0.0
    recall = true_positive / (true_positive + false_negative) if truth.any() else 0.0
    return {
        "roc_auc": round(float(roc_auc_score(truth, probabilities)), 6),
        "pr_auc": round(float(average_precision_score(truth, probabilities)), 6),
        "ks": ks_statistic(truth, probabilities),
        "accuracy": round(float((predicted == truth).mean()), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(
            float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0, 6
        ),
        "threshold": threshold,
        "positive_rate": round(float(truth.mean()), 6),
    }


def train_pd_regressor(train: pd.DataFrame, config: dict[str, Any] | None = None) -> TrainedModel:
    """XGBoost regressor on ``Probability_of_Default``."""
    cfg = config if config is not None else load_config()
    params = dict(cfg["model_params"]["pd_regressor"])
    params.setdefault("random_state", int(cfg["model_params"]["random_state"]))
    matrix, names, categorical = build_feature_matrix(train, cfg)
    target = pd.to_numeric(train["Probability_of_Default"], errors="coerce")

    estimator = XGBRegressor(**params)
    estimator.fit(matrix, target)
    metrics = evaluate_probability(
        target, np.asarray(estimator.predict(matrix)), n_features=len(names)
    )
    logger.info("PD regressor trained", features=len(names), train_adj_r2=metrics["adj_r2"])
    return TrainedModel(
        name="pd_regressor",
        estimator=estimator,
        features=names,
        categorical_features=categorical,
        metrics={f"train_{k}": v for k, v in metrics.items()},
        params=params,
    )


def train_eligibility_classifier(
    train: pd.DataFrame, config: dict[str, Any] | None = None
) -> TrainedModel:
    """XGBoost classifier on ``Credit_Eligible``."""
    cfg = config if config is not None else load_config()
    params = dict(cfg["model_params"]["eligibility_classifier"])
    params.setdefault("random_state", int(cfg["model_params"]["random_state"]))
    matrix, names, categorical = build_feature_matrix(train, cfg)
    target = train["Credit_Eligible"].astype("string").eq("Yes").astype(int)

    estimator = XGBClassifier(**params)
    estimator.fit(matrix, target)
    probabilities = np.asarray(estimator.predict_proba(matrix))[:, 1]
    metrics = evaluate_classification(target, probabilities)
    logger.info("Eligibility classifier trained", features=len(names), train_auc=metrics["roc_auc"])
    return TrainedModel(
        name="eligibility_classifier",
        estimator=estimator,
        features=names,
        categorical_features=categorical,
        metrics={f"train_{k}": v for k, v in metrics.items()},
        params=params,
    )


def predict_pd(
    model: TrainedModel, frame: pd.DataFrame, config: dict[str, Any] | None = None
) -> NDArray[np.float64]:
    matrix, _, _ = build_feature_matrix(frame, config, features=model.features)
    return np.clip(np.asarray(model.estimator.predict(matrix), dtype="float64"), 0.0, 1.0)


def predict_eligibility(
    model: TrainedModel, frame: pd.DataFrame, config: dict[str, Any] | None = None
) -> NDArray[np.float64]:
    matrix, _, _ = build_feature_matrix(frame, config, features=model.features)
    return np.asarray(model.estimator.predict_proba(matrix), dtype="float64")[:, 1]


def feature_importance(
    model: TrainedModel, *, top_n: int | None = None
) -> list[dict[str, float | str]]:
    """Gain-based importances, normalised to sum to 1 so runs are comparable.

    Read off ``feature_importances_`` rather than the booster's own score map:
    the booster omits features it never split on, which would silently misalign
    the array against ``model.features``. The scikit-learn attribute is dense and
    positional, so ``zip(strict=True)`` below can actually catch a mismatch.

    Returned as an ordered list rather than a dict because the ranking *is* the
    information, and a dict loses it the moment it is serialised by anything that
    sorts keys - which our JSON writer does, deliberately, for stable diffs.
    """
    raw = np.asarray(model.estimator.feature_importances_, dtype="float64")
    total = raw.sum() or 1.0
    pairs = sorted(zip(model.features, (raw / total), strict=True), key=lambda kv: -kv[1])
    if top_n is not None:
        pairs = pairs[:top_n]
    return [
        {"rank": rank, "feature": name, "importance": round(float(value), 6)}
        for rank, (name, value) in enumerate(pairs, start=1)
    ]
