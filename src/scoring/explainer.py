"""Layer E - explanations a credit officer can read out loud.

Two independent witnesses, deliberately:

1. **The pillar waterfall** (Layer A) is *exactly* decomposable. Every point of
   every pillar is attributable to a named driver with the observed value beside
   it, and the arithmetic closes. This is what goes in an adverse-action notice,
   because it is arithmetic rather than attribution.
2. **SHAP on the PD model** (Layer B) explains the *statistical* risk view, in
   percentage points of default probability. It catches interactions the rubric
   cannot express, and it disagrees with the rubric sometimes - which is
   informative, not embarrassing.

Both are rendered as plain-English reason codes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.scoring.pillars import NEUTRAL_SCORE, SCORE_MAX, PillarResult, pillar_band
from src.utils.config import load_config
from src.utils.logger import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from src.model.components import TrainedModel

logger = get_logger(__name__)

_PP = 100.0  # probability -> percentage points


@dataclass(frozen=True)
class ReasonCode:
    """One plain-English driver of the decision."""

    source: str  # "pillar" | "model"
    code: str
    pillar: str | None
    text: str
    impact: float  # pillar points, or percentage points of PD
    unit: str
    direction: str  # "positive" | "negative"
    value: float | None = None
    formatted_value: str | None = None


@dataclass
class Explanation:
    reason_codes: list[ReasonCode] = field(default_factory=list)
    pillar_waterfall: list[dict[str, Any]] = field(default_factory=list)
    shap_base_value: float | None = None
    shap_contributions: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def strengths(self) -> list[ReasonCode]:
        return [r for r in self.reason_codes if r.direction == "positive"]

    @property
    def weaknesses(self) -> list[ReasonCode]:
        return [r for r in self.reason_codes if r.direction == "negative"]


def format_value(source: str, value: float | None) -> str:
    """Render a driver value in its natural unit, inferred from the column name.

    Keeping the unit out of ``params.yaml`` avoids a config field that would have
    to be maintained in lockstep with the column naming convention it duplicates.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "not available"
    if source.endswith("_Pct"):
        return f"{value:.1f}%"
    if source.endswith("_INR") or source.startswith("monthly_turnover"):
        if abs(value) >= 1e7:
            return f"Rs {value / 1e7:.2f} cr"
        if abs(value) >= 1e5:
            return f"Rs {value / 1e5:.2f} lakh"
        return f"Rs {value:,.0f}"
    if source.endswith("_Days"):
        return f"{value:.0f} days"
    if source.endswith("_Count"):
        return f"{value:.0f}"
    if source == "Years_in_Operation":
        return f"{value:.1f} years"
    if source.endswith(("_Ratio", "_gap", "_ratio")):
        return f"{value:.0%}"
    if source.endswith("_standing"):
        return f"{value:.2f}"
    return f"{value:,.2f}"


def pillar_reason_codes(
    pillar_results: dict[str, PillarResult],
    config: dict[str, Any] | None = None,
) -> list[ReasonCode]:
    """Reason codes from the exact pillar decomposition, largest impact first."""
    cfg = config if config is not None else load_config()
    spec = cfg["explainability"]
    minimum = float(spec["min_abs_contribution"])
    codes: list[ReasonCode] = []

    for key, result in pillar_results.items():
        lo, hi = pillar_band(key, cfg)
        span = hi - lo
        for driver in result.contributions:
            if not driver.available:
                continue
            formatted = format_value(driver.source, driver.value)
            if driver.driver_score >= NEUTRAL_SCORE:
                impact = driver.contribution
                if impact < minimum:
                    continue
                text = (
                    f"Healthy {driver.label} ({formatted}) contributed "
                    f"{impact:.1f} points to {result.label}"
                )
                direction = "positive"
            else:
                # Points forgone against a perfect driver: what the firm can recover.
                impact = (SCORE_MAX - driver.driver_score) * driver.weight * span / SCORE_MAX
                if impact < minimum:
                    continue
                text = (
                    f"Weak {driver.label} ({formatted}) cost {impact:.1f} points on {result.label}"
                )
                direction = "negative"
            codes.append(
                ReasonCode(
                    source="pillar",
                    code=f"{key}.{driver.name}",
                    pillar=key,
                    text=text,
                    impact=round(impact, 3),
                    unit="points",
                    direction=direction,
                    value=driver.value,
                    formatted_value=formatted,
                )
            )

        if result.thin_file:
            codes.append(
                ReasonCode(
                    source="pillar",
                    code=f"{key}.thin_file",
                    pillar=key,
                    text=(
                        f"No formal repayment track record — {result.label} was scored "
                        f"on the remaining evidence rather than assuming a history"
                    ),
                    impact=0.0,
                    unit="points",
                    direction="positive",
                    value=None,
                    formatted_value=None,
                )
            )

    return sorted(codes, key=lambda c: -c.impact)


class ShapExplainer:
    """SHAP TreeExplainer over an XGBoost head.

    Uses the tree-path-dependent algorithm, which needs no background dataset —
    important because the alternative is shipping a 200-row sample inside the
    registry artifact and hoping it stays representative.
    """

    def __init__(self, model: TrainedModel, config: dict[str, Any] | None = None) -> None:
        import shap  # imported lazily: shap pulls in numba and is slow to import

        self.config = config if config is not None else load_config()
        self.model = model
        self.explainer = shap.TreeExplainer(model.estimator)
        self.feature_labels = {name: _humanise(name) for name in model.features}

    def contributions(self, matrix: pd.DataFrame) -> tuple[NDArray[np.float64], float]:
        """``(shap_values, base_value)`` for the rows in ``matrix``."""
        values = self.explainer.shap_values(matrix)
        if isinstance(values, list):  # classifier: one array per class
            values = values[-1]
        array = np.asarray(values, dtype="float64")
        if array.ndim == 3:  # (rows, features, classes)
            array = array[:, :, -1]
        expected = self.explainer.expected_value
        if isinstance(expected, (list, np.ndarray)):
            base = float(np.asarray(expected).ravel()[-1])
        else:
            base = float(expected)
        return array, base

    def explain_row(
        self, matrix: pd.DataFrame, *, top_n: int | None = None
    ) -> tuple[list[dict[str, Any]], float]:
        """Per-feature PD contributions for a single-row matrix, strongest first."""
        if len(matrix) != 1:
            raise ValueError(f"explain_row expects exactly one row, got {len(matrix)}")
        values, base = self.contributions(matrix)
        limit = top_n or int(self.config["explainability"]["top_n_drivers"])
        row = values[0]
        order = np.argsort(-np.abs(row))[:limit]
        contributions = [
            {
                "feature": self.model.features[int(i)],
                "label": self.feature_labels[self.model.features[int(i)]],
                "value": _cell_value(matrix, int(i)),
                "shap": round(float(row[int(i)]), 8),
                "shap_pp": round(float(row[int(i)]) * _PP, 4),
            }
            for i in order
        ]
        return contributions, base


def _cell_value(matrix: pd.DataFrame, position: int) -> float | None:
    """The numeric value at ``[0, position]``, or None for a categorical/missing cell."""
    if not pd.api.types.is_numeric_dtype(matrix.dtypes.iloc[position]):
        return None
    cell = matrix.iat[0, position]
    if cell is None or pd.isna(cell):
        return None
    return float(cell)  # type: ignore[arg-type]


def _humanise(name: str) -> str:
    """Column name -> readable label, without a hand-maintained lookup table."""
    cleaned = name.replace("_INR", "").replace("_Pct", "").replace("_", " ").strip()
    replacements = {
        "gst": "GST",
        "upi": "UPI",
        "emi": "EMI",
        "dscr": "DSCR",
        "epfo": "EPFO",
        "pd": "PD",
    }
    words = [replacements.get(w.lower(), w) for w in cleaned.split()]
    label = " ".join(words)
    return label[0].upper() + label[1:] if label else name


def shap_reason_codes(
    contributions: list[dict[str, Any]], config: dict[str, Any] | None = None
) -> list[ReasonCode]:
    """Turn SHAP contributions into sentences about default probability."""
    _ = config if config is not None else load_config()
    codes: list[ReasonCode] = []
    for item in contributions:
        impact_pp = float(item["shap_pp"])
        if abs(impact_pp) < 0.01:
            continue
        feature = str(item["feature"])
        formatted = format_value(feature, item.get("value"))
        raised = impact_pp > 0
        verb = "raised" if raised else "lowered"
        codes.append(
            ReasonCode(
                source="model",
                code=f"model.{feature}",
                pillar=None,
                text=(
                    f"{item['label']} ({formatted}) {verb} estimated default "
                    f"probability by {abs(impact_pp):.2f} pp"
                ),
                impact=round(abs(impact_pp), 4),
                unit="pp_default_probability",
                direction="negative" if raised else "positive",
                value=item.get("value"),
                formatted_value=formatted,
            )
        )
    return sorted(codes, key=lambda c: -c.impact)


def build_explanation(
    pillar_results: dict[str, PillarResult],
    shap_contributions: list[dict[str, Any]] | None = None,
    shap_base_value: float | None = None,
    config: dict[str, Any] | None = None,
) -> Explanation:
    """Combine both witnesses into one explanation payload."""
    cfg = config if config is not None else load_config()
    top_n = int(cfg["explainability"]["top_n_drivers"])

    codes = pillar_reason_codes(pillar_results, cfg)
    if shap_contributions:
        codes = codes + shap_reason_codes(shap_contributions, cfg)

    strengths = [c for c in codes if c.direction == "positive"][:top_n]
    weaknesses = [c for c in codes if c.direction == "negative"][:top_n]

    waterfall = [
        {
            "pillar": result.key,
            "label": result.label,
            "score": result.score,
            "baseline": result.baseline,
            "thin_file": result.thin_file,
            "drivers": [
                {
                    "name": driver.name,
                    "label": driver.label,
                    "value": driver.value,
                    "formatted_value": format_value(driver.source, driver.value),
                    "driver_score": driver.driver_score,
                    "weight": driver.weight,
                    "contribution": driver.contribution,
                    "available": driver.available,
                }
                for driver in result.contributions
            ],
        }
        for result in pillar_results.values()
    ]

    notes: list[str] = []
    if any(r.thin_file for r in pillar_results.values()):
        notes.append(
            "This firm has no formal credit history. Scores rely on GST, banking, "
            "UPI and payroll evidence; no repayment behaviour was assumed."
        )

    return Explanation(
        reason_codes=[*strengths, *weaknesses],
        pillar_waterfall=waterfall,
        shap_base_value=round(shap_base_value, 8) if shap_base_value is not None else None,
        shap_contributions=shap_contributions or [],
        notes=notes,
    )
