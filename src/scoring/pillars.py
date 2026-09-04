"""Layer A - the six transparent pillar rubrics.

Design constraint that shapes everything here: **the decomposition must be exact**.
A credit officer declining a firm has to be able to say "your Compliance score is
71 because late GST filings cost you 12 points", and that arithmetic has to add
up. So a pillar is one linear interpolation across a fitted band::

    fraction = sum_i (effective_weight_i * ramp_i)   # 0-1
    score    = lo + (hi - lo) * fraction

``lo`` is the score of a firm sitting at every driver's ``worst`` anchor and
``hi`` of a firm at every ``best``. The band is fitted against the dataset's own
pillar columns (see :mod:`src.scoring.pillar_calibration`) because a rubric built
from domain anchors ranks firms correctly but lands on its own scale; fitting the
band makes the transparent rubric provably track ground truth instead of merely
claiming to.

Because the effective weights always sum to 1, the band is a property of the
pillar rather than a transform bolted on after it. The score is inside ``[lo, hi]``
by construction, so nothing is ever clamped and the contributions always sum to
the reported score without correction. Each ramp is clamped and linear between its
anchors, so every driver is monotone and every contribution is a real number of
points.

**Missing drivers redistribute rather than impute.** ``EMI_On_Time_Rate_Pct`` is
structurally NULL for the 65% of firms with no loan. Imputing the mean would hand
a thin-file firm a fabricated repayment record - exactly the bias this project
exists to remove. Instead the driver's weight is spread pro-rata over the drivers
that *are* observed, and ``thin_file`` is set so the card can say so out loud.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

SCORE_MIN = 0.0
SCORE_MAX = 100.0
NEUTRAL_SCORE = 50.0
_WEIGHT_EPS = 1e-9


@dataclass(frozen=True)
class DriverContribution:
    """One driver's exact contribution to one pillar, in pillar points."""

    name: str
    label: str
    source: str
    value: float | None
    driver_score: float
    weight: float
    contribution: float
    available: bool


@dataclass
class PillarResult:
    """A pillar score plus the contributions that exactly sum to it."""

    key: str
    label: str
    score: float
    baseline: float
    contributions: list[DriverContribution] = field(default_factory=list)
    thin_file: bool = False
    unavailable_drivers: list[str] = field(default_factory=list)

    def decomposition_error(self) -> float:
        """``|score - (baseline + sum(contributions))|``, asserted ~0 by the tests."""
        return abs(self.score - (self.baseline + sum(c.contribution for c in self.contributions)))


def pillar_keys(config: dict[str, Any] | None = None) -> list[str]:
    """Pillar keys in reporting order: heaviest FHS weight first."""
    cfg = config if config is not None else load_config()
    weights = cfg["pillars"]["fhs_weights"]
    return sorted(weights, key=lambda k: (-float(weights[k]), k))


def pillar_labels(config: dict[str, Any] | None = None) -> dict[str, str]:
    cfg = config if config is not None else load_config()
    return {key: str(cfg["pillars"][key]["label"]) for key in pillar_keys(cfg)}


def pillar_band(pillar_key: str, config: dict[str, Any] | None = None) -> tuple[float, float]:
    """``(lo, hi)`` - the fitted score band, worst-case firm to best-case firm."""
    cfg = config if config is not None else load_config()
    lo, hi = cfg["pillars"][pillar_key]["band"]
    return float(lo), float(hi)


def driver_label(name: str, driver: dict[str, Any]) -> str:
    """Narration label: the configured override, else the humanised driver name."""
    return str(driver.get("label", name.replace("_", " ")))


def ramp(
    values: pd.Series | NDArray[np.float64] | float, worst: float, best: float
) -> NDArray[np.float64]:
    """Map values onto 0-1 by a clamped linear ramp from ``worst`` to ``best``.

    A ``best`` below ``worst`` is how a lower-is-better driver is written - the
    ramp simply runs downhill, and no separate direction flag has to be kept in
    step with the anchors. NaN in, NaN out: availability is the caller's decision,
    never silently made here.
    """
    array = np.asarray(values, dtype="float64")
    span = float(best) - float(worst)
    if span == 0.0:  # guarded by validate_config; defensive for direct callers
        raise ValueError(f"zero-width ramp: worst == best == {worst}")
    fraction: NDArray[np.float64] = np.clip((array - float(worst)) / span, 0.0, 1.0)
    return fraction


def driver_scores(
    frame: pd.DataFrame, pillar_key: str, config: dict[str, Any] | None = None
) -> pd.DataFrame:
    """0-100 ramp score per driver for one pillar. NaN where the source is absent."""
    cfg = config if config is not None else load_config()
    spec: dict[str, dict[str, Any]] = cfg["pillars"][pillar_key]["drivers"]
    out = pd.DataFrame(index=frame.index, dtype="float64")
    for name, driver in spec.items():
        source = str(driver["source"])
        if source not in frame.columns:
            # A configured driver whose source was never engineered is a wiring
            # bug, but failing the whole portfolio is worse than dropping it loudly.
            logger.warning(
                "Pillar driver source missing from frame",
                pillar=pillar_key,
                driver=name,
                source=source,
            )
            out[name] = np.nan
            continue
        out[name] = (
            ramp(
                pd.to_numeric(frame[source], errors="coerce"),
                float(driver["worst"]),
                float(driver["best"]),
            )
            * SCORE_MAX
        )
    return out


def _effective_weights(
    scores: pd.DataFrame, pillar_key: str, config: dict[str, Any]
) -> pd.DataFrame:
    """Nominal weights, zeroed where unobserved and renormalised row-wise to 1."""
    spec: dict[str, dict[str, Any]] = config["pillars"][pillar_key]["drivers"]
    nominal = pd.DataFrame(
        {
            name: np.where(scores[name].notna(), float(driver["weight"]), 0.0)
            for name, driver in spec.items()
        },
        index=scores.index,
    )
    total = nominal.sum(axis=1)
    return nominal.div(total.where(total > _WEIGHT_EPS), axis=0).fillna(0.0)


def compute_pillar(
    frame: pd.DataFrame, pillar_key: str, config: dict[str, Any] | None = None
) -> pd.Series:
    """Vectorised score for one pillar over a whole frame.

    A firm with no observed driver at all cannot be scored; NaN beats a bogus band
    floor that would read as a real, merely poor, score.
    """
    cfg = config if config is not None else load_config()
    scores = driver_scores(frame, pillar_key, cfg)
    effective = _effective_weights(scores, pillar_key, cfg)
    lo, hi = pillar_band(pillar_key, cfg)
    fraction = (scores.fillna(0.0) * effective).sum(axis=1) / SCORE_MAX
    return (lo + (hi - lo) * fraction).where(effective.sum(axis=1) > _WEIGHT_EPS)


def compute_pillars(frame: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """All six pillar scores as ``<pillar_key>_score`` columns."""
    cfg = config if config is not None else load_config()
    out = pd.DataFrame(index=frame.index)
    for key in pillar_keys(cfg):
        out[f"{key}_score"] = compute_pillar(frame, key, cfg)
    logger.info(
        "Pillars computed",
        rows=len(out),
        nulls=int(out.isna().any(axis=1).sum()),
    )
    return out


def explain_pillar(
    row: pd.Series | dict[str, Any], pillar_key: str, config: dict[str, Any] | None = None
) -> PillarResult:
    """Full decomposition of one pillar for one firm.

    The band floor is surfaced as an explicit ``baseline`` term rather than being
    smeared across the drivers - it is the score a firm already holds with every
    driver at its worst anchor, and attributing it to a driver would credit points
    that driver did not earn. ``baseline + sum(contributions) == score`` holds
    exactly and needs no clamping correction, because the score cannot leave the
    band in the first place.
    """
    cfg = config if config is not None else load_config()
    frame = pd.DataFrame([dict(row)])
    spec: dict[str, dict[str, Any]] = cfg["pillars"][pillar_key]["drivers"]
    scores = driver_scores(frame, pillar_key, cfg)
    effective = _effective_weights(scores, pillar_key, cfg)
    lo, hi = pillar_band(pillar_key, cfg)
    span = hi - lo

    contributions: list[DriverContribution] = []
    unavailable: list[str] = []
    for name, driver in spec.items():
        weight = float(effective.iloc[0][name])
        available = weight > _WEIGHT_EPS
        if not available:
            unavailable.append(name)
        driver_score = scores.iloc[0][name]
        driver_score = float(driver_score) if pd.notna(driver_score) else 0.0
        source = str(driver["source"])
        raw_value = frame.iloc[0].get(source)
        contributions.append(
            DriverContribution(
                name=name,
                label=driver_label(name, driver),
                source=source,
                value=(
                    float(raw_value)
                    if pd.notna(raw_value) and isinstance(raw_value, (int, float, np.number))
                    else None
                ),
                driver_score=round(driver_score, 4),
                weight=round(weight, 6),
                contribution=round(span * weight * driver_score / SCORE_MAX, 4),
                available=available,
            )
        )

    fraction = float((scores.fillna(0.0).iloc[0] * effective.iloc[0]).sum()) / SCORE_MAX
    thin_file = any(name in unavailable and bool(spec[name].get("optional")) for name in spec)
    return PillarResult(
        key=pillar_key,
        label=str(cfg["pillars"][pillar_key]["label"]),
        score=round(lo + span * fraction, 4),
        baseline=round(lo, 4),
        contributions=sorted(contributions, key=lambda c: -c.contribution),
        thin_file=thin_file,
        unavailable_drivers=unavailable,
    )


def explain_pillars(
    row: pd.Series | dict[str, Any], config: dict[str, Any] | None = None
) -> dict[str, PillarResult]:
    """Full decomposition of all six pillars for one firm."""
    cfg = config if config is not None else load_config()
    return {key: explain_pillar(row, key, cfg) for key in pillar_keys(cfg)}
