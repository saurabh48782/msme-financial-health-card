"""Layer D — the credit policy engine: risk band, eligibility, limit, terms.

Everything here is deterministic, versioned by ``policy_version``, and driven
entirely by ``params.yaml``. That is a regulatory requirement, not tidiness: under
RBI digital lending norms a declined applicant is entitled to a reason, and a
reason has to come from a rule you can point at and a version you can date.

Reverse-engineered from the dataset, and documented in ``params.yaml``:

* **Risk band** is a clean PD banding at 0.06 / 0.15.
* **Eligibility** is very nearly the PD gate alone - ``band != High`` reproduces
  37,165 of 37,174 eligible labels. The knock-out rules layered on top are *our*
  underwriting addition, not something the labels contain; their thresholds sit at
  the observed floor of the eligible population so the engine agrees with ground
  truth out of the box while still being a real, tunable gate. A credit policy team
  can tighten them without touching code.
* **Limit** is ``turnover x band_multiple x Cashflow_Score/100``, rounded to
  1,000 rupees (R^2 = 0.76 against ``Recommended_Credit_Limit_INR``). The cashflow
  scaling is the interesting part: the generator sizes the limit by the firm's
  ability to service it, not by turnover alone, which is also how a lender should.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.scoring.anomaly import RiskFlag
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

BAND_LOW = "Low"
BAND_MEDIUM = "Medium"
BAND_HIGH = "High"
BAND_ORDER = (BAND_LOW, BAND_MEDIUM, BAND_HIGH)
_MONTHS_PER_YEAR = 12


@dataclass(frozen=True)
class KnockOut:
    """A hard decline reason: the rule, the observed value and the limit breached."""

    code: str
    message: str
    metric: str
    value: float | None
    threshold: float | None


@dataclass
class PolicyDecision:
    """The full credit decision for one firm, with every reason attached."""

    policy_version: str
    probability_of_default: float
    risk_band: str
    eligible: bool
    credit_limit_inr: float
    tenor_months: int
    indicative_rate_pct: float
    knockouts: list[KnockOut] = field(default_factory=list)
    limit_basis: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def decline_reasons(self) -> list[str]:
        return [k.message for k in self.knockouts]


def risk_band_for(pd_value: float, config: dict[str, Any] | None = None) -> str:
    """PD -> risk band. Edges are inclusive of the lower band (PD 0.06 is Low)."""
    cfg = config if config is not None else load_config()
    bands = cfg["policy"]["pd_bands"]
    if pd_value <= float(bands["low_max"]):
        return BAND_LOW
    if pd_value <= float(bands["medium_max"]):
        return BAND_MEDIUM
    return BAND_HIGH


def risk_band_series(pd_values: pd.Series, config: dict[str, Any] | None = None) -> pd.Series:
    """Vectorised banding for the batch path."""
    cfg = config if config is not None else load_config()
    bands = cfg["policy"]["pd_bands"]
    values = pd.to_numeric(pd_values, errors="coerce")
    return pd.Series(
        np.select(
            [values <= float(bands["low_max"]), values <= float(bands["medium_max"])],
            [BAND_LOW, BAND_MEDIUM],
            default=BAND_HIGH,
        ),
        index=values.index,
        name="risk_band",
    )


def _knockouts(
    features: dict[str, Any],
    flags: list[RiskFlag],
    config: dict[str, Any],
) -> list[KnockOut]:
    spec = config["policy"]["knockouts"]
    if not bool(spec.get("enabled", True)):
        return []

    def value_of(name: str) -> float | None:
        raw = features.get(name)
        if raw is None:
            return None
        try:
            numeric = float(raw)
        except (TypeError, ValueError):
            return None
        return None if math.isnan(numeric) else numeric

    checks: list[tuple[str, str, str, str, float]] = [
        (
            "min_vintage",
            "Years_in_Operation",
            "min_years_in_operation",
            "Business is younger than the minimum vintage of {threshold:g} years",
            1.0,
        ),
        (
            "min_turnover",
            "Annual_Turnover_INR",
            "min_annual_turnover_inr",
            "Annual turnover is below the minimum of Rs {threshold:,.0f}",
            1.0,
        ),
        (
            "compliance_floor",
            "GST_Filing_Timeliness_Pct",
            "min_gst_filing_timeliness_pct",
            "GST filing timeliness is below the {threshold:g}% policy floor",
            1.0,
        ),
        (
            "overdraft_ceiling",
            "Overdraft_Usage_Ratio",
            "max_overdraft_usage_ratio",
            "Overdraft utilisation exceeds the {threshold:.0%} ceiling",
            -1.0,
        ),
    ]

    knockouts: list[KnockOut] = []
    for code, metric, key, template, direction in checks:
        if key not in spec:
            continue
        threshold = float(spec[key])
        observed = value_of(metric)
        if observed is None:
            continue
        breached = observed < threshold if direction > 0 else observed > threshold
        if breached:
            knockouts.append(
                KnockOut(
                    code=code,
                    message=template.format(threshold=threshold),
                    metric=metric,
                    value=round(observed, 6),
                    threshold=threshold,
                )
            )

    blocking = {str(s).lower() for s in spec.get("block_on_anomaly_severity", [])}
    for flag in flags:
        if flag.severity.lower() in blocking:
            knockouts.append(
                KnockOut(
                    code=f"anomaly_{flag.code}",
                    message=f"{flag.message} ({flag.severity} risk flag)",
                    metric=flag.metric,
                    value=flag.value,
                    threshold=flag.threshold,
                )
            )
    return knockouts


def _numeric_column(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    """A numeric Series for ``column``, or a constant Series when it is absent.

    ``DataFrame.get`` returns a scalar for a missing column, which then has no
    ``.fillna`` — this keeps the vectorised policy path working on a frame that is
    missing an optional input rather than raising three lines later.
    """
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def round_limit(amount: float, rounding: float) -> float:
    """Round down to the nearest ``rounding`` rupees — never sanction more than sized."""
    if rounding <= 0:
        return float(amount)
    return float(math.floor(amount / rounding) * rounding)


def annuity_principal(emi: Any, tenor_months: Any, annual_rate_pct: float) -> Any:
    """Present value of ``emi`` paid for ``tenor_months`` at ``annual_rate_pct``.

    Works on scalars and on numpy arrays, which is what keeps the scalar
    :func:`decide` and the vectorised :func:`decide_frame` arithmetically identical
    rather than merely similar.
    """
    monthly_rate = (annual_rate_pct / 100.0) / _MONTHS_PER_YEAR
    if monthly_rate <= 0:
        return emi * tenor_months
    # Standard annuity present value: EMI * (1 - (1+r)^-n) / r
    return (
        emi
        * (1.0 - (1.0 + monthly_rate) ** (-np.asarray(tenor_months, dtype="float64")))
        / monthly_rate
    )


def _dscr_cap(
    features: dict[str, Any], tenor_months: int, config: dict[str, Any]
) -> tuple[float | None, dict[str, float]]:
    """Largest principal the firm's free cashflow can service at the configured DSCR.

    Sizing a limit off turnover alone is how lenders end up with a borrower who
    cannot pay the instalment. This caps the offer at what the observed cashflow
    actually supports, and it binds on roughly 7% of this portfolio — it is a real
    constraint, not decoration.
    """
    spec = config["policy"]["limits"]["dscr_headroom"]
    if not bool(spec.get("enabled", True)):
        return None, {}

    credits = float(features.get("Monthly_Bank_Credits_INR") or 0.0)
    debits = float(features.get("Monthly_Bank_Debits_INR") or 0.0)
    existing_emi = float(features.get("Monthly_Loan_EMI_INR") or 0.0)
    free_cashflow = credits - debits + existing_emi
    if free_cashflow <= 0.0:
        return 0.0, {"free_monthly_cashflow": round(free_cashflow, 2)}

    min_dscr = float(spec["min_dscr"])
    affordable_emi = max(free_cashflow / min_dscr - existing_emi, 0.0)
    principal = float(
        annuity_principal(affordable_emi, tenor_months, float(spec["assumed_annual_rate_pct"]))
    )

    return principal, {
        "free_monthly_cashflow": round(free_cashflow, 2),
        "affordable_emi": round(affordable_emi, 2),
        "min_dscr": min_dscr,
        "dscr_capped_principal": round(principal, 2),
    }


def decide(
    features: pd.Series | dict[str, Any],
    probability_of_default: float,
    cashflow_score: float,
    flags: list[RiskFlag] | None = None,
    config: dict[str, Any] | None = None,
) -> PolicyDecision:
    """Apply the full policy to one firm.

    ``cashflow_score`` is the Layer A pillar, which is why the limit is a function
    of demonstrated cashflow rather than of turnover alone.
    """
    cfg = config if config is not None else load_config()
    policy = cfg["policy"]
    limits = policy["limits"]
    row = dict(features)
    flags = flags or []

    pd_value = float(np.clip(probability_of_default, 0.0, 1.0))
    band = risk_band_for(pd_value, cfg)
    knockouts = _knockouts(row, flags, cfg)
    eligible = band != BAND_HIGH and not knockouts

    tenor = int(policy["tenor_months"][band.lower()])
    rate = float(policy["base_rate_pct"]) + float(policy["pricing_spread_pct"][band.lower()])

    notes: list[str] = []
    basis: dict[str, float] = {}
    limit = 0.0

    if eligible:
        turnover = float(row.get("Annual_Turnover_INR") or 0.0)
        multiple = float(limits["band_multiple"][band.lower()])
        scaling = (
            float(np.clip(cashflow_score, 0.0, 100.0)) / 100.0
            if bool(limits.get("cashflow_scaling", True))
            else 1.0
        )
        sized = turnover * multiple * scaling
        basis = {
            "annual_turnover_inr": round(turnover, 2),
            "band_multiple": multiple,
            "cashflow_scaling": round(scaling, 6),
            "sized_limit_inr": round(sized, 2),
        }

        capped, dscr_basis = _dscr_cap(row, tenor, cfg)
        basis.update(dscr_basis)
        if capped is not None and capped < sized:
            sized = capped
            notes.append("Limit capped by debt-service headroom rather than turnover")

        limit = round_limit(sized, float(limits["rounding_inr"]))
        limit = min(limit, float(limits["max_limit_inr"]))
        minimum = float(limits["min_limit_inr"])
        if limit < minimum:
            # Below the minimum viable ticket the offer is not worth booking; say
            # so as a decline reason rather than sanctioning an unusable amount.
            notes.append(
                f"Sized limit Rs {limit:,.0f} is below the minimum ticket of Rs {minimum:,.0f}"
            )
            eligible = False
            limit = 0.0
            knockouts.append(
                KnockOut(
                    code="below_min_ticket",
                    message=f"Serviceable limit is below the minimum ticket of Rs {minimum:,.0f}",
                    metric="sized_limit_inr",
                    value=round(basis.get("sized_limit_inr", 0.0), 2),
                    threshold=minimum,
                )
            )
    else:
        notes.append(
            f"Risk band {band} is outside appetite"
            if band == BAND_HIGH
            else "Declined on policy knock-out rules"
        )

    return PolicyDecision(
        policy_version=str(cfg["policy_version"]),
        probability_of_default=round(pd_value, 6),
        risk_band=band,
        eligible=eligible,
        credit_limit_inr=round(limit, 2),
        tenor_months=tenor if eligible else 0,
        indicative_rate_pct=round(rate, 2) if eligible else 0.0,
        knockouts=knockouts,
        limit_basis=basis,
        notes=notes,
    )


def decide_frame(
    frame: pd.DataFrame,
    pd_values: pd.Series,
    cashflow_scores: pd.Series,
    config: dict[str, Any] | None = None,
    *,
    severity_column: str = "max_severity",
) -> pd.DataFrame:
    """Vectorised policy for the batch path.

    The knock-out rules are applied as column comparisons rather than by looping
    :func:`decide` over 50,000 rows, which keeps a full portfolio re-score to
    roughly a second. :func:`decide` remains the reference implementation and the
    unit tests assert the two agree.
    """
    cfg = config if config is not None else load_config()
    policy = cfg["policy"]
    limits = policy["limits"]
    spec = policy["knockouts"]

    band = risk_band_series(pd_values, cfg)
    out = pd.DataFrame(
        {"probability_of_default": pd_values.to_numpy(), "risk_band": band.to_numpy()},
        index=frame.index,
    )

    knocked = pd.Series(False, index=frame.index)
    reasons: list[pd.Series] = []
    if bool(spec.get("enabled", True)):
        checks = [
            ("min_vintage", "Years_in_Operation", "min_years_in_operation", True),
            ("min_turnover", "Annual_Turnover_INR", "min_annual_turnover_inr", True),
            (
                "compliance_floor",
                "GST_Filing_Timeliness_Pct",
                "min_gst_filing_timeliness_pct",
                True,
            ),
            ("overdraft_ceiling", "Overdraft_Usage_Ratio", "max_overdraft_usage_ratio", False),
        ]
        for code, metric, key, is_floor in checks:
            if key not in spec or metric not in frame.columns:
                continue
            threshold = float(spec[key])
            values = pd.to_numeric(frame[metric], errors="coerce")
            breached = (values < threshold) if is_floor else (values > threshold)
            breached = breached.fillna(False)
            knocked |= breached
            reasons.append(breached.map({True: code, False: ""}))

        blocking = {str(s).lower() for s in spec.get("block_on_anomaly_severity", [])}
        if blocking and severity_column in frame.columns:
            severe = (
                frame[severity_column].astype("string").str.lower().isin(blocking).fillna(False)
            )
            knocked |= severe
            reasons.append(severe.map({True: "anomaly_critical", False: ""}))

    # Being outside risk appetite is itself a decline reason and must be coded, or
    # 12,826 portfolio rows read as "declined, no reason given".
    reasons.append((band == BAND_HIGH).map({True: "risk_band_high", False: ""}))
    out["knockout_codes"] = (
        pd.concat(reasons, axis=1).apply(lambda r: ",".join(c for c in r if c), axis=1)
        if reasons
        else ""
    )
    eligible = (band != BAND_HIGH) & ~knocked
    out["eligible"] = eligible

    turnover = _numeric_column(frame, "Annual_Turnover_INR")
    multiple = (
        band.str.lower()
        .map({k: float(v) for k, v in limits["band_multiple"].items()})
        .astype("float64")
    )
    scaling = (
        pd.to_numeric(cashflow_scores, errors="coerce").clip(0.0, 100.0).fillna(0.0) / 100.0
        if bool(limits.get("cashflow_scaling", True))
        else 1.0
    )
    sized = turnover * multiple * scaling

    # Same debt-service cap the scalar path applies. Omitting it here would let the
    # batch portfolio and a real-time card disagree on the same firm, which is the
    # kind of divergence nobody notices until an auditor does.
    tenor_map = {k: int(v) for k, v in policy["tenor_months"].items()}
    dscr = limits["dscr_headroom"]
    if bool(dscr.get("enabled", True)):
        credits = _numeric_column(frame, "Monthly_Bank_Credits_INR")
        debits = _numeric_column(frame, "Monthly_Bank_Debits_INR")
        existing_emi = _numeric_column(frame, "Monthly_Loan_EMI_INR")
        free_cashflow = credits - debits + existing_emi
        affordable_emi = (free_cashflow / float(dscr["min_dscr"]) - existing_emi).clip(lower=0.0)
        tenor = band.str.lower().map(tenor_map).fillna(0).astype("float64")
        capped = pd.Series(
            annuity_principal(
                affordable_emi.to_numpy(), tenor.to_numpy(), float(dscr["assumed_annual_rate_pct"])
            ),
            index=frame.index,
        )
        capped = capped.where(free_cashflow > 0.0, 0.0)
        out["dscr_capped"] = capped < sized
        sized = pd.Series(np.minimum(sized.to_numpy(), capped.to_numpy()), index=frame.index)
    else:
        out["dscr_capped"] = False

    rounding = float(limits["rounding_inr"])
    limit = np.floor(sized / rounding) * rounding if rounding > 0 else sized
    limit = np.minimum(limit, float(limits["max_limit_inr"]))
    limit = np.where(eligible.to_numpy(), limit, 0.0)
    limit = np.where(limit < float(limits["min_limit_inr"]), 0.0, limit)
    out["credit_limit_inr"] = np.round(limit, 2)

    # A sized limit below the minimum ticket is a decline, matching decide(). Record
    # the reason: a portfolio row that says "declined" with no code is unauditable,
    # and this branch is the one that fires for over-levered firms whose free
    # cashflow cannot service even the minimum facility.
    below_ticket = eligible.to_numpy() & (out["credit_limit_inr"].to_numpy() <= 0.0)
    out["eligible"] = out["eligible"] & (out["credit_limit_inr"] > 0.0)
    ticket_code = np.where(
        below_ticket & out["dscr_capped"].to_numpy(), "below_min_ticket_dscr", ""
    )
    ticket_code = np.where(
        below_ticket & ~out["dscr_capped"].to_numpy(), "below_min_ticket", ticket_code
    )
    out["knockout_codes"] = [
        ",".join(part for part in (existing, extra) if part)
        for existing, extra in zip(out["knockout_codes"], ticket_code, strict=True)
    ]

    spread_map = {k: float(v) for k, v in policy["pricing_spread_pct"].items()}
    lower = band.str.lower()
    out["tenor_months"] = np.where(out["eligible"], lower.map(tenor_map).fillna(0).astype(int), 0)
    out["indicative_rate_pct"] = np.where(
        out["eligible"], float(policy["base_rate_pct"]) + lower.map(spread_map).fillna(0.0), 0.0
    )
    out["policy_version"] = str(cfg["policy_version"])

    logger.info(
        "Policy applied",
        rows=len(out),
        approval_rate=round(float(out["eligible"].mean()), 4),
        band_mix=out["risk_band"].value_counts().to_dict(),
    )
    return out
