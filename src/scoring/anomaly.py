"""Layer C - anomaly detection by explicit, named rules.

Each rule encodes something a credit officer knows is wrong: GST purchases
exceeding sales, payroll consuming 80% of turnover, revenue contracting while the
overdraft climbs. A rule is named, severity-tagged and carries the number that
tripped it, so a flag is defensible in an audit and actionable by a human - the
reviewer can see *which* invariant broke and go look at that.

The rules live here rather than in ``params.yaml`` because they are domain
invariants, not tunables: purchases above sales is a negative gross margin at any
threshold, and nobody reviewing credit policy would ever flip its comparison. What
*is* policy - which severities knock out eligibility - stays in
``policy.knockouts.block_on_anomaly_severity``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

SEVERITY_ORDER: dict[str, int] = {"low": 1, "medium": 2, "high": 3, "critical": 4}
_SEVERITY_BY_RANK: dict[int, str] = {rank: name for name, rank in SEVERITY_ORDER.items()}


@dataclass(frozen=True)
class Rule:
    """One invariant: a metric, the edge it may not cross, and how bad that is."""

    code: str
    severity: str
    metric: str
    threshold: float
    trips_above: bool  # False -> the threshold is a floor the metric must stay above
    message: str


RULES: tuple[Rule, ...] = (
    Rule(
        "negative_free_cashflow",
        "critical",
        "credit_debit_ratio",
        0.95,
        False,
        "Bank debits persistently exceed credits",
    ),
    Rule(
        "purchases_exceed_sales",
        "high",
        "purchases_to_sales_ratio",
        1.0,
        True,
        "GST purchases exceed GST sales - negative gross margin",
    ),
    Rule(
        "revenue_source_divergence",
        "high",
        "revenue_triangulation_gap",
        0.25,
        True,
        "GST, bank and UPI revenue disagree by more than 25%",
    ),
    Rule(
        "contraction_with_overdraft",
        "high",
        "contraction_overdraft_signal",
        0.0,
        True,
        "Revenue contracting while overdraft usage is elevated",
    ),
    Rule(
        "gst_bank_mismatch",
        "medium",
        "gst_bank_reconciliation_gap",
        0.20,
        True,
        "Declared GST sales differ materially from bank credits",
    ),
    Rule(
        "payroll_disproportionate",
        "medium",
        "payroll_to_turnover_ratio",
        0.80,
        True,
        "Payroll consumes over 80% of turnover",
    ),
    Rule(
        "customer_concentration",
        "medium",
        "Customer_Concentration_Ratio",
        0.85,
        True,
        "Over 85% of revenue depends on a single customer",
    ),
    Rule(
        "thin_cash_buffer",
        "medium",
        "balance_days_of_outflow",
        3.0,
        False,
        "Cash buffer covers under 3 days of outflow",
    ),
)


@dataclass(frozen=True)
class RiskFlag:
    """One triggered rule, with the evidence that triggered it."""

    code: str
    severity: str
    message: str
    metric: str
    value: float | None
    threshold: float

    @property
    def severity_rank(self) -> int:
        return SEVERITY_ORDER.get(self.severity, 0)


def _breached(values: pd.Series, rule: Rule) -> pd.Series:
    """Boolean mask of rows that break ``rule``.

    A NaN metric is unknown, not a violation - flagging on missing data would
    penalise exactly the thin-file firms this system is meant to include.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    breach = numeric > rule.threshold if rule.trips_above else numeric < rule.threshold
    return breach.fillna(False).astype(bool)


def flags_for_row(row: pd.Series | dict[str, Any]) -> list[RiskFlag]:
    """Triggered rules for one firm, most severe first."""
    getter = row.get
    flags: list[RiskFlag] = []
    for rule in RULES:
        raw = getter(rule.metric)
        try:
            value = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if np.isnan(value):
            continue
        if not (value > rule.threshold if rule.trips_above else value < rule.threshold):
            continue
        flags.append(
            RiskFlag(
                code=rule.code,
                severity=rule.severity,
                message=rule.message,
                metric=rule.metric,
                value=round(value, 6),
                threshold=rule.threshold,
            )
        )
    return sorted(flags, key=lambda f: (-f.severity_rank, f.code))


def detect(frame: pd.DataFrame) -> pd.DataFrame:
    """Batch form: one ``flag_<code>`` column per rule, plus ``flag_count`` and
    ``max_severity``."""
    triggered = pd.DataFrame(index=frame.index)
    rank = pd.Series(0, index=frame.index, dtype="int64")
    for rule in RULES:
        if rule.metric not in frame.columns:
            logger.warning(
                "Anomaly rule metric missing from frame", rule=rule.code, metric=rule.metric
            )
            breach = pd.Series(False, index=frame.index, dtype="bool")
        else:
            breach = _breached(frame[rule.metric], rule)
        triggered[f"flag_{rule.code}"] = breach
        rank = rank.mask(breach, rank.clip(lower=SEVERITY_ORDER[rule.severity]))

    out = triggered.copy()
    out["flag_count"] = triggered.sum(axis=1).astype(int)
    out["max_severity"] = rank.map(_SEVERITY_BY_RANK.get)

    logger.info(
        "Anomalies detected",
        rows=len(out),
        flagged=int((out["flag_count"] > 0).sum()),
    )
    return out
