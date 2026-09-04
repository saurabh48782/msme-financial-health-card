"""Derived features: the ratios a credit officer would actually compute.

Two conventions:
* **Safe denominators.** Division uses ``preprocessing.min_denominator`` as a
  floor, so a firm with zero bank debits yields a large-but-finite ratio instead
  of an inf that later becomes a silent NaN.
* **Structural nulls are preserved.** ``has_repayment_history`` is derived, but
  ``EMI_On_Time_Rate_Pct`` itself is left NaN. "No track record" is a state the
  model is told about, never a number we invent.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Columns produced here, in the order they are documented in the model card.
DERIVED_COLUMNS: tuple[str, ...] = (
    "has_repayment_history",
    "monthly_turnover_run_rate",
    "net_monthly_cashflow",
    "credit_debit_ratio",
    "balance_days_of_outflow",
    "dscr_proxy",
    "upi_share_of_credits",
    "purchases_to_sales_ratio",
    "gst_turnover_realisation",
    "gst_bank_reconciliation_gap",
    "revenue_triangulation_gap",
    "payroll_to_turnover_ratio",
    "turnover_per_employee",
    "contraction_overdraft_signal",
)

_DSCR_CAP = 25.0  # a firm with no debt has infinite coverage; cap for a finite feature
_CONTRACTION_GROWTH_PCT = -10.0  # revenue shrinking, not merely flat
_ELEVATED_OVERDRAFT_RATIO = 0.35  # the overdraft is being lived on, not dipped into


def _safe_divide(numerator: pd.Series, denominator: pd.Series, floor: float) -> pd.Series:
    """Elementwise division with the denominator floored away from zero."""
    safe = denominator.astype("float64").abs().clip(lower=floor)
    return numerator.astype("float64") / safe


def _relative_gap(left: pd.Series, right: pd.Series, floor: float) -> pd.Series:
    """``|left - right| / max(left, right)`` - a symmetric, scale-free disagreement."""
    scale = pd.concat([left.abs(), right.abs()], axis=1).max(axis=1).clip(lower=floor)
    return (left - right).abs() / scale


def add_derived_features(frame: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Append every derived column."""

    cfg = config if config is not None else load_config()
    floor = float(cfg["preprocessing"]["min_denominator"])
    out = frame.copy()

    # repayment track record
    has_loan = out["Has_Existing_Loan"].astype("string").eq("Yes")
    out["has_repayment_history"] = (has_loan & out["EMI_On_Time_Rate_Pct"].notna()).astype(bool)

    # scale normalisers
    out["monthly_turnover_run_rate"] = out["Annual_Turnover_INR"] / 12.0

    # cashflow
    out["net_monthly_cashflow"] = out["Monthly_Bank_Credits_INR"] - out["Monthly_Bank_Debits_INR"]
    out["credit_debit_ratio"] = _safe_divide(
        out["Monthly_Bank_Credits_INR"], out["Monthly_Bank_Debits_INR"], floor
    )
    out["balance_days_of_outflow"] = _safe_divide(
        out["Average_Bank_Balance_INR"], out["Monthly_Bank_Debits_INR"] / 30.0, floor
    )
    # Free cashflow before debt service, over debt service. No debt -> capped high.
    free_cashflow = out["net_monthly_cashflow"] + out["Monthly_Loan_EMI_INR"]
    out["dscr_proxy"] = np.where(
        out["Monthly_Loan_EMI_INR"] > floor,
        _safe_divide(free_cashflow, out["Monthly_Loan_EMI_INR"], floor),
        _DSCR_CAP,
    )
    out["dscr_proxy"] = out["dscr_proxy"].clip(upper=_DSCR_CAP)

    # Digital rails. UPI is a subset of bank credits, so this is a containment
    # check rather than a revenue estimate — it feeds revenue_triangulation_gap.
    out["upi_share_of_credits"] = _safe_divide(
        out["Monthly_UPI_Inflow_INR"], out["Monthly_Bank_Credits_INR"], floor
    )

    # GST
    out["purchases_to_sales_ratio"] = _safe_divide(
        out["Monthly_GST_Purchases_INR"], out["Monthly_GST_Sales_INR"], floor
    )
    out["gst_turnover_realisation"] = _safe_divide(
        out["Monthly_GST_Sales_INR"], out["monthly_turnover_run_rate"], floor
    )
    # Compliance lens: are declared GST sales consistent with money actually banked?
    out["gst_bank_reconciliation_gap"] = _relative_gap(
        out["Monthly_GST_Sales_INR"], out["Monthly_Bank_Credits_INR"], floor
    )

    # Consistency lens: do all three independent revenue witnesses agree?
    # GSTN (declared sales), bank/AA (credits) and the firm's own declared turnover.
    # UPI is a subset of bank credits, so it enters as a containment check rather
    # than a fourth revenue estimate.
    gaps = pd.concat(
        [
            out["gst_bank_reconciliation_gap"],
            _relative_gap(out["Monthly_GST_Sales_INR"], out["monthly_turnover_run_rate"], floor),
            _relative_gap(out["Monthly_Bank_Credits_INR"], out["monthly_turnover_run_rate"], floor),
            (out["upi_share_of_credits"] - 1.0).clip(lower=0.0),
        ],
        axis=1,
    )
    out["revenue_triangulation_gap"] = gaps.max(axis=1)

    # payroll / productivity
    out["payroll_to_turnover_ratio"] = _safe_divide(
        out["Monthly_Payroll_INR"] * 12.0, out["Annual_Turnover_INR"], floor
    )
    out["turnover_per_employee"] = _safe_divide(
        out["Annual_Turnover_INR"], out["Employee_Count"], 1.0
    )

    # Composite stress signal: shrinking revenue funded by the overdraft. Either
    # alone is ordinary; together they are the classic pre-default pattern, which
    # is why the anomaly layer flags the conjunction and not the parts.
    out["contraction_overdraft_signal"] = (
        (out["Revenue_Growth_Rate_Pct"] < _CONTRACTION_GROWTH_PCT)
        & (out["Overdraft_Usage_Ratio"] > _ELEVATED_OVERDRAFT_RATIO)
    ).astype("float64")

    out = out.replace([np.inf, -np.inf], np.nan)
    logger.info(
        "Derived features added",
        rows=len(out),
        derived=len(DERIVED_COLUMNS),
        thin_file_rows=int((~out["has_repayment_history"]).sum()),
    )
    return out
