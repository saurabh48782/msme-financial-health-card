"""Deterministic test doubles and fixtures — defined once, used everywhere.

A double lives here rather than in the suite that first needed it, so an
interface change is fixed in one place instead of hunted across files.

Note what is *not* here: there is no stub card store. The real
:class:`~src.data_access.csv_store.CsvCardStore` takes its two frames by
constructor injection, so the integration suite drives the production store over
in-memory frames rather than a parallel implementation that could drift from it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.schemas.healthcard import (
    CreditRecommendation,
    HealthCard,
    PillarScore,
)

if TYPE_CHECKING:  # pragma: no cover - avoids importing the scoring stack at collection
    from src.scoring.scoring_orchestration import ScoringBundle

# Two firms with deliberately opposite outcomes, so the integration suite can
# exercise both policy branches over real HTTP.
ELIGIBLE_ID = "MSME9000001"
INELIGIBLE_ID = "MSME9000002"

_CATEGORICALS: dict[str, list[str]] = {
    "Customer_Segment": ["NTC", "NTB", "Existing-to-Credit"],
    "Business_Type": ["Proprietorship", "Partnership", "LLP", "Private Limited"],
    "Industry": ["Retail", "Services", "Manufacturing", "Trading", "Logistics"],
    "Location_Category": ["Metro", "Tier 1", "Tier 2", "Tier 3", "Rural"],
    "GST_Return_Frequency": ["Monthly", "Quarterly"],
}


_THIN_FILE_FIRM: dict[str, Any] = {
    "Has_Existing_Loan": "No",
    "EMI_On_Time_Rate_Pct": np.nan,
    "Monthly_Loan_EMI_INR": 0.0,
    "Annual_Turnover_INR": 1_200_000.0,
    "Monthly_Bank_Credits_INR": 100_000.0,
    "Monthly_Bank_Debits_INR": 80_000.0,
    "Average_Bank_Balance_INR": 50_000.0,
    "Monthly_GST_Sales_INR": 98_000.0,
    "Monthly_GST_Purchases_INR": 60_000.0,
    "Monthly_UPI_Inflow_INR": 30_000.0,
    "Monthly_UPI_Outflow_INR": 20_000.0,
    "UPI_Avg_Ticket_Size_INR": 500.0,
    "UPI_Daily_Txn_Count": 8.0,
    "Monthly_Payroll_INR": 20_000.0,
    "Employee_Count": 4.0,
    "Revenue_Growth_Rate_Pct": 5.0,
    "Overdraft_Usage_Ratio": 0.2,
}


def thin_file_frame(**overrides: Any) -> pd.DataFrame:
    """One debt-free firm as a raw single-row frame, ready for the feature pipeline.

    ``EMI_On_Time_Rate_Pct`` is the structural NULL. ``overrides`` lets a test zero
    a single denominator without restating seventeen columns around it.
    """
    return pd.DataFrame([{**_THIN_FILE_FIRM, **overrides}])


def synthetic_raw_frame(rows: int = 200, *, seed: int = 7) -> pd.DataFrame:
    """A dataset-shaped frame, including the structural NULL relationship.

    Used when the DVC-tracked CSV is unavailable (CI). It reproduces the property
    the tests care about: ``EMI_On_Time_Rate_Pct`` is null exactly where
    ``Has_Existing_Loan == "No"``.
    """
    rng = np.random.default_rng(seed)
    turnover = rng.uniform(3e5, 5e7, rows)
    monthly = turnover / 12.0
    has_loan = rng.random(rows) < 0.35

    frame = pd.DataFrame(
        {
            "MSME_ID": [f"MSME{i:07d}" for i in range(1, rows + 1)],
            "Years_in_Operation": rng.uniform(0.3, 20.0, rows).round(1),
            "Employee_Count": rng.integers(1, 60, rows),
            "Annual_Turnover_INR": turnover.round(0),
            "Monthly_GST_Sales_INR": (monthly * rng.uniform(0.85, 1.12, rows)).round(0),
            "Monthly_GST_Purchases_INR": (monthly * rng.uniform(0.35, 0.85, rows)).round(0),
            "GST_Filing_Timeliness_Pct": rng.uniform(55.0, 100.0, rows).round(1),
            "Monthly_UPI_Inflow_INR": (monthly * rng.uniform(0.05, 0.8, rows)).round(0),
            "Monthly_UPI_Outflow_INR": (monthly * rng.uniform(0.03, 0.5, rows)).round(0),
            "UPI_Avg_Ticket_Size_INR": rng.uniform(100, 40000, rows).round(0),
            "UPI_Daily_Txn_Count": rng.uniform(0.5, 40.0, rows).round(1),
            "Transaction_Volatility_Index": rng.uniform(0.03, 0.7, rows).round(3),
            "Monthly_Bank_Credits_INR": (monthly * rng.uniform(0.9, 1.15, rows)).round(0),
            "Monthly_Bank_Debits_INR": (monthly * rng.uniform(0.7, 1.0, rows)).round(0),
            "Average_Bank_Balance_INR": (monthly * rng.uniform(0.05, 1.5, rows)).round(0),
            "Cashflow_Stability_Index": rng.uniform(0.3, 0.98, rows).round(3),
            "Overdraft_Usage_Ratio": rng.uniform(0.0, 0.58, rows).round(3),
            "Monthly_Payroll_INR": (monthly * rng.uniform(0.03, 0.5, rows)).round(0),
            "Salary_Consistency_Pct": rng.uniform(55.0, 100.0, rows).round(1),
            "Employee_Attrition_Rate_Pct": rng.uniform(1.0, 44.0, rows).round(1),
            "Avg_Invoice_Payment_Delay_Days": rng.uniform(0.0, 90.0, rows).round(1),
            "Customer_Concentration_Ratio": rng.uniform(0.05, 0.95, rows).round(3),
            "Vendor_Payment_Timeliness_Pct": rng.uniform(50.0, 100.0, rows).round(1),
            "Seasonality_Index": rng.uniform(0.0, 1.0, rows).round(3),
            "Revenue_Growth_Rate_Pct": rng.uniform(-50.0, 70.0, rows).round(1),
            "Has_Existing_Loan": np.where(has_loan, "Yes", "No"),
            "Monthly_Loan_EMI_INR": np.where(has_loan, (monthly * 0.12).round(0), 0.0),
            "EMI_On_Time_Rate_Pct": np.where(
                has_loan, rng.uniform(70.0, 100.0, rows).round(1), np.nan
            ),
            "Credit_History_Months": np.where(has_loan, rng.integers(6, 180, rows), 0),
        }
    )
    for column, levels in _CATEGORICALS.items():
        frame[column] = rng.choice(levels, rows)

    # Label block, so calibration and evaluation paths have something to fit.
    frame["Business_Stability_Score"] = rng.uniform(40, 97, rows).round(1)
    frame["Cashflow_Score"] = rng.uniform(38, 96, rows).round(1)
    frame["Revenue_Consistency_Score"] = rng.uniform(35, 92, rows).round(1)
    frame["Payment_Behaviour_Score"] = rng.uniform(52, 100, rows).round(1)
    frame["Business_Growth_Score"] = rng.uniform(3, 83, rows).round(1)
    frame["Compliance_Score"] = rng.uniform(52, 100, rows).round(1)
    frame["Financial_Health_Score"] = (
        0.272 * frame["Compliance_Score"]
        + 0.238 * frame["Cashflow_Score"]
        + 0.225 * frame["Payment_Behaviour_Score"]
        + 0.103 * frame["Revenue_Consistency_Score"]
        + 0.085 * frame["Business_Stability_Score"]
        + 0.077 * frame["Business_Growth_Score"]
    ).round(1)
    frame["Probability_of_Default"] = np.clip(
        np.exp(-0.1284 * frame["Financial_Health_Score"] + 7.138), 0.002, 0.75
    ).round(4)
    frame["Credit_Risk_Category"] = np.select(
        [frame["Probability_of_Default"] <= 0.06, frame["Probability_of_Default"] <= 0.15],
        ["Low", "Medium"],
        default="High",
    )
    frame["Credit_Eligible"] = np.where(frame["Credit_Risk_Category"] != "High", "Yes", "No")
    frame["Recommended_Credit_Limit_INR"] = np.where(
        frame["Credit_Eligible"] == "Yes",
        (frame["Annual_Turnover_INR"] * 0.12 * frame["Cashflow_Score"] / 100 / 1000).round(0)
        * 1000,
        0.0,
    )
    return frame


def make_card(
    msme_id: str = ELIGIBLE_ID,
    *,
    eligible: bool = True,
    score: float = 78.5,
) -> HealthCard:
    """A minimal but schema-valid card, for tests that only need a card shape."""
    return HealthCard(
        msme_id=msme_id,
        financial_health_score=score,
        grade="A" if eligible else "D",
        grade_label="Strong" if eligible else "Weak",
        customer_segment="NTC",
        industry="Retail",
        location_category="Tier 2",
        pillars=[
            PillarScore(
                key="compliance",
                label="Compliance",
                score=score,
                weight=0.272,
                contribution=round(score * 0.272, 4),
                baseline=62.6,
                thin_file=False,
                drivers=[],
            )
        ],
        credit=CreditRecommendation(
            eligible=eligible,
            risk_band="Low" if eligible else "High",
            probability_of_default=0.03 if eligible else 0.42,
            credit_limit_inr=500_000.0 if eligible else 0.0,
            tenor_months=36 if eligible else 0,
            indicative_rate_pct=11.5 if eligible else 0.0,
            decline_reasons=[] if eligible else ["Risk band High is outside appetite"],
            policy_version="policy-1.0.0",
        ),
        risk_flags=[],
        thin_file=True,
        model_version="stub",
        policy_version="policy-1.0.0",
        rubric_version="rubric-1.0.0",
        feature_snapshot_hash="0123456789abcdef",
        scored_at=datetime.now(UTC),
    )


def rubric_only_bundle() -> ScoringBundle:
    """A bundle with no ML heads — exercises the fitted PD fallback path.

    The import is deferred so this module stays importable by tests that only need
    ``make_card`` or ``synthetic_raw_frame``, without pulling in the whole scoring
    stack at collection time.
    """
    from src.scoring.scoring_orchestration import ScoringBundle

    return ScoringBundle()
