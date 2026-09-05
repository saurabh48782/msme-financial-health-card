"""The MSME feature contract - the 33 consent-based alternative-data inputs.

Field names are snake_case for Python; each carries an ``alias`` equal to the
dataset / GSTN-style column name, and ``populate_by_name`` is on, so an integrator
can post either spelling. A malformed payload is rejected at the edge with a 422
that names the field rather than becoming a NaN five layers down.

The bounds below are the *only* definition of each column's domain in the project:
``src.data.data_validator`` reads them straight back off these fields, so the
batch CSV path and the API can never drift apart.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

CustomerSegment = Literal["NTC", "NTB", "Existing-to-Credit"]
BusinessType = Literal["Proprietorship", "Partnership", "LLP", "Private Limited"]
LocationCategory = Literal["Metro", "Tier 1", "Tier 2", "Tier 3", "Rural"]
GstReturnFrequency = Literal["Monthly", "Quarterly"]
YesNo = Literal["Yes", "No"]

Pct = Annotated[float, Field(ge=0.0, le=100.0)]
Ratio = Annotated[float, Field(ge=0.0, le=1.0)]
Money = Annotated[float, Field(ge=0.0)]
# A magnitude, count or duration: floored, with no defensible ceiling.
NonNegative = Annotated[float, Field(ge=0.0)]
Headcount = Annotated[int, Field(ge=1)]
# A firm cannot shrink by more than all of itself; growth has no ceiling.
Growth = Annotated[float, Field(ge=-100.0)]


class MSMEFeatures(BaseModel):
    """One MSME's alternative-data snapshot."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid", str_strip_whitespace=True)

    msme_id: str = Field(alias="MSME_ID", examples=["MSME0000001"], min_length=1, max_length=64)

    # --- firmographics -------------------------------------------------------
    customer_segment: CustomerSegment = Field(alias="Customer_Segment", examples=["NTC"])
    business_type: BusinessType = Field(alias="Business_Type", examples=["Proprietorship"])
    industry: str = Field(alias="Industry", examples=["Retail"], min_length=1, max_length=64)
    location_category: LocationCategory = Field(alias="Location_Category", examples=["Tier 2"])
    years_in_operation: NonNegative = Field(alias="Years_in_Operation", examples=[4.3])
    employee_count: Headcount = Field(alias="Employee_Count", examples=[3])
    annual_turnover_inr: Money = Field(alias="Annual_Turnover_INR", examples=[5_391_103.0])

    # --- GSTN ----------------------------------------------------------------
    monthly_gst_sales_inr: Money = Field(alias="Monthly_GST_Sales_INR", examples=[437_000.0])
    monthly_gst_purchases_inr: Money = Field(
        alias="Monthly_GST_Purchases_INR", examples=[271_000.0]
    )
    gst_filing_timeliness_pct: Pct = Field(alias="GST_Filing_Timeliness_Pct", examples=[88.4])
    gst_return_frequency: GstReturnFrequency = Field(
        alias="GST_Return_Frequency", examples=["Monthly"]
    )

    # --- UPI / NPCI ----------------------------------------------------------
    monthly_upi_inflow_inr: Money = Field(alias="Monthly_UPI_Inflow_INR", examples=[151_000.0])
    monthly_upi_outflow_inr: Money = Field(alias="Monthly_UPI_Outflow_INR", examples=[92_000.0])
    upi_avg_ticket_size_inr: Money = Field(alias="UPI_Avg_Ticket_Size_INR", examples=[540.0])
    upi_daily_txn_count: NonNegative = Field(alias="UPI_Daily_Txn_Count", examples=[9.4])
    transaction_volatility_index: NonNegative = Field(
        alias="Transaction_Volatility_Index", examples=[0.273]
    )

    # --- bank / Account Aggregator -------------------------------------------
    monthly_bank_credits_inr: Money = Field(alias="Monthly_Bank_Credits_INR", examples=[450_000.0])
    monthly_bank_debits_inr: Money = Field(alias="Monthly_Bank_Debits_INR", examples=[359_000.0])
    average_bank_balance_inr: Money = Field(alias="Average_Bank_Balance_INR", examples=[222_000.0])
    cashflow_stability_index: Ratio = Field(alias="Cashflow_Stability_Index", examples=[0.682])
    overdraft_usage_ratio: Ratio = Field(alias="Overdraft_Usage_Ratio", examples=[0.241])

    # --- credit --------------------------------------------------------------
    has_existing_loan: YesNo = Field(alias="Has_Existing_Loan", examples=["No"])
    monthly_loan_emi_inr: Money = Field(alias="Monthly_Loan_EMI_INR", examples=[0.0])
    # Structurally NULL whenever has_existing_loan is "No". Never imputed - see
    # src/scoring/pillars.py for how the weight is redistributed instead.
    emi_on_time_rate_pct: Pct | None = Field(
        default=None, alias="EMI_On_Time_Rate_Pct", examples=[None]
    )
    credit_history_months: NonNegative = Field(alias="Credit_History_Months", examples=[0.0])

    # --- EPFO / payroll ------------------------------------------------------
    monthly_payroll_inr: Money = Field(alias="Monthly_Payroll_INR", examples=[64_000.0])
    salary_consistency_pct: Pct = Field(alias="Salary_Consistency_Pct", examples=[84.8])
    employee_attrition_rate_pct: Pct = Field(alias="Employee_Attrition_Rate_Pct", examples=[15.7])

    # --- invoices / receivables ----------------------------------------------
    avg_invoice_payment_delay_days: NonNegative = Field(
        alias="Avg_Invoice_Payment_Delay_Days", examples=[12.4]
    )
    customer_concentration_ratio: Ratio = Field(
        alias="Customer_Concentration_Ratio", examples=[0.399]
    )
    vendor_payment_timeliness_pct: Pct = Field(
        alias="Vendor_Payment_Timeliness_Pct", examples=[82.2]
    )

    # --- trend ---------------------------------------------------------------
    seasonality_index: NonNegative = Field(alias="Seasonality_Index", examples=[0.337])
    revenue_growth_rate_pct: Growth = Field(alias="Revenue_Growth_Rate_Pct", examples=[9.3])

    @field_validator("emi_on_time_rate_pct")
    @classmethod
    def _repayment_rate_needs_a_loan(cls, value: float | None) -> float | None:
        return value

    def to_row(self) -> dict[str, object]:
        """Dataset-column-named dict, ready for the feature pipeline."""
        return self.model_dump(by_alias=True)


class ScoreRequest(BaseModel):
    """Stateless scoring request: one or many MSMEs, no persistence."""

    model_config = ConfigDict(extra="forbid")

    records: list[MSMEFeatures] = Field(min_length=1)
    include_explanation: bool = Field(
        default=True, description="Attach SHAP drivers and the pillar waterfall"
    )


class SimulationRequest(BaseModel):
    """What-if: override named features and re-score."""

    model_config = ConfigDict(extra="forbid")

    overrides: dict[str, float | int | str] = Field(
        min_length=1,
        examples=[{"GST_Filing_Timeliness_Pct": 98.0, "Overdraft_Usage_Ratio": 0.1}],
        description="Dataset column name -> replacement value",
    )
