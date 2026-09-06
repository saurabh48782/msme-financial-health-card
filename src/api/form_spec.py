"""The assessment form's field spec, derived from ``MSMEFeatures`` itself.

The form is not a second copy of the feature contract: every input, its type, its
bounds and its example value are read straight off the pydantic model, exactly as
``src.data.data_validator`` reads the bounds for the batch path. Add a field to
``MSMEFeatures`` and it appears on the page; change a bound and the browser
enforces the new one. Only the section headings and a handful of unreadable
auto-generated labels live here.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union, get_args, get_origin

import annotated_types as at

from src.schemas.msme import MSMEFeatures

# The first field of each section. Anything after it belongs to that section, so a
# new field added to the model lands in the section it was written into.
SECTIONS: dict[str, str] = {
    "msme_id": "Identity and firmographics",
    "monthly_gst_sales_inr": "GST returns (GSTN)",
    "monthly_upi_inflow_inr": "Digital payments (UPI / NPCI)",
    "monthly_bank_credits_inr": "Bank account (Account Aggregator)",
    "has_existing_loan": "Credit history",
    "monthly_payroll_inr": "Payroll (EPFO)",
    "avg_invoice_payment_delay_days": "Invoices and receivables",
    "seasonality_index": "Trend",
}

# Only the names the generic humaniser gets wrong.
LABELS: dict[str, str] = {
    "MSME_ID": "MSME ID",
    "UPI_Daily_Txn_Count": "UPI transactions per day",
    "EMI_On_Time_Rate_Pct": "EMI on-time repayment rate",
    "Avg_Invoice_Payment_Delay_Days": "Average invoice payment delay",
}

# Words the humaniser must not lowercase.
ACRONYMS = frozenset({"GST", "UPI", "EMI", "MSME", "ID", "EPFO", "INR"})

# Trailing tokens that are a unit rather than part of the name.
UNITS: dict[str, str] = {"INR": "₹", "PCT": "%", "DAYS": "days", "MONTHS": "months"}

HINTS: dict[str, str] = {
    "EMI_On_Time_Rate_Pct": "Leave blank when the firm has no existing loan — the weight is "
    "redistributed rather than imputed.",
    "Transaction_Volatility_Index": "Coefficient of variation of daily transaction value.",
    "Cashflow_Stability_Index": "1.0 is perfectly steady month-to-month cash flow.",
    "Customer_Concentration_Ratio": "Revenue share of the single largest customer.",
    "Seasonality_Index": "Amplitude of the annual revenue cycle.",
}


class FormField:
    """One rendered input. Plain attributes; the template reads them directly."""

    def __init__(self, name: str, info: Any) -> None:
        alias = str(info.alias or name)
        annotation, optional = _unwrap(info.annotation)
        metadata = list(info.metadata) or _inner_metadata(info.annotation)

        self.name = alias
        self.label, self.unit = _humanise(alias)
        self.hint = HINTS.get(alias)
        self.required = not optional
        self.choices: tuple[str, ...] = ()
        self.kind = "text"
        self.min: float | None = None
        self.max: float | None = None
        self.step: str | None = None

        if get_origin(annotation) is Literal:
            self.kind = "select"
            self.choices = tuple(str(choice) for choice in get_args(annotation))
        elif annotation in (int, float):
            self.kind = "number"
            self.step = "1" if annotation is int else "any"
            for item in metadata:
                if isinstance(item, at.Ge):
                    self.min = float(item.ge)  # type: ignore[arg-type]
                elif isinstance(item, at.Le):
                    self.max = float(item.le)  # type: ignore[arg-type]

        example = (info.examples or [None])[0]
        self.example = "" if example is None else str(example)


def assess_form_sections() -> list[tuple[str, list[FormField]]]:
    """The 33 inputs, grouped into the sections a credit analyst reads them in."""
    sections: list[tuple[str, list[FormField]]] = []
    for name, info in MSMEFeatures.model_fields.items():
        if name in SECTIONS or not sections:
            sections.append((SECTIONS.get(name, "Other"), []))
        sections[-1][1].append(FormField(name, info))
    return sections


def _unwrap(annotation: Any) -> tuple[Any, bool]:
    """Strip ``Annotated`` and ``| None``, reporting whether the field was optional."""
    optional = False
    while True:
        origin = get_origin(annotation)
        if origin is Annotated:
            annotation = get_args(annotation)[0]
        elif origin is Union or origin is type(int | str):
            args = [arg for arg in get_args(annotation) if arg is not type(None)]
            optional = len(args) != len(get_args(annotation))
            annotation = args[0]
        else:
            return annotation, optional


def _inner_metadata(annotation: Any) -> list[Any]:
    """Bounds declared inside an ``Optional[Annotated[...]]`` alias.

    ``Pct | None`` keeps its ``Ge``/``Le`` on the inner ``Annotated``, where
    ``FieldInfo.metadata`` cannot see them.
    """
    for arg in get_args(annotation):
        if get_origin(arg) is Annotated:
            for item in get_args(arg)[1:]:
                metadata = getattr(item, "metadata", None)
                if metadata:
                    return list(metadata)
    return []


def _humanise(alias: str) -> tuple[str, str]:
    """``Monthly_GST_Sales_INR`` -> ``("Monthly GST sales", "₹")``."""
    tokens = alias.split("_")
    unit = ""
    if len(tokens) > 1 and tokens[-1].upper() in UNITS:
        unit = UNITS[tokens.pop().upper()]
    if alias in LABELS:
        return LABELS[alias], unit
    words = [token if token.upper() in ACRONYMS else token.lower() for token in tokens]
    label = " ".join(words)
    return label[:1].upper() + label[1:], unit
