"""The assessment form is derived from ``MSMEFeatures``, not typed out beside it.

These tests are the guard on that: a field added to the contract must appear on
the page, and the bounds the browser enforces must be the bounds the API
validates against. A hand-maintained form would drift from both silently.
"""

from __future__ import annotations

import annotated_types as at

from src.api.form_spec import FormField, assess_form_sections
from src.schemas.msme import MSMEFeatures


def rendered_field(name: str) -> FormField:
    """The rendered input for one dataset column."""
    for _, fields in assess_form_sections():
        for rendered in fields:
            if rendered.name == name:
                return rendered
    raise AssertionError(f"{name} is not rendered")


class TestCoverage:
    def test_every_contract_field_is_rendered_exactly_once(self) -> None:
        rendered = [field.name for _, fields in assess_form_sections() for field in fields]
        aliases = [str(info.alias) for info in MSMEFeatures.model_fields.values()]
        assert rendered == aliases

    def test_no_field_lands_in_the_fallback_section(self) -> None:
        """A new field inherits the section it was written into, never "Other"."""
        assert "Other" not in {title for title, _ in assess_form_sections()}


class TestFieldTypes:
    def test_literals_become_selects_carrying_every_choice(self) -> None:
        field = rendered_field("Customer_Segment")
        assert field.kind == "select"
        assert field.choices == ("NTC", "NTB", "Existing-to-Credit")

    def test_identifiers_stay_free_text(self) -> None:
        assert rendered_field("MSME_ID").kind == "text"

    def test_headcount_is_a_whole_number(self) -> None:
        field = rendered_field("Employee_Count")
        assert (field.kind, field.step, field.min) == ("number", "1", 1.0)

    def test_bounds_come_from_the_contract(self) -> None:
        field = rendered_field("GST_Filing_Timeliness_Pct")
        assert (field.min, field.max) == (0.0, 100.0)

    def test_a_floored_field_has_no_invented_ceiling(self) -> None:
        field = rendered_field("Annual_Turnover_INR")
        assert (field.min, field.max) == (0.0, None)

    def test_contraction_beyond_total_loss_is_still_rejected(self) -> None:
        assert rendered_field("Revenue_Growth_Rate_Pct").min == -100.0

    def test_the_structurally_null_field_is_optional_and_still_bounded(self) -> None:
        """``Pct | None`` hides its bounds on the inner ``Annotated``; losing them
        would let the browser post a 200% repayment rate."""
        field = rendered_field("EMI_On_Time_Rate_Pct")
        assert field.required is False
        assert (field.min, field.max) == (0.0, 100.0)

    def test_every_numeric_bound_matches_the_model(self) -> None:
        for name, info in MSMEFeatures.model_fields.items():
            field = rendered_field(str(info.alias))
            if field.kind != "number":
                continue
            metadata = list(info.metadata)
            expected_min = next((float(m.ge) for m in metadata if isinstance(m, at.Ge)), None)
            expected_max = next((float(m.le) for m in metadata if isinstance(m, at.Le)), None)
            if name == "emi_on_time_rate_pct":  # bounds live on the inner Annotated
                expected_min, expected_max = 0.0, 100.0
            assert (field.min, field.max) == (expected_min, expected_max), name


class TestLabels:
    def test_a_trailing_unit_becomes_a_unit_not_part_of_the_name(self) -> None:
        field = rendered_field("Monthly_GST_Sales_INR")
        assert (field.label, field.unit) == ("Monthly GST sales", "₹")

    def test_acronyms_survive_the_humaniser(self) -> None:
        assert rendered_field("Monthly_UPI_Inflow_INR").label == "Monthly UPI inflow"

    def test_examples_prefill_the_form(self) -> None:
        assert rendered_field("GST_Filing_Timeliness_Pct").example == "88.4"
