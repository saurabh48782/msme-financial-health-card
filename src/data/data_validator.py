"""Schema, range and structural-null assertions on the raw dataset.

Validation is deliberately loud about one thing in particular: the structural
NULLs. ``EMI_On_Time_Rate_Pct`` is null for exactly the firms with no existing
loan. If that ever stops being true the dataset has changed shape, the
``has_repayment_history`` feature stops meaning what it means, and the payment
pillar silently mis-scores 65% of the portfolio. So it is asserted, not assumed.

The range check does not carry its own bounds. It reads them off
:class:`~src.schemas.msme.MSMEFeatures`, the pydantic contract the API already
rejects bad payloads with, so there is exactly one definition of each column's
domain and the batch CSV path cannot drift away from the edge.
The outliers are preprocessing's job, clipped at percentiles the data decides.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, get_args

import numpy as np
import pandas as pd
from annotated_types import Ge, Le
from pydantic.fields import FieldInfo

from src.schemas.msme import MSMEFeatures
from src.utils.config import load_config
from src.utils.custom_error import CustomError
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationReport:
    """The outcome of validating a frame. ``errors`` block, ``warnings`` do not."""

    rows: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    null_counts: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            raise CustomError("Dataset validation failed: " + "; ".join(self.errors))


def _constraints(info: FieldInfo) -> Iterator[Any]:
    """Every constraint on a field.

    Optional fields such as ``EMI_On_Time_Rate_Pct`` annotate as
    ``Optional[Annotated[float, FieldInfo(...)]]``, nesting their ``Ge``/``Le``
    one ``FieldInfo`` deeper than a plain field does, so unwrap that too.
    """
    yield from info.metadata
    for argument in get_args(info.annotation):
        for nested in getattr(argument, "__metadata__", ()):
            if isinstance(nested, FieldInfo):
                yield from nested.metadata
            else:
                yield nested


@lru_cache(maxsize=1)
def invariant_bounds() -> dict[str, tuple[float, float]]:
    """``{dataset column: (low, high)}`` read off the API contract.

    Infinite on a side the contract leaves open. Cached because the model is
    immutable once imported.
    """
    bounds: dict[str, tuple[float, float]] = {}
    for info in MSMEFeatures.model_fields.values():
        if not info.alias:
            continue
        low, high = -np.inf, np.inf
        for constraint in _constraints(info):
            if isinstance(constraint, Ge):
                low = float(constraint.ge)  # type: ignore[arg-type]
            elif isinstance(constraint, Le):
                high = float(constraint.le)  # type: ignore[arg-type]
        if low != -np.inf or high != np.inf:
            bounds[info.alias] = (low, high)
    return bounds


def _check_invariants(frame: pd.DataFrame, report: ValidationReport) -> None:
    """Definitional bounds. A violation means the ingestion contract was broken,
    not that a firm is unusual - winsorisation handles unusual."""
    for column, (low, high) in invariant_bounds().items():
        if column not in frame.columns:
            continue
        series = pd.to_numeric(frame[column], errors="coerce")
        offending = int(((series < low) | (series > high)).sum())
        if offending:
            report.errors.append(
                f"{column}: {offending} value(s) outside the domain [{low}, {high}]"
            )


def _check_categoricals(
    frame: pd.DataFrame, config: dict[str, Any], report: ValidationReport
) -> None:
    standing = config.get("categorical_standing", {})
    driver_sources = {
        "Business_Type": "business_type_standing",
        "Location_Category": "location_standing",
        "GST_Return_Frequency": "gst_return_frequency_standing",
    }
    for column, standing_key in driver_sources.items():
        if column not in frame.columns or standing_key not in standing:
            continue
        known = set(standing[standing_key])
        seen = {str(v) for v in frame[column].dropna().unique()}
        unknown = seen - known
        if unknown:
            # A new level would score 0 on that driver, quietly penalising a firm.
            report.errors.append(
                f"{column}: level(s) {sorted(unknown)} have no categorical_standing entry"
            )


def _check_structural_nulls(
    frame: pd.DataFrame, config: dict[str, Any], report: ValidationReport
) -> None:
    rules: dict[str, str] = config["schema"].get("structural_null_columns", {})
    for column, condition in rules.items():
        if column not in frame.columns:
            continue
        try:
            expected_null = frame.eval(condition)
        except (ValueError, SyntaxError, pd.errors.UndefinedVariableError) as error:
            report.errors.append(f"structural null rule for {column!r} is not evaluable: {error}")
            continue
        actual_null = frame[column].isna()
        expected_series = pd.Series(expected_null, index=frame.index).astype(bool)
        mismatch = int((expected_series != actual_null).sum())
        if mismatch:
            report.errors.append(
                f"{column}: nullness disagrees with its structural rule "
                f"({condition}) on {mismatch} row(s)"
            )


def validate(frame: pd.DataFrame, config: dict[str, Any] | None = None) -> ValidationReport:
    """Validate a raw or feature frame. Never mutates the input."""
    cfg = config if config is not None else load_config()
    schema = cfg["schema"]
    report = ValidationReport(rows=len(frame))

    if frame.empty:
        report.errors.append("dataset is empty")
        return report

    id_column = schema["id_column"]
    if id_column not in frame.columns:
        report.errors.append(f"missing id column {id_column!r}")
    else:
        duplicates = int(frame[id_column].duplicated().sum())
        if duplicates:
            report.errors.append(f"{id_column}: {duplicates} duplicate identifier(s)")
        if frame[id_column].isna().any():
            report.errors.append(f"{id_column}: contains nulls")

    missing = [c for c in schema["feature_columns"] if c not in frame.columns]
    if missing:
        report.errors.append(f"missing feature column(s): {missing}")

    leaked = [c for c in schema["label_columns"] if c in frame.columns]
    if leaked:
        report.warnings.append(f"label column(s) present - must not reach a model: {leaked}")

    _check_invariants(frame, report)
    _check_categoricals(frame, cfg, report)
    _check_structural_nulls(frame, cfg, report)

    structural = set(schema.get("structural_null_columns", {}))
    for column in schema["feature_columns"]:
        if column not in frame.columns:
            continue
        nulls = int(frame[column].isna().sum())
        if nulls:
            report.null_counts[column] = nulls
            if column not in structural:
                report.errors.append(f"{column}: {nulls} unexpected null(s)")

    logger.info(
        "Dataset validated",
        rows=report.rows,
        ok=report.ok,
        errors=len(report.errors),
        warnings=len(report.warnings),
        structural_nulls=report.null_counts,
    )
    return report
