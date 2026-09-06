"""Server-rendered dashboard.

The views call the same service layer the JSON API does, so a number on a page and
the same number from ``/api/v1`` can never disagree. Templates receive plain
dicts and pydantic models — no view-model layer, because there is nothing here a
template cannot read directly.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from typing import Any

import numpy as np
from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from pydantic import BaseModel

from src.api.deps import ConfigDep, PortfolioDep, ServiceDep
from src.api.form_spec import assess_form_sections
from src.api.routers.portfolio import load_training_report
from src.api.templates import templates
from src.service.healthcard_service import MSMENotFoundError
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["dashboard"], include_in_schema=False)

SEGMENTS = ("NTC", "NTB", "Existing-to-Credit")
GRADES = ("A+", "A", "B", "C", "D")
BANDS = ("Low", "Medium", "High")


def _chart_payload(value: Any) -> str:
    """Serialise a chart payload to JSON for a ``<script type=application/json>`` block.

    Done here rather than with Jinja's ``tojson`` filter because these are pydantic
    models: ``tojson`` cannot serialise them, and reaching for ``__dict__`` in a
    template silently produces nested model objects instead of data.
    """
    return json.dumps(value, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return str(value)


def _distinct(frame: Any, column: str) -> list[str]:
    """Sorted unique values of a portfolio column, for a <select> or <datalist>."""
    if frame is None or frame.empty or column not in frame.columns:
        return []
    return sorted(str(value) for value in frame[column].dropna().unique())


def _form_value(value: Any) -> str:
    """A stored profile value as an ``<input value=...>``.

    Floats are trimmed because a form is not a place to show
    ``0.27300000000000002``, and a missing value must render as an empty input
    rather than the string ``nan``.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return str(int(number)) if number.is_integer() else f"{round(number, 4)}"
    return str(value)


def _context(request: Request, config: dict[str, Any], **extra: Any) -> dict[str, Any]:
    bundle = getattr(request.app.state, "bundle", None)
    return {
        "request": request,
        "policy_version": config["policy_version"],
        "rubric_version": config["rubric_version"],
        "model_version": getattr(bundle, "model_version", None),
        **extra,
    }


@router.get("/", response_class=HTMLResponse)
async def homepage(request: Request, config: ConfigDep, portfolio: PortfolioDep) -> HTMLResponse:
    summary = await portfolio.summary()
    return templates.TemplateResponse(
        request, "homepage.html", _context(request, config, summary=summary)
    )


@router.get("/portfolio", response_class=HTMLResponse)
async def view_portfolio(
    request: Request, config: ConfigDep, portfolio: PortfolioDep
) -> HTMLResponse:
    summary = await portfolio.summary()
    return templates.TemplateResponse(
        request,
        "view_portfolio.html",
        _context(
            request,
            config,
            summary=summary,
            chart_data=_chart_payload(
                {
                    "histogram": summary.health_score_histogram,
                    "bands": summary.risk_band_mix,
                    "grades": summary.grade_mix,
                    "industries": summary.industry_mix,
                    "segments": summary.segment_lift,
                }
            ),
        ),
    )


@router.get("/healthcards", response_class=HTMLResponse)
async def view_healthcards(
    request: Request,
    config: ConfigDep,
    service: ServiceDep,
    page: int = Query(1, ge=1),
    grade: str | None = None,
    risk_band: str | None = None,
    customer_segment: str | None = None,
    eligible: bool | None = None,
) -> HTMLResponse:
    filters = {
        k: v
        for k, v in {
            "grade": grade,
            "risk_band": risk_band,
            "customer_segment": customer_segment,
            "eligible": eligible,
        }.items()
        # An untouched <select> submits "", which must not become an equality filter.
        if v is not None and v != ""
    }
    result = await service.list_cards(
        page=page, page_size=int(config["api"]["page_size"]), filters=filters
    )
    return templates.TemplateResponse(
        request,
        "view_healthcards.html",
        _context(
            request,
            config,
            result=result,
            filters=filters,
            grades=GRADES,
            bands=BANDS,
            segments=SEGMENTS,
        ),
    )


@router.get("/healthcard/{msme_id}", response_class=HTMLResponse)
async def view_healthcard_detail(
    request: Request, msme_id: str, config: ConfigDep, service: ServiceDep
) -> HTMLResponse:
    try:
        card = await service.get_card(msme_id)
    except MSMENotFoundError:
        return templates.TemplateResponse(
            request,
            "view_not_found.html",
            _context(request, config, msme_id=msme_id),
            status_code=404,
        )
    return templates.TemplateResponse(
        request,
        "view_healthcard_detail.html",
        _context(
            request,
            config,
            card=card,
            chart_data=_chart_payload(
                {
                    "pillars": card.pillars,
                    "shap": card.explanation.shap_contributions if card.explanation else [],
                }
            ),
        ),
    )


@router.get("/search", response_class=HTMLResponse)
async def view_search(  # noqa: PLR0913
    request: Request,
    config: ConfigDep,
    service: ServiceDep,
    msme_id: str | None = None,
    customer_segment: str | None = None,
    industry: str | None = None,
    location_category: str | None = None,
    grade: str | None = None,
    risk_band: str | None = None,
    page: int = Query(1, ge=1),
) -> Response:
    # A direct id lookup is the common case; jump straight to the card.
    if msme_id:
        return RedirectResponse(url=f"/healthcard/{msme_id.strip()}", status_code=303)
    filters = {
        k: v
        for k, v in {
            "customer_segment": customer_segment,
            "industry": industry,
            "location_category": location_category,
            "grade": grade,
            "risk_band": risk_band,
        }.items()
        if v
    }
    result = await service.list_cards(
        page=page, page_size=int(config["api"]["page_size"]), filters=filters
    )
    frame = await service.store.portfolio_frame()
    industries = _distinct(frame, "Industry")
    locations = _distinct(frame, "Location_Category")
    return templates.TemplateResponse(
        request,
        "view_search.html",
        _context(
            request,
            config,
            result=result,
            filters=filters,
            segments=SEGMENTS,
            grades=GRADES,
            bands=BANDS,
            industries=industries,
            locations=locations,
        ),
    )


@router.get("/anomalies", response_class=HTMLResponse)
async def view_anomalies(
    request: Request,
    config: ConfigDep,
    portfolio: PortfolioDep,
    limit: int = Query(100, ge=1, le=500),
) -> HTMLResponse:
    frame = await portfolio.anomaly_queue(limit)
    rows = frame.to_dict(orient="records") if not frame.empty else []
    summary = await portfolio.summary()
    return templates.TemplateResponse(
        request,
        "view_anomalies.html",
        _context(
            request,
            config,
            rows=rows,
            flag_mix=summary.flag_mix,
            id_column=config["schema"]["id_column"],
        ),
    )


@router.get("/simulator", response_class=HTMLResponse)
async def view_simulator(
    request: Request,
    config: ConfigDep,
    service: ServiceDep,
    msme_id: str | None = None,
) -> HTMLResponse:
    card = None
    profile: dict[str, Any] | None = None
    error: str | None = None
    if msme_id:
        try:
            card = await service.get_card(msme_id.strip())
            profile = await service.store.get_profile(msme_id.strip())
        except MSMENotFoundError as exc:
            error = str(exc)
    # The levers a firm can realistically move, with their current values.
    levers = [
        ("GST_Filing_Timeliness_Pct", "GST filing timeliness (%)", 0, 100, 0.1),
        ("Vendor_Payment_Timeliness_Pct", "Supplier payment punctuality (%)", 0, 100, 0.1),
        ("Avg_Invoice_Payment_Delay_Days", "Average invoice delay (days)", 0, 120, 0.5),
        ("Overdraft_Usage_Ratio", "Overdraft utilisation (0-1)", 0, 1, 0.01),
        ("Salary_Consistency_Pct", "Payroll consistency (%)", 0, 100, 0.1),
        ("Customer_Concentration_Ratio", "Customer concentration (0-1)", 0, 1, 0.01),
        ("Revenue_Growth_Rate_Pct", "Revenue growth (%)", -60, 120, 0.5),
        ("Transaction_Volatility_Index", "Transaction volatility (0-1)", 0, 1, 0.01),
    ]
    return templates.TemplateResponse(
        request,
        "view_simulator.html",
        _context(
            request,
            config,
            card=card,
            profile=profile,
            levers=levers,
            msme_id=msme_id,
            error=error,
        ),
    )


@router.get("/assess", response_class=HTMLResponse)
async def view_assess(
    request: Request,
    config: ConfigDep,
    service: ServiceDep,
    msme_id: str | None = None,
) -> HTMLResponse:
    """The underwriting desk: type a firm's evidence in, get the decision back.

    The page renders the form and nothing else; the verdict comes from a browser
    POST to ``/api/v1/score``, so what an analyst reads here is byte-for-byte what
    an integrator gets from the API rather than a second rendering of it.
    """
    sections = assess_form_sections()
    prefill: dict[str, str] = {}
    error: str | None = None
    if msme_id:
        profile = await service.store.get_profile(msme_id.strip())
        if profile is None:
            error = f"No stored profile for {msme_id.strip()}."
        else:
            prefill = {
                field.name: _form_value(profile.get(field.name))
                for _, fields in sections
                for field in fields
            }
            prefill[config["schema"]["id_column"]] = msme_id.strip()
    return templates.TemplateResponse(
        request,
        "view_assess.html",
        _context(
            request,
            config,
            sections=sections,
            prefill=prefill,
            msme_id=msme_id,
            error=error,
            industries=_distinct(await service.store.portfolio_frame(), "Industry"),
        ),
    )


@router.get("/metrics", response_class=HTMLResponse)
async def view_metrics(request: Request, config: ConfigDep) -> HTMLResponse:
    report = load_training_report(config)
    labels = {k: v["label"] for k, v in config["pillars"].items() if k != "fhs_weights"}
    thresholds = config["evaluation"]["thresholds"]
    return templates.TemplateResponse(
        request,
        "view_metrics.html",
        _context(
            request,
            config,
            report=report,
            thresholds=thresholds,
            pillar_labels=labels,
            chart_data=_chart_payload(
                {
                    "reliability": (report or {}).get("reliability_curve", []),
                    "pillars": (report or {}).get("pillar_metrics", {}),
                    "labels": labels,
                    "pillar_adj_r2_min": thresholds["pillar_adj_r2_min"],
                }
            ),
        ),
    )
