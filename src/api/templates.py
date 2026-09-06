"""Jinja2 environment and the static mount.

Server-rendered, no SPA, no build step: the dashboard is HTML that a browser can
read with JavaScript disabled, plus a little vanilla ES module for the charts and
the simulator. Nothing here needs npm.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.utils.config import ROOT_DIR

TEMPLATE_DIR = ROOT_DIR / "src" / "templates"
STATIC_DIR = ROOT_DIR / "src" / "api" / "static"

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


def rupees(value: Any) -> str:
    """Format a number the way Indian finance actually writes it: lakh and crore."""
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return "—"
    if amount == 0:
        return "₹0"
    sign = "-" if amount < 0 else ""
    amount = abs(amount)
    if amount >= 1e7:
        return f"{sign}₹{amount / 1e7:,.2f} cr"
    if amount >= 1e5:
        return f"{sign}₹{amount / 1e5:,.2f} L"
    return f"{sign}₹{amount:,.0f}"


def percent(value: Any, places: int = 1) -> str:
    try:
        return f"{float(value) * 100:.{places}f}%"
    except (TypeError, ValueError):
        return "—"


def number(value: Any, places: int = 1) -> str:
    try:
        return f"{float(value):,.{places}f}"
    except (TypeError, ValueError):
        return "—"


def grade_colour(grade: str) -> str:
    """Bootstrap contextual class per grade, used by badges and progress bars."""
    return {
        "A+": "success",
        "A": "success",
        "B": "primary",
        "C": "warning",
        "D": "danger",
    }.get(str(grade), "secondary")


def band_colour(band: str) -> str:
    return {"Low": "success", "Medium": "warning", "High": "danger"}.get(str(band), "secondary")


def severity_colour(severity: str) -> str:
    return {
        "critical": "danger",
        "high": "danger",
        "medium": "warning",
        "low": "info",
    }.get(str(severity).lower(), "secondary")


def asset(request: Any, path: str) -> str:
    """``url_for('static', ...)`` plus the file's mtime, so a browser cannot serve a
    stale CSS/JS after a redeploy."""
    url = request.url_for("static", path=path)
    file = STATIC_DIR / path.lstrip("/")
    stamp = int(file.stat().st_mtime) if file.is_file() else 0
    return f"{url}?v={stamp}"


templates.env.globals["asset"] = asset
templates.env.filters["rupees"] = rupees
templates.env.filters["percent"] = percent
templates.env.filters["number"] = number
templates.env.filters["grade_colour"] = grade_colour
templates.env.filters["band_colour"] = band_colour
templates.env.filters["severity_colour"] = severity_colour


def mount_static(app: FastAPI) -> None:
    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
