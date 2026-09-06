"""Assertions on rendered HTML.

View tests assert on what a user actually sees. A card page that returns 200 while
silently dropping the SHAP drivers is a broken page, and only a body assertion
catches that.
"""

from __future__ import annotations

import re


def assert_status(response, expected: int) -> None:  # type: ignore[no-untyped-def]
    assert response.status_code == expected, (
        f"expected {expected}, got {response.status_code}: {response.text[:400]}"
    )


def body(response) -> str:  # type: ignore[no-untyped-def]
    return response.text


def assert_contains(html: str, *needles: str) -> None:
    missing = [n for n in needles if n not in html]
    assert not missing, f"page is missing: {missing}"


def assert_pillar_radar_shown(html: str) -> None:
    assert 'id="pillarRadar"' in html
    assert "Six pillars" in html


def assert_reason_codes_shown(html: str) -> None:
    assert "What is working" in html
    assert "What is holding this firm back" in html


def assert_provenance_shown(html: str) -> None:
    """Every card must carry the versions that produced it."""
    assert_contains(html, "Provenance", "Policy version", "Rubric version", "Feature snapshot")


def assert_no_server_error(html: str) -> None:
    assert "Internal Server Error" not in html
    assert "Traceback" not in html


def chart_payload(html: str, element_id: str) -> str:
    match = re.search(
        rf'<script id="{element_id}" type="application/json">(.*?)</script>', html, re.S
    )
    assert match, f"no chart payload #{element_id} in the page"
    return match.group(1)


def total_firms(html: str) -> int:
    """The listing header count, e.g. ``50,000 firms``."""
    match = re.search(r"([\d,]+) firms", html)
    assert match, "no firm count in the page"
    return int(match.group(1).replace(",", ""))


def submitted_form(html: str) -> dict[str, object]:
    """The rendered assessment form as the JSON record a browser would post.

    ``data-kind`` is read off each control exactly as ``assess.js`` reads it, so
    the payload this builds is the payload the page sends — an assertion about the
    real form rather than about a dict written next to it.
    """
    record: dict[str, object] = {}
    for attrs, options in re.findall(r"<select([^>]*)>(.*?)</select>", html, re.S):
        name = re.search(r'name="([^"]+)"', attrs)
        chosen = re.search(r'value="([^"]*)"\s+selected', options)
        if name and chosen:
            record[name.group(1)] = chosen.group(1)
    for attrs in re.findall(r"<input([^>]*)>", html):
        name = re.search(r'name="([^"]+)"', attrs)
        kind = re.search(r'data-kind="([^"]+)"', attrs)
        value = re.search(r'value="([^"]*)"', attrs)
        if not (name and kind and value):
            continue
        raw = value.group(1)
        if raw == "":
            record[name.group(1)] = None
        else:
            record[name.group(1)] = float(raw) if kind.group(1) == "number" else raw
    return record
