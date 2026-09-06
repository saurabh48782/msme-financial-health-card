"""In-process API contract tests over httpx ASGITransport."""

from __future__ import annotations

import json
from typing import Any

import pytest
from httpx import AsyncClient

from src.schemas.msme import MSMEFeatures
from tests.integration.helpers import (
    assert_no_server_error,
    assert_pillar_radar_shown,
    assert_provenance_shown,
    assert_reason_codes_shown,
    assert_status,
    chart_payload,
    submitted_form,
    total_firms,
)
from tests.stubs import ELIGIBLE_ID, INELIGIBLE_ID


class TestHealthProbes:
    async def test_liveness_touches_no_dependency(self, client: AsyncClient) -> None:
        response = await client.get("/healthcheck")
        assert_status(response, 200)
        assert response.json() == {"status": "ok"}

    async def test_readiness_reports_each_check(self, client: AsyncClient) -> None:
        response = await client.get("/readiness")
        assert_status(response, 200)
        checks = response.json()["checks"]
        assert "model" in checks
        assert "store" in checks

    async def test_readiness_checks_can_be_disabled(self, client: AsyncClient) -> None:
        response = await client.get("/readiness?check_store=false&check_model=false")
        assert_status(response, 200)
        assert response.json()["checks"] == {}


class TestScoring:
    async def test_score_returns_a_complete_card(
        self, client: AsyncClient, profiles: dict[str, dict[str, Any]]
    ) -> None:
        payload = {"records": [profiles[ELIGIBLE_ID]], "include_explanation": True}
        response = await client.post("/api/v1/score", json=payload)
        assert_status(response, 200)
        card = response.json()[0]
        assert card["msme_id"] == ELIGIBLE_ID
        assert 0 <= card["financial_health_score"] <= 100
        assert len(card["pillars"]) == 6
        assert card["explanation"]["reason_codes"]
        assert card["policy_version"]
        assert card["feature_snapshot_hash"]

    async def test_pillar_contributions_sum_to_the_score(
        self, client: AsyncClient, profiles: dict[str, dict[str, Any]]
    ) -> None:
        """The exactness claim, asserted over the wire."""
        response = await client.post("/api/v1/score", json={"records": [profiles[ELIGIBLE_ID]]})
        card = response.json()[0]
        total = sum(p["contribution"] for p in card["pillars"])
        assert total == pytest.approx(card["financial_health_score"], abs=1e-2)
        for pillar in card["pillars"]:
            drivers = sum(d["contribution"] for d in pillar["drivers"])
            assert pillar["baseline"] + drivers == pytest.approx(pillar["score"], abs=1e-2)

    async def test_batch_scoring_returns_one_card_per_record(
        self, client: AsyncClient, profiles: dict[str, dict[str, Any]]
    ) -> None:
        payload = {"records": list(profiles.values()), "include_explanation": False}
        response = await client.post("/api/v1/score", json=payload)
        assert_status(response, 200)
        assert len(response.json()) == 2

    async def test_malformed_record_is_422(self, client: AsyncClient) -> None:
        response = await client.post("/api/v1/score", json={"records": [{"MSME_ID": "X"}]})
        assert_status(response, 422)

    async def test_empty_batch_is_422(self, client: AsyncClient) -> None:
        assert_status(await client.post("/api/v1/score", json={"records": []}), 422)

    async def test_oversized_batch_is_413(
        self, client: AsyncClient, profiles: dict[str, dict[str, Any]], config: dict[str, Any]
    ) -> None:
        limit = int(config["api"]["max_batch_size"])
        payload = {"records": [profiles[ELIGIBLE_ID]] * (limit + 1)}
        assert_status(await client.post("/api/v1/score", json=payload), 413)


class TestCardEndpoints:
    async def test_card_is_scored_on_demand(self, client: AsyncClient) -> None:
        response = await client.get(f"/api/v1/healthcard/{ELIGIBLE_ID}")
        assert_status(response, 200)
        assert response.json()["msme_id"] == ELIGIBLE_ID

    async def test_unknown_msme_is_404(self, client: AsyncClient) -> None:
        assert_status(await client.get("/api/v1/healthcard/MSME_NOPE"), 404)

    async def test_repeat_gets_agree(self, client: AsyncClient) -> None:
        """Scoring is deterministic: the same firm scores the same twice."""
        first = (await client.get(f"/api/v1/healthcard/{ELIGIBLE_ID}")).json()
        second = (await client.get(f"/api/v1/healthcard/{ELIGIBLE_ID}")).json()
        assert first["financial_health_score"] == second["financial_health_score"]
        assert first["feature_snapshot_hash"] == second["feature_snapshot_hash"]

    async def test_explain_returns_reason_codes(self, client: AsyncClient) -> None:
        response = await client.get(f"/api/v1/explain/{ELIGIBLE_ID}")
        assert_status(response, 200)
        assert response.json()["reason_codes"]

    async def test_pillar_endpoint_returns_six(self, client: AsyncClient) -> None:
        response = await client.get(f"/api/v1/explain/{ELIGIBLE_ID}/pillars")
        assert len(response.json()) == 6

    async def test_credit_recommendation_carries_a_policy_version(
        self, client: AsyncClient
    ) -> None:
        response = await client.get(f"/api/v1/credit-recommendation/{ELIGIBLE_ID}")
        assert_status(response, 200)
        assert response.json()["policy_version"]


class TestPolicyBranches:
    """Both branches, over HTTP. A suite that only ever tests approvals proves half
    the engine."""

    async def test_healthy_firm_is_eligible_with_a_limit(self, client: AsyncClient) -> None:
        card = (await client.get(f"/api/v1/healthcard/{ELIGIBLE_ID}")).json()
        assert card["credit"]["eligible"] is True
        assert card["credit"]["credit_limit_inr"] > 0
        assert card["credit"]["tenor_months"] > 0

    async def test_weak_firm_is_declined_with_reasons(self, client: AsyncClient) -> None:
        card = (await client.get(f"/api/v1/healthcard/{INELIGIBLE_ID}")).json()
        assert card["credit"]["eligible"] is False
        assert card["credit"]["credit_limit_inr"] == 0
        assert card["credit"]["decline_reasons"] or card["credit"]["notes"]


class TestSimulation:
    async def test_simulation_returns_before_and_after(self, client: AsyncClient) -> None:
        response = await client.post(
            f"/api/v1/simulate/{INELIGIBLE_ID}",
            json={"overrides": {"GST_Filing_Timeliness_Pct": 99.0}},
        )
        assert_status(response, 200)
        result = response.json()
        assert len(result["deltas"]) == 5
        assert len(result["before_pillars"]) == len(result["after_pillars"]) == 6

    async def test_unknown_override_is_422(self, client: AsyncClient) -> None:
        response = await client.post(
            f"/api/v1/simulate/{ELIGIBLE_ID}", json={"overrides": {"Nope": 1}}
        )
        assert_status(response, 422)

    async def test_unknown_msme_is_404(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/simulate/MSME_NOPE", json={"overrides": {"Seasonality_Index": 0.1}}
        )
        assert_status(response, 404)


class TestPortfolioAndRegistry:
    async def test_summary_covers_the_scored_book(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/portfolio/summary")
        assert_status(response, 200)
        assert response.json()["firms"] == 2

    async def test_registry_outage_returns_an_empty_list_not_500(self, client: AsyncClient) -> None:
        """The API scores from a local artifact; MLflow being down must not break it."""
        response = await client.get("/api/v1/models/versions")
        assert_status(response, 200)
        assert isinstance(response.json(), list)


class TestHtmlViews:
    async def test_every_page_renders(self, client: AsyncClient) -> None:
        for path in (
            "/",
            "/portfolio",
            "/healthcards",
            "/search",
            "/assess",
            "/anomalies",
            "/metrics",
            "/simulator",
        ):
            response = await client.get(path)
            assert_status(response, 200)
            assert_no_server_error(response.text)

    async def test_blank_filter_selects_do_not_filter(self, client: AsyncClient) -> None:
        """An untouched <select> submits "" — treating that as an equality filter
        matched nothing and emptied the listing."""
        unfiltered = (await client.get("/healthcards")).text
        blank = await client.get("/healthcards?grade=&risk_band=&customer_segment=")
        assert_status(blank, 200)
        assert total_firms(blank.text) == total_firms(unfiltered) == 2

    async def test_card_page_shows_the_full_card(self, client: AsyncClient) -> None:
        response = await client.get(f"/healthcard/{ELIGIBLE_ID}")
        assert_status(response, 200)
        html = response.text
        assert_pillar_radar_shown(html)
        assert_reason_codes_shown(html)
        assert_provenance_shown(html)

    async def test_card_chart_payload_is_valid_json(self, client: AsyncClient) -> None:
        """The charts read server-rendered JSON; malformed payloads break silently
        in the browser and would otherwise never fail a test."""
        html = (await client.get(f"/healthcard/{ELIGIBLE_ID}")).text
        payload = json.loads(chart_payload(html, "card-data"))
        assert len(payload["pillars"]) == 6

    async def test_portfolio_chart_payload_is_valid_json(self, client: AsyncClient) -> None:
        html = (await client.get("/portfolio")).text
        payload = json.loads(chart_payload(html, "portfolio-data"))
        assert set(payload) == {"histogram", "bands", "grades", "industries", "segments"}

    async def test_unknown_msme_page_is_404_not_500(self, client: AsyncClient) -> None:
        response = await client.get("/healthcard/MSME_NOPE")
        assert_status(response, 404)
        assert "No MSME with ID" in response.text

    async def test_search_by_id_redirects_to_the_card(self, client: AsyncClient) -> None:
        response = await client.get(f"/search?msme_id={ELIGIBLE_ID}")
        assert response.status_code == 303
        assert response.headers["location"] == f"/healthcard/{ELIGIBLE_ID}"

    async def test_assess_page_renders_every_contract_field(self, client: AsyncClient) -> None:
        html = (await client.get("/assess")).text
        for info in MSMEFeatures.model_fields.values():
            assert f'name="{info.alias}"' in html, info.alias

    async def test_assess_prefill_loads_a_stored_firm(self, client: AsyncClient) -> None:
        response = await client.get(f"/assess?msme_id={ELIGIBLE_ID}")
        assert_status(response, 200)
        assert f'value="{ELIGIBLE_ID}"' in response.text

    async def test_assess_prefill_of_an_unknown_firm_is_a_message_not_a_500(
        self, client: AsyncClient
    ) -> None:
        response = await client.get("/assess?msme_id=MSME_NOPE")
        assert_status(response, 200)
        assert "No stored profile" in response.text
        assert_no_server_error(response.text)

    async def test_the_prefilled_form_scores_through_the_public_api(
        self, client: AsyncClient
    ) -> None:
        """The page's whole contract: what the form renders, /api/v1/score accepts.

        The form is built from ``MSMEFeatures``, so a rendered value the API then
        rejects means the derivation is wrong — which no amount of testing the two
        sides separately would catch.
        """
        html = (await client.get(f"/assess?msme_id={ELIGIBLE_ID}")).text
        response = await client.post(
            "/api/v1/score", json={"records": [submitted_form(html)], "include_explanation": True}
        )
        assert_status(response, 200)
        card = response.json()[0]
        assert card["msme_id"] == ELIGIBLE_ID
        assert card["credit"]["eligible"] is True
        assert len(card["pillars"]) == 6

    async def test_simulator_page_loads_a_firm(self, client: AsyncClient) -> None:
        response = await client.get(f"/simulator?msme_id={ELIGIBLE_ID}")
        assert_status(response, 200)
        assert "Levers" in response.text
        assert "simulate-form" in response.text
