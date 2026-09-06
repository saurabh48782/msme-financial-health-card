"""Security middleware: API key, request context, CORS.

These are asserted with a bare Starlette app rather than the real one so a
failure points at the middleware rather than at whatever the app happened to be
doing.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.security import (
    API_KEY_HEADER,
    REQUEST_ID_HEADER,
    ApiKeyMiddleware,
    RequestContextMiddleware,
    cors_origins,
    middleware_order,
)


def build(**middleware: object) -> FastAPI:
    app = FastAPI()

    @app.get("/healthcheck")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/thing")
    async def thing() -> dict[str, bool]:
        return {"ok": True}

    @app.post("/api/v1/thing")
    async def create(payload: dict) -> dict[str, int]:
        return {"size": len(payload)}

    if "api_key" in middleware:
        app.add_middleware(ApiKeyMiddleware, api_key=middleware["api_key"])
    if middleware.get("context"):
        app.add_middleware(RequestContextMiddleware)
    return app


class TestApiKey:
    @pytest.mark.parametrize(
        ("configured", "sent", "path", "status"),
        [
            pytest.param("", None, "/api/v1/thing", 200, id="no-key-configured-is-open"),
            pytest.param("secret", None, "/api/v1/thing", 401, id="missing-key"),
            pytest.param("secret", "nope", "/api/v1/thing", 401, id="wrong-key"),
            pytest.param("secret", "secret", "/api/v1/thing", 200, id="correct-key"),
            # A probe that needs a credential fails during a credential rotation.
            pytest.param("secret", None, "/healthcheck", 200, id="healthcheck-stays-open"),
        ],
    )
    def test_only_a_matching_key_reaches_a_protected_path(
        self, configured: str, sent: str | None, path: str, status: int
    ) -> None:
        headers = {} if sent is None else {API_KEY_HEADER: sent}
        with TestClient(build(api_key=configured)) as client:
            assert client.get(path, headers=headers).status_code == status


class TestRequestContext:
    @pytest.mark.parametrize(
        "supplied",
        [pytest.param(None, id="generated"), pytest.param("trace-123", id="preserved")],
    )
    def test_the_request_id_is_echoed(self, supplied: str | None) -> None:
        headers = {} if supplied is None else {REQUEST_ID_HEADER: supplied}
        with TestClient(build(context=True)) as client:
            response = client.get("/api/v1/thing", headers=headers)
            echoed = response.headers[REQUEST_ID_HEADER]
            if supplied is None:
                assert echoed
            else:
                assert echoed == supplied
            assert "X-Response-Time-ms" in response.headers


class TestCorsConfig:
    @pytest.mark.parametrize("raw", ["", None])
    def test_empty_means_no_cors(self, raw: str | None) -> None:
        """The bundled dashboard is same-origin, so the safe default is none at all."""
        assert cors_origins(raw) == []

    def test_comma_separated_origins_are_parsed(self) -> None:
        assert cors_origins("http://a.com, http://b.com") == ["http://a.com", "http://b.com"]


class TestMiddlewareOrder:
    def test_documented_order_matches_the_app(self) -> None:
        """Order is load-bearing: the request id must bind before a handler logs,
        so it has to sit inside the auth check rather than outside it."""
        from src.api.app import create_app

        async def bootstrap(app: FastAPI) -> None:
            app.state.config = __import__(
                "src.utils.config", fromlist=["load_config"]
            ).load_config()

        app = create_app(bootstrap)
        applied = [m.cls.__name__ for m in app.user_middleware]
        expected = [name for name in middleware_order() if name in applied]
        assert applied == expected
