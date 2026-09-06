"""API-key auth, request context and CORS.

Every control is env-driven and every default is safe, so local development needs
no configuration and a deployment cannot accidentally ship wide open without
someone having set a variable to make it so.

One detail that is easy to get subtly wrong and is therefore deliberate here:
**API keys are compared with** ``secrets.compare_digest``, because a ``==`` on a
secret leaks its length and prefix through timing.

Rate limiting and a pre-buffering body-size cap belong in front of a real
deployment. They are deliberately not in this repository — at this scale they
would be an unmetered edge concern implemented in the wrong tier; see
``docs/BEYOND_SCOPE.md``.
"""

from __future__ import annotations

import secrets
import time
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from src.utils.logger import get_logger

logger = get_logger(__name__)

API_KEY_HEADER = "X-API-Key"
REQUEST_ID_HEADER = "X-Request-ID"
PROTECTED_PREFIXES = ("/api/",)
# Probes and the bundled dashboard stay open: a liveness check that needs a
# credential is a liveness check that fails during a credential rotation.
OPEN_PATHS = ("/healthcheck", "/readiness", "/static", "/docs", "/redoc", "/openapi.json")


def _is_protected(path: str) -> bool:
    return path.startswith(PROTECTED_PREFIXES) and not path.startswith(OPEN_PATHS)


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Require ``X-API-Key`` on ``/api/*`` when a key is configured."""

    def __init__(self, app: ASGIApp, api_key: str | None) -> None:
        super().__init__(app)
        self.api_key = (api_key or "").strip()
        if not self.api_key:
            logger.warning("API authentication is DISABLED — set HEALTHCARD_API_KEY to enable it")

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if self.api_key and _is_protected(request.url.path):
            presented = request.headers.get(API_KEY_HEADER, "")
            if not secrets.compare_digest(presented, self.api_key):
                logger.warning(
                    "Rejected unauthenticated request",
                    path=request.url.path,
                    has_header=bool(presented),
                )
                return JSONResponse(
                    {"detail": f"Missing or invalid {API_KEY_HEADER}"}, status_code=401
                )
        return await call_next(request)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Bind a request id into the structlog context and echo it back.

    Every log line emitted while serving a request then carries the same id, which
    is the difference between "a card scored slowly" and "*this* card scored
    slowly, and here is every stage timing for it".
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        import uuid

        import structlog

        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:16]
        structlog.contextvars.bind_contextvars(request_id=request_id, path=request.url.path)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.unbind_contextvars("request_id", "path")
        response.headers[REQUEST_ID_HEADER] = request_id
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        response.headers["X-Response-Time-ms"] = str(elapsed_ms)
        return response


def cors_origins(raw: str | None) -> list[str]:
    """Parse ``CORS_ALLOW_ORIGINS``. Empty means no CORS at all, which is the default.

    The bundled dashboard is same-origin, so there is no reason for a browser from
    another origin to call this API unless someone deliberately says so.
    """
    if not raw:
        return []
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def middleware_order() -> list[str]:
    """The asserted order, outermost first. Tested, because order is load-bearing."""
    return ["ApiKeyMiddleware", "RequestContextMiddleware", "CORSMiddleware"]


def describe(config: dict[str, Any]) -> dict[str, Any]:
    """What is switched on, for the startup log line."""
    api = config["api"]
    return {
        "api_key_required": bool(str(api.get("api_key", ""))),
        "cors_origins": cors_origins(str(api.get("cors_allow_origins", ""))),
    }
