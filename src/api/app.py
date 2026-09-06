"""The FastAPI application factory.

``create_app(bootstrap=None)`` is the whole testability story. In production the
lifespan validates config, loads the store and the model and warms SHAP. In a
test, ``bootstrap`` replaces that entirely with a callable that populates
``app.state`` with stubs — so an integration suite needs no MLflow and no
trained model, and there is nothing to monkeypatch.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routers import (
    credit,
    explain,
    healthcard,
    portfolio,
    risk_flags,
    score,
    simulate,
    views,
)
from src.api.security import (
    ApiKeyMiddleware,
    RequestContextMiddleware,
    cors_origins,
    describe,
)
from src.api.templates import mount_static
from src.utils.config import load_config, validate_config
from src.utils.custom_error import CustomError
from src.utils.logger import get_logger
from src.utils.pii import redact_text

logger = get_logger(__name__)

Bootstrap = Callable[[FastAPI], Any]


async def _default_bootstrap(app: FastAPI) -> None:
    """Production wiring: config -> store -> model -> services."""
    from src.data_access.csv_store import CsvCardStore
    from src.model.utilities import ModelLoader
    from src.service.healthcard_service import HealthCardService
    from src.service.portfolio_service import PortfolioService
    from src.service.simulator import Simulator

    config = validate_config(load_config())
    app.state.config = config
    app.state.store = CsvCardStore(config)

    bundle = ModelLoader(config).load()
    app.state.bundle = bundle
    app.state.healthcard_service = HealthCardService(app.state.store, bundle, config)
    app.state.portfolio_service = PortfolioService(app.state.store, config)
    app.state.simulator = Simulator(bundle, config)

    # Warm SHAP: the first explainer call compiles numba kernels and would
    # otherwise land on a real user as a multi-second request.
    if bundle.shap_explainer is not None and bundle.pd_model is not None:
        try:
            import pandas as pd

            from src.model.components import build_feature_matrix

            warm = pd.DataFrame([{c: 0.0 for c in bundle.pd_model.features}])
            matrix, _, _ = build_feature_matrix(warm, config, features=bundle.pd_model.features)
            bundle.shap_explainer.explain_row(matrix)
            logger.info("SHAP explainer warmed")
        except Exception as error:  # noqa: BLE001 - warming is an optimisation
            logger.warning("SHAP warm-up skipped", error=str(error))


def _lifespan(bootstrap: Bootstrap | None):  # type: ignore[no-untyped-def]
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info("--- Starting API ---")
        try:
            result = (bootstrap or _default_bootstrap)(app)
            if hasattr(result, "__await__"):
                await result
            logger.info(
                "API ready",
                store=type(getattr(app.state, "store", None)).__name__,
                model_version=getattr(getattr(app.state, "bundle", None), "model_version", None),
                **describe(getattr(app.state, "config", None) or load_config()),
            )
            yield
        finally:
            logger.info("--- API Finished ---")

    return lifespan


def create_app(bootstrap: Bootstrap | None = None) -> FastAPI:
    """Build the application. Pass ``bootstrap`` to replace the production wiring."""
    config = load_config()
    api = config["api"]

    app = FastAPI(
        title=str(api["title"]),
        version=str(api["version"]),
        root_path=str(api.get("root_path", "")),
        lifespan=_lifespan(bootstrap),
        description=(
            "Explainable, alternative-data credit scoring for New-to-Credit and "
            "New-to-Bank Indian MSMEs. Consumes consent-based GST, UPI, bank/AA, "
            "EPFO and invoice signals; returns a 0-100 Financial Health Score, six "
            "pillar sub-scores, typed risk flags, SHAP-backed reason codes and a "
            "credit recommendation."
        ),
    )

    # Added last runs first. This order is asserted by a unit test because it is
    # load-bearing: the request id has to be bound before a handler logs anything,
    # so it must sit inside the auth check rather than outside it.
    origins = cors_origins(str(api.get("cors_allow_origins", "")))
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(ApiKeyMiddleware, api_key=str(api.get("api_key", "")))

    for router in (
        score.router,
        healthcard.router,
        credit.router,
        explain.router,
        risk_flags.router,
        portfolio.router,
        simulate.router,
        views.router,
    ):
        app.include_router(router)

    mount_static(app)

    @app.exception_handler(CustomError)
    async def _custom_error_handler(request: Request, exc: CustomError) -> JSONResponse:
        # Internals never reach the client: the detail goes to the log, a generic
        # message goes over the wire. The detail is redacted first — a pipeline
        # exception message can quote the offending row, and this is the one path
        # where a raw borrower payload could otherwise land in a log line.
        logger.error(
            "Unhandled pipeline error", path=request.url.path, detail=redact_text(str(exc))
        )
        return JSONResponse(
            {"detail": "The request could not be processed"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    @app.get("/healthcheck", tags=["health"], summary="Liveness — touches no dependency")
    async def healthcheck() -> dict[str, str]:
        """Deliberately dependency-free.

        A liveness probe that checks the database restarts a healthy container
        every time the database hiccups, turning a brief outage into an outage plus
        a restart storm.
        """
        return {"status": "ok"}

    @app.get("/readiness", tags=["health"], summary="Dependency probe")
    async def readiness(
        request: Request, check_model: bool = True, check_store: bool = True
    ) -> JSONResponse:
        checks: dict[str, Any] = {}
        healthy = True

        if check_model:
            bundle = getattr(request.app.state, "bundle", None)
            checks["model"] = {
                "loaded": bundle is not None,
                "has_ml": bool(bundle is not None and bundle.has_ml),
                "version": getattr(bundle, "model_version", None),
            }
            healthy &= bundle is not None

        if check_store:
            store = getattr(request.app.state, "store", None)
            store_ok = bool(store is not None and await store.health())
            checks["store"] = {"backend": type(store).__name__, "healthy": store_ok}
            healthy &= store_ok

        return JSONResponse(
            {"status": "ready" if healthy else "not ready", "checks": checks},
            status_code=(status.HTTP_200_OK if healthy else status.HTTP_503_SERVICE_UNAVAILABLE),
        )

    return app


app = create_app()
