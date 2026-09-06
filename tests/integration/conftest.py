"""Integration fixtures: the real app over in-memory frames.

``create_app(bootstrap=...)`` replaces the production lifespan entirely, so this
suite needs no MLflow, no trained model and no network. That is the whole reason
the app is built by a factory that takes a bootstrap callable. Requests go through
httpx's ``ASGITransport`` in-process: one HTTP layer, asserted in Python, rather
than a second YAML-driven suite booting live servers to make the same assertions.

The store is the *production* ``CsvCardStore``, handed its two frames by
constructor injection rather than reading them off disk. So the suite exercises
the real listing, filtering and pagination code instead of a stub that could
quietly diverge from it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any

import pandas as pd
import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.api.app import create_app
from src.data_access.csv_store import CsvCardStore
from src.scoring.scoring_orchestration import score_portfolio
from src.service.healthcard_service import HealthCardService
from src.service.portfolio_service import PortfolioService
from src.service.simulator import Simulator
from src.utils.config import load_config
from tests.stubs import ELIGIBLE_ID, INELIGIBLE_ID, rubric_only_bundle


def _profiles(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Two real firms renamed to fixed IDs, one healthy and one weak.

    Using real rows keeps the scoring path honest — a hand-built dict would drift
    from the dataset's actual shape. Sorting by health score picks the extremes.
    """
    columns = [config["schema"]["id_column"], *config["schema"]["feature_columns"]]
    ranked = frame.sort_values("Financial_Health_Score", ascending=False)
    best = ranked.iloc[0][columns].to_dict()
    worst = ranked.iloc[-1][columns].to_dict()
    best[config["schema"]["id_column"]] = ELIGIBLE_ID
    worst[config["schema"]["id_column"]] = INELIGIBLE_ID
    clean = lambda row: {k: (None if pd.isna(v) else v) for k, v in row.items()}  # noqa: E731
    return {ELIGIBLE_ID: clean(best), INELIGIBLE_ID: clean(worst)}


def build_stub_app(
    profiles: dict[str, dict[str, Any]], config: dict[str, Any] | None = None
) -> FastAPI:
    """An app wired to in-memory frames.

    The portfolio frame is produced by the real ``score_portfolio`` — the same
    function that writes ``portfolio_scores.csv`` in production — so the
    portfolio views are tested against the column shape they actually receive.
    """
    cfg = config if config is not None else load_config()
    profile_frame = pd.DataFrame(list(profiles.values()))
    scores = score_portfolio(profile_frame, rubric_only_bundle(), cfg)

    async def bootstrap(app: FastAPI) -> None:
        bundle = rubric_only_bundle()
        store = CsvCardStore(cfg, scores=scores.copy(), profiles=profile_frame.copy())
        app.state.config = cfg
        app.state.store = store
        app.state.bundle = bundle
        app.state.healthcard_service = HealthCardService(store, bundle, cfg)
        app.state.portfolio_service = PortfolioService(store, cfg)
        app.state.simulator = Simulator(bundle, cfg)

    return create_app(bootstrap)


@pytest.fixture(scope="session")
def profiles(featured_frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return _profiles(featured_frame, config)


@pytest.fixture(scope="session")
def stub_app_factory(
    profiles: dict[str, dict[str, Any]], config: dict[str, Any]
) -> Callable[[], FastAPI]:
    return lambda: build_stub_app(profiles, config)


@pytest_asyncio.fixture
async def client(stub_app_factory: Callable[[], FastAPI]) -> AsyncIterator[AsyncClient]:
    """In-process HTTP against the stub-wired app, lifespan included."""
    app = stub_app_factory()
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as http:
            yield http
