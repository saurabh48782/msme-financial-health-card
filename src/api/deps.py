"""Dependency accessors.

Everything a router needs is read off ``app.state`` through these callables, never
imported as a module global. That is what lets a test swap in a stub by populating
``app.state`` in a bootstrap callable — no monkeypatching, no global ``MODEL``
dict, no import-order surprises.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, HTTPException, Request, status

from src.scoring.scoring_orchestration import ScoringBundle
from src.service.healthcard_service import HealthCardService
from src.service.portfolio_service import PortfolioService
from src.service.simulator import Simulator


def _require(request: Request, name: str) -> Any:
    value = getattr(request.app.state, name, None)
    if value is None:
        # 503, not 500: the dependency is missing, the request was not malformed,
        # and a load balancer should route elsewhere rather than retry here.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{name} is not available",
        )
    return value


def get_config(request: Request) -> dict[str, Any]:
    return _require(request, "config")  # type: ignore[no-any-return]


def get_bundle(request: Request) -> ScoringBundle:
    return _require(request, "bundle")  # type: ignore[no-any-return]


def get_service(request: Request) -> HealthCardService:
    return _require(request, "healthcard_service")  # type: ignore[no-any-return]


def get_portfolio_service(request: Request) -> PortfolioService:
    return _require(request, "portfolio_service")  # type: ignore[no-any-return]


def get_simulator(request: Request) -> Simulator:
    return _require(request, "simulator")  # type: ignore[no-any-return]


ConfigDep = Annotated[dict[str, Any], Depends(get_config)]
BundleDep = Annotated[ScoringBundle, Depends(get_bundle)]
ServiceDep = Annotated[HealthCardService, Depends(get_service)]
PortfolioDep = Annotated[PortfolioService, Depends(get_portfolio_service)]
SimulatorDep = Annotated[Simulator, Depends(get_simulator)]
