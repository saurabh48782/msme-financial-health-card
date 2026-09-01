"""Per-stage timing for the scoring pipeline.

Values are never logged raw — they go through :func:`src.utils.pii.summarise`, so
a frame becomes its shape and a payload becomes its key names.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, ParamSpec, TypeVar

from src.utils.logger import get_logger
from src.utils.pii import summarise

P = ParamSpec("P")
R = TypeVar("R")

logger = get_logger(__name__)


@contextmanager
def trace_stage(stage: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Time a block of work as a named stage.

    Yields a mutable dict; anything put in it is emitted with ``stage.done``,
    which is how a stage reports what it actually produced.
    """
    context = {key: summarise(value, key=key) for key, value in fields.items()}
    logger.debug("stage.start", stage=stage, **context)
    started = time.perf_counter()
    result: dict[str, Any] = {}
    try:
        yield result
    except Exception as error:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.error(
            "stage.error",
            stage=stage,
            ms=elapsed_ms,
            error=type(error).__name__,
            detail=str(error)[:256],
            **context,
            exc_info=True,
        )
        raise
    else:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "stage.done",
            stage=stage,
            ms=elapsed_ms,
            **context,
            **{key: summarise(value, key=key) for key, value in result.items()},
        )


def traced_stage(stage: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator form of :func:`trace_stage`."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with trace_stage(stage):
                return func(*args, **kwargs)

        return wrapper

    return decorator
