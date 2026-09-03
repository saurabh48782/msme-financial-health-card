"""structlog configuration shared by pipelines, the API and the CLIs.

Two conventions the whole codebase relies on:

* **Static event string + keyword fields** — ``logger.info("Card scored",
  msme_id=..., fhs=...)`` rather than an f-string. Every field is then queryable
  in ``jq`` and CloudWatch Insights instead of being trapped in prose.
* **Console renderer on a tty, JSON everywhere else** — humans get colour, log
  shippers get one object per line. ``LOG_TARGET=stdout`` suppresses the file
  handler entirely, which is what containers want (stdout -> CloudWatch).
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
_NOISY_LOGGERS = (
    "asyncio",
    "botocore",
    "urllib3",
    "s3transfer",
    "matplotlib",
    "mlflow",
    "git",
    "numba",
    "shap",
    "py4j",
)
_configured = False

# Processors that run for both structlog-native and stdlib records.
_SHARED_PROCESSORS: list[Any] = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_log_level,
    structlog.stdlib.add_logger_name,
    structlog.processors.TimeStamper(fmt="iso", utc=True),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.UnicodeDecoder(),
]


def _use_console() -> bool:
    """Colourised console output only for an interactive terminal."""
    override = os.getenv("LOG_FORMAT", "").lower()
    if override in {"json", "console"}:
        return override == "console"
    return sys.stderr.isatty()


def setup_logging(level: str | None = None, *, force: bool = False) -> None:
    """Configure structlog and the stdlib root logger.

    Idempotent, except that an explicit ``level`` always reconfigures.
    """
    global _configured
    if _configured and not force and level is None:
        return

    requested = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    log_level = getattr(logging, requested, logging.INFO)
    console = _use_console()

    renderer: Any = (
        structlog.dev.ConsoleRenderer(colors=True)
        if console
        else structlog.processors.JSONRenderer()
    )

    structlog.configure(
        processors=[
            *_SHARED_PROCESSORS,
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # One formatter bridges stdlib records (uvicorn, xgboost, mlflow) into the
    # same processor chain, so third-party logs are shaped like ours.
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=_SHARED_PROCESSORS,
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
    )

    handlers: list[logging.Handler] = []
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    handlers.append(stream)

    file_error: OSError | None = None
    if os.getenv("LOG_TARGET", "file").lower() != "stdout":
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(UTC).strftime("%Y-%m-%d")
            file_handler = logging.FileHandler(LOG_DIR / f"healthcard-{stamp}.log")
            # A log file is shipped, never read by eye — always JSON.
            file_handler.setFormatter(
                structlog.stdlib.ProcessorFormatter(
                    foreign_pre_chain=_SHARED_PROCESSORS,
                    processors=[
                        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                        structlog.processors.JSONRenderer(),
                    ],
                )
            )
            handlers.append(file_handler)
        except OSError as exc:
            file_error = exc

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(log_level)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    _configured = True

    if file_error is not None:
        structlog.stdlib.get_logger(__name__).warning(
            "File logging disabled, falling back to stderr only",
            log_dir=str(LOG_DIR),
            error=str(file_error),
        )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound logger, configuring logging on first use."""
    setup_logging()
    return structlog.stdlib.get_logger(name)  # type: ignore[no-any-return]
