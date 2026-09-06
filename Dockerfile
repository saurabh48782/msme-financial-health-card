# syntax=docker/dockerfile:1.7
#
# One target per purpose. Dependencies are installed before the source is copied,
# and the dependency install is split from the project install behind a BuildKit
# cache mount, so editing a source file rebuilds in seconds instead of re-resolving
# the whole environment.

# ---------------------------------------------------------------------------
FROM python:3.12-slim AS python_base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    LOG_TARGET=stdout \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /usr/local/bin/uv

RUN useradd --create-home --uid 10001 appuser
WORKDIR /app
RUN uv venv /opt/venv
COPY uv.lock pyproject.toml README.md ./

# ---------------------------------------------------------------------------
FROM python_base AS api_base
# Step one: dependencies only. This layer is reused until the lockfile changes.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project
COPY . /app/
# Step two: the project itself, which is cheap.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev \
    && mkdir -p /app/logs /app/model_artifact \
    && chown -R appuser:appuser /app /opt/venv

# ---------------------------------------------------------------------------
FROM api_base AS api
USER 10001
EXPOSE 8000
# Liveness only - a readiness check here would restart the container every time a
# dependency hiccups, turning a brief outage into a restart storm.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD ["/bin/sh", "-c", "curl -fsS http://localhost:8000/healthcheck || exit 1"]
CMD ["bash", "scripts/start_api.sh"]

# ---------------------------------------------------------------------------
FROM python_base AS api_base_test
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project
COPY . /app/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen \
    && git init -q /app \
    && uv run pre-commit install-hooks \
    && mkdir -p /app/logs \
    && chown -R appuser:appuser /app /opt/venv

FROM api_base_test AS unit_test
CMD ["bash", "scripts/unit_test.sh"]

FROM api_base_test AS integration_test
CMD ["bash", "scripts/integration_test.sh"]

FROM api_base_test AS precommit_check
CMD ["bash", "scripts/precommit_check.sh"]

# ---------------------------------------------------------------------------
FROM api_base AS trainer
# Training writes artifacts and talks to MLflow; it is not the serving image.
CMD ["bash", "scripts/train.sh"]
