#!/usr/bin/env bash
# Lint, format, type-check, secret-scan and Dockerfile-lint everything.
set -ex
uv run pre-commit run --all-files "$@"
