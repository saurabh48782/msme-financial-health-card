#!/usr/bin/env bash
# Integration tests.
#
# Default is CI-safe: the suite is stub-wired, so it needs no database, no MLflow
# and no trained model. `--full` additionally runs the tests marked
# `aws_credentials`, which do need real infrastructure.
set -ex

MARKS=(-m "not aws_credentials")
if [[ "${1:-}" == "--full" ]]; then
  MARKS=()
  shift
else
  export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-dummy}"
  export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-dummy}"  # pragma: allowlist secret
  export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-ap-south-1}"
fi

uv run pytest -vv tests/integration "${MARKS[@]}" "$@"
