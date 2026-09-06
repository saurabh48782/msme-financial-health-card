#!/usr/bin/env bash
# Unit tests. The identical command runs locally, in the container target and in CI,
# so the three cannot diverge.
set -ex
uv run pytest -vv tests/unit "$@"
