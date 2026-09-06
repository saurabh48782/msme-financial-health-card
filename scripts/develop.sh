#!/usr/bin/env bash
# Local development: uvicorn with reload against the batch-scored CSV store.
#
# No database and no compose stack to bring up. Run the data and model pipelines
# once (bash scripts/train.sh) and every one of the 50,000 firms is drillable.
set -ex

uv run uvicorn src.api.app:app --host 0.0.0.0 --port "${PORT:-8000}" --reload
