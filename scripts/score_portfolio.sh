#!/usr/bin/env bash
# Batch-score the whole portfolio into CSV, then refresh the evaluation report.
set -ex
uv run python -m src.model.batch_inference "$@"
uv run python -m src.evaluation report
