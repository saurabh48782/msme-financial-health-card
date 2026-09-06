#!/usr/bin/env bash
# Full training pipeline: data -> features -> calibrate -> train -> evaluate.
set -ex

# MLflow's log_model wants a pip-style requirement list and cannot read uv.lock,
# so the lockfile stays the source of truth and this renders it. Gitignored and
# regenerated on every run so a stale export can never be logged with a model.
uv export --no-dev --no-hashes --format requirements-txt > .mlflow-requirements.txt

uv run python -m src.data.data_orchestration "$@"

# Report the pillar fit. It does not overwrite params.yaml: a rubric weight change
# is a reviewed commit, not something a training job does behind your back.
uv run python -m src.scoring.pillar_calibration || true

uv run python -m src.model.model_orchestration
uv run python -m src.model.batch_inference
uv run python -m src.evaluation report

echo
echo "Trained; the bundle is in ./model_artifact and the run is logged to MLflow."
echo "Review data/eval/reports/latest.md."
