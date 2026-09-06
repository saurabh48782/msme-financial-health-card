#!/usr/bin/env bash
# Container entrypoint.
#
# The model is baked in as ./model_artifact by the training run, so there is no
# registry call at boot and an MLflow outage cannot stop the API from serving.
# With no artifact present the API still comes up on the rubric and the fitted PD
# fallback, and /readiness reports the model as absent rather than lying.
set -ex

exec uvicorn src.api.app:app \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --workers "${WORKERS:-4}" \
  --no-access-log
