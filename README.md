# MSME Financial Health Card

<div align="center">

[![CI](https://github.com/saurabh48782/msme-financial-health-card/actions/workflows/ci.yml/badge.svg)](https://github.com/saurabh48782/msme-financial-health-card/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![Tests](https://img.shields.io/badge/tests-313%20passing-2ea44f?style=flat-square&logo=pytest&logoColor=white)
![mypy](https://img.shields.io/badge/mypy-strict%20clean-2A6DB2?style=flat-square)

**Core**

![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-v2%20contracts-E92063?style=flat-square&logo=pydantic&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-multi--stage%20targets-2496ED?style=flat-square&logo=docker&logoColor=white)

**Modelling & Explainability**

![XGBoost](https://img.shields.io/badge/XGBoost-PD%20%2B%20eligibility-EC4E20?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-NNLS%20%2B%20metrics-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-TreeExplainer-6A1B9A?style=flat-square)
![MLflow](https://img.shields.io/badge/MLflow-tracking%20%2B%20registry-0194E2?style=flat-square&logo=mlflow&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-data%20%2B%205%20stages-13ADC7?style=flat-square&logo=dvc&logoColor=white)

**Dashboard**

![Jinja](https://img.shields.io/badge/Jinja2-server--rendered-B41717?style=flat-square&logo=jinja&logoColor=white)
![Bootstrap](https://img.shields.io/badge/Bootstrap%205-SRI--pinned%20CDN-7952B3?style=flat-square&logo=bootstrap&logoColor=white)
![Chart.js](https://img.shields.io/badge/Chart.js-FF6384?style=flat-square&logo=chartdotjs&logoColor=white)

**Quality Gates**

![uv](https://img.shields.io/badge/uv-DE5FE9?style=flat-square&logo=uv&logoColor=white)
![Ruff](https://img.shields.io/badge/Ruff-lint%20%2B%20format-D7FF64?style=flat-square&logo=ruff&logoColor=black)
![pytest](https://img.shields.io/badge/pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white)
![pre-commit](https://img.shields.io/badge/pre--commit-12%20hooks-FAB040?style=flat-square&logo=precommit&logoColor=black)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-CI-2088FF?style=flat-square&logo=githubactions&logoColor=white)

**Data Rails Modelled**

![GSTN](https://img.shields.io/badge/GSTN-filing%20discipline-FF9933?style=flat-square)
![UPI](https://img.shields.io/badge/UPI%20%2F%20NPCI-transaction%20cadence-097939?style=flat-square)
![Account Aggregator](https://img.shields.io/badge/Account%20Aggregator-bank%20flows-000080?style=flat-square)
![EPFO](https://img.shields.io/badge/EPFO-payroll%20regularity-1F4E79?style=flat-square)
![TReDS](https://img.shields.io/badge/Invoices%20%2F%20TReDS-receivables-6B4E9E?style=flat-square)

</div>

Explainable, alternative-data credit scoring for **New-to-Credit (NTC)** and
**New-to-Bank (NTB)** Indian MSMEs: the firms banks cannot underwrite because bureau
scores, audited financials and collateral records do not exist for them.

It consumes 33 consent-based GST, UPI, bank/AA, EPFO and invoice fields and returns a
0–100 Financial Health Score with six pillar sub-scores, a calibrated probability of
default, typed risk flags, SHAP-backed reason codes, and a credit recommendation with a
sized limit, in roughly 250 ms, every decision stamped with the model, policy and rubric
version that produced it.

> **Trained on synthetic data.** `Probability_of_Default` is a generated label, not
> observed default. See [docs/MODEL_CARD.md](docs/MODEL_CARD.md) for what that means and
> what retraining on real performance would require.

## Table of Contents

- [MSME Financial Health Card](#msme-financial-health-card)
  - [Table of Contents](#table-of-contents)
  - [1. Overview](#1-overview)
  - [2. Results](#2-results)
  - [3. Installation](#3-installation)
  - [4. Data \& Pipelines](#4-data--pipelines)
  - [5. Local Development](#5-local-development)
  - [6. API Reference](#6-api-reference)
  - [7. Configuration](#7-configuration)
  - [8. Evaluation](#8-evaluation)
  - [9. Testing](#9-testing)
  - [10. CI/CD](#10-cicd)
  - [11. Project Structure](#11-project-structure)
  - [12. Limitations](#12-limitations)
  - [13. Further Documentation](#13-further-documentation)

## 1. Overview

A New-to-Credit MSME has no bureau score, no audited financials and no collateral. But
it does file GST, take UPI payments, run payroll through EPFO and raise invoices. This
service scores a firm on that evidence instead.

The engine is hybrid, because the dataset's `Financial_Health_Score` **is itself a
weighted formula** over its six pillars, so pure ML on it would be circular and
unexplainable to a credit officer.

| Layer | Owns | Implementation |
|---|---|---|
| **A. Pillar rubrics** | the score | `src/scoring/pillars.py`. Transparent, monotone, exactly decomposable; weights fitted by NNLS against the dataset's own pillars |
| **B. XGBoost** | the risk | `src/model/components.py`. PD regressor + eligibility classifier over the 33 raw inputs, NaN handled natively |
| **C. Anomaly rules** | the queue | `src/scoring/anomaly.py`. Eight deterministic consistency rules, each naming the invariant it broke |
| **D. Policy engine** | the decision | `src/scoring/policy.py`. Risk band, knock-outs, limit sized by cashflow and capped by DSCR headroom, tenure, pricing |
| **E. Explainability** | the reasons | `src/scoring/explainer.py`. Rubric arithmetic plus SHAP attribution, both as plain English |

Every layer is timed by `traced_stage`: ~250 ms per card (features 58 ms · ML 66 ·
pillars 47 · explain 46 · anomaly 28 · policy 0.2).

## 2. Results

Model version `18`, policy `policy-1.0.0`, rubric `rubric-1.0.0`. Trained on the 33
alternative-data inputs alone, with every label-block column held behind a hard allowlist.

| | 20% holdout (10,000 firms) | Full portfolio (50,000) |
|---|---|---|
| Probability of default | **adj. R² 0.9053**, MAE 0.02045 | adj. R² 0.9554, MAE 0.01424, ECE 0.00104 |
| Credit eligibility | **ROC-AUC 0.9776**, KS 0.8317, PR-AUC 0.9923 | ROC-AUC 0.9886, KS 0.8831 |
| Financial Health Score | adj. R² 0.8155, MAE 2.175 pts | adj. R² 0.8148, **MAE 2.150 pts** |
| Six pillar rubrics | n/a | adj. R² 0.876–0.994, all exactly decomposable |
| Agreement with reference decisions | n/a | risk band 86.45%, eligibility 93.29%, limit MAPE 11.89% |
| Segment neutrality | n/a | approval spread **2.4 pts** across NTC/NTB/ETC, narrower than the reference decisions' own 5.0 pts |

The top PD driver is `GST_Filing_Timeliness_Pct` at 34.5% of gain, and no bureau-style
variable appears anywhere in the top ten. Thin-file firms are not penalised for the
absence of a credit record: where a repayment driver is missing, its rubric weight is
redistributed across evidence that exists rather than imputed.

## 3. Installation

**Prerequisites:** Python 3.12, [uv](https://docs.astral.sh/uv/). Optionally direnv, and
Docker for the containerised targets. No database and no MLflow server are required.

```bash
git clone git@github.com:saurabh48782/msme-financial-health-card.git
cd msme-financial-health-card

uv sync --all-groups          # create .venv and install everything (dev + notebook)
cp .env.example .env          # then edit; every var in it is optional
direnv allow                  # optional: loads .env, activates the venv, sets PYTHONPATH
uv run pre-commit install     # optional: run the gate on every commit
```

Place the raw dataset at `data/raw/msme_synthetic_50k.csv` (DVC-tracked, not in git),
then run the pipeline end to end:

```bash
uv run python -m src.data.data_orchestration     # raw CSV -> features (~4 s)
uv run python -m src.model.model_orchestration   # train, calibrate, log to MLflow (~60 s)
uv run python -m src.model.batch_inference       # score all 50,000 firms (~4 s)
uv run uvicorn src.api.app:app --port 8000       # dashboard + API
```

Or all of it in one command: `bash scripts/train.sh`.

Then open <http://localhost:8000>. Without a trained artifact the API still comes up on
the rubric plus the fitted PD fallback, and `/readiness` reports the model as absent.

Docker, if you prefer:

```bash
docker compose --profile dev up            # MLflow + API
docker compose --profile train up          # the training pipeline
docker compose --profile all up --build    # everything, including the test targets
```

## 4. Data & Pipelines

| Property | Value |
|---|---|
| Source | `data/raw/msme_synthetic_50k.csv`, synthetic |
| Shape | 50,000 rows × 45 columns = 1 identifier + 33 inputs + 11 supervision targets |
| Segment mix | NTC 21,068 · NTB 14,967 · Existing-to-Credit 13,965 |
| Structural nulls | `EMI_On_Time_Rate_Pct` is null for all 32,556 firms with no existing loan |
| Split | 80/20, stratified on `Customer_Segment` |
| Model features | the 33 raw inputs only; derived ratios are ablated out, label-block columns blocked by allowlist |

```bash
uv run python -m src.data.data_orchestration       # validate -> winsorise -> derive 14 ratios
uv run python -m src.scoring.pillar_calibration    # refit the rubrics; prints YAML to review
uv run python -m src.model.model_orchestration     # train -> evaluate -> MLflow -> ./model_artifact
uv run python -m src.model.batch_inference         # score the whole portfolio to CSV
bash scripts/train.sh                              # all of the above, then the eval report
bash scripts/score_portfolio.sh                    # re-score + refresh the report, no retrain
```

Calibration is advisory: it prints a paste-ready YAML block and writes a JSON artifact,
but never overwrites `params.yaml`. A credit-policy change is a reviewed commit.

**What lands where**

```
data/processed/features.csv            validated + winsorised + derived
data/processed/winsor_limits.json      the clipping bounds actually applied
data/processed/portfolio_scores.csv    all 50,000 scored cards, the read-only store
data/artifacts/pillar_calibration.json the fitted rubric weights, for review
data/eval/reports/latest.{json,md}     the evaluation report, plus a timestamped history
model_artifact/scoring_bundle.joblib   the model baked in for serving
mlflow.db / mlruns/                    local MLflow tracking store
```

**Data versioning.** DVC is an optional extra (`uv sync --extra data`); `data/` is never
git-tracked.

```bash
dvc add data/raw && git add data/raw.dvc   # the raw input
dvc commit                                 # record the dvc.yaml stage outputs
dvc repro                                  # rerun only the stages whose inputs changed
dvc push / dvc pull                        # needs a configured S3 remote
```

`dvc.yaml` declares five stages (`prepare`, `calibrate`, `train`, `score`,
`evaluate`), each naming the `params.yaml` sections it depends on.

## 5. Local Development

```bash
bash scripts/develop.sh                                # uvicorn with --reload
uv run uvicorn src.api.app:app --port 8000 --reload    # the same thing, explicitly
```

No compose stack and no database to bring up. The store is a read-only view over the
batch-scored portfolio CSV and an individual card is scored on demand, so once the
pipelines have run all 50,000 firms are drillable.

**The nine views.** Server-rendered Jinja2 + Bootstrap 5, Chart.js, vanilla ES modules.
No npm, no bundler, no SPA.

| Page | What it shows |
|---|---|
| [`/`](http://localhost:8000/) | Headline stats plus the segment-lift table |
| [`/portfolio`](http://localhost:8000/portfolio) | FHS histogram, risk-band mix, approval by segment, industry mix, decline reasons |
| [`/healthcards`](http://localhost:8000/healthcards) | Paginated, filterable list of all 50,000 |
| [`/healthcard/MSME0000042`](http://localhost:8000/healthcard/MSME0000042) | **A complete Health Card**: gauge, pillar radar, per-driver arithmetic, reason codes, SHAP bars, risk flags, provenance |
| [`/simulator?msme_id=MSME0000002`](http://localhost:8000/simulator?msme_id=MSME0000002) | A declined firm; raise GST compliance and cut overdraft use and watch it become eligible with a limit |
| [`/assess`](http://localhost:8000/assess) | Type a firm's 33 fields in, get the decision back |
| [`/anomalies`](http://localhost:8000/anomalies) | Rule-trigger frequency and the severity-ordered review queue |
| [`/metrics`](http://localhost:8000/metrics) | Threshold gates, PD reliability curve, pillar adj. R² vs gate, fairness slices |
| [`/search`](http://localhost:8000/search) | Filter by segment/industry/location; an ID lookup redirects to the card |

Plus [`/docs`](http://localhost:8000/docs) for the OpenAPI reference.

**Everyday commands**

```bash
bash scripts/unit_test.sh          # tests/unit
bash scripts/integration_test.sh   # tests/integration (--full adds the marked tests)
bash scripts/precommit_check.sh    # ruff, ruff-format, mypy strict, detect-secrets, hadolint
uv run pytest -m eval              # the metric regression gate (needs the dataset)
uv run mypy src                    # strict, must stay clean
uv run ruff check --fix src tests
uv run ruff format src tests
```

**Exploration**

```bash
uv run jupyter lab notebooks/01_eda_and_feature_design.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=900 notebooks/01_eda_and_feature_design.ipynb
```

The notebook calls the pipeline's own functions rather than reimplementing them and is on
no code path. Needs `uv sync --all-groups` (the `notebook` group).

## 6. API Reference

All JSON routes live under `/api/v1`. Set `HEALTHCARD_API_KEY` and each requires an
`X-API-Key` header; leave it unset and the API is open, which is the local default.

| Method | Route | Purpose |
|---|---|---|
| `POST` | `/api/v1/score` | Score one or more MSMEs without persisting anything (max 500 per batch) |
| `GET` | `/api/v1/healthcards` | List scored MSMEs, paginated and filterable |
| `GET` | `/api/v1/healthcard/{msme_id}` | One complete Health Card |
| `GET` | `/api/v1/credit-recommendation/{msme_id}` | The credit decision with its knock-out reasons |
| `POST` | `/api/v1/simulate/{msme_id}` | Re-score with overridden inputs and report the before/after delta |
| `GET` | `/api/v1/explain/{msme_id}` | Why this MSME scored what it scored |
| `GET` | `/api/v1/explain/{msme_id}/pillars` | Exact pillar decomposition; contributions sum to the score |
| `GET` | `/api/v1/risk-flags/{msme_id}` | Typed risk flags for one MSME |
| `GET` | `/api/v1/anomalies` | Portfolio anomaly queue, most severe first |
| `GET` | `/api/v1/portfolio/summary` | Distributions, cohort cuts and approval-rate lift by segment |
| `GET` | `/api/v1/models/metrics` | Holdout metrics, calibration curve and fairness slices |
| `GET` | `/api/v1/models/versions` | Registry versions and their aliases |
| `GET` | `/healthcheck` | Liveness; touches no dependency |
| `GET` | `/readiness` | Dependency probe; 503 when the model or store is unhealthy |

Errors map to 503 (dependency), 504 (timeout), 422 (schema) and 500 (generic); the detail
goes to the log, never to the caller. Rate limiting and body-size caps are left to the
edge. See [docs/BEYOND_SCOPE.md](docs/BEYOND_SCOPE.md).

## 7. Configuration

`params.yaml` is the single source of truth for every tunable: 408 lines across 15
top-level sections. Nothing in `src/` hard-codes a weight, threshold, path or band edge.
`${VAR}` placeholders resolve at load (an unset **or empty** var raises;
`${VAR:-default}` falls back), and `validate_config` rejects every bad shape at startup.

| Section | Governs |
|---|---|
| `data_paths` | every input and output location |
| `schema` | column groups, the feature allowlist, the label block |
| `preprocessing` | winsorisation limits, ordinal standing maps |
| `pillars` · `grades` | the six rubrics (drivers, weights, ramps, band) and the grade edges |
| `model_params` | XGBoost hyperparameters, split, seed |
| `policy` | PD band edges, knock-outs, limit sizing, DSCR headroom, tenure, pricing |
| `explainability` | SHAP background size, top-N drivers, narration floor |
| `mlflow_config` | tracking URI, experiment, registered model name |
| `api` | title, batch cap, page size, timeouts, key and CORS bindings |
| `evaluation` | the threshold gates enforced by `pytest -m eval` |

**Environment variables.** All optional. Copy `.env.example` to `.env` (gitignored;
direnv picks it up via `.envrc`).

| Variable | Default | Purpose |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | Unset → local SQLite store |
| `LOG_LEVEL` · `LOG_TARGET` · `LOG_FORMAT` | `INFO` · `file` · tty-detected | structlog config; `stdout` in containers |
| `WORKERS` · `PORT` | `4` · `8000` | Uvicorn, in the container entrypoint |
| `HEALTHCARD_API_KEY` | unset | Set → `X-API-Key` required on `/api/*` |
| `CORS_ALLOW_ORIGINS` | unset | Comma-separated allowlist |
| `API_ROOT_PATH` | unset | When served behind a path-stripping proxy |
| `ALLOW_MISSING_MODEL` | unset | Boot on the rubric + PD fallback with no artifact |
| `AWS_PROFILE` · `AWS_REGION` | unset | DVC S3 remote |

## 8. Evaluation

The harness lives in `src/evaluation/` and has no HTTP surface. It scores the full
portfolio against the dataset's own supervision columns and writes a JSON report plus a
markdown twin into `data/eval/reports/`.

```bash
uv run python -m src.evaluation calibrate   # pillar rubric R² against the dataset
uv run python -m src.evaluation backtest    # PD calibration, discrimination, limit sanity
uv run python -m src.evaluation fairness    # slice tables by segment / location / industry
uv run python -m src.evaluation report      # full JSON + markdown, with gate pass/fail
uv run pytest -m eval                       # the regression gate (excluded by default)
```

**Portfolio metrics.** Model `18`, all 50,000 firms. All configured thresholds met.

| Metric | Value | Gate |
|---|---|---|
| Financial Health Score adj. R² / MAE | 0.8148 / 2.150 points | MAE ≤ 2.5 |
| PD adj. R² / MAE | 0.9554 / 0.01424 | MAE ≤ 0.025 |
| PD expected calibration error | 0.00104 | n/a |
| Eligibility ROC-AUC / KS | 0.9886 / 0.8831 | AUC ≥ 0.95 |
| Risk-band / eligibility agreement | 0.8645 / 0.9329 | n/a |
| Credit limit MAPE | 0.1189 | ≤ 0.15 |

**Pillar rubric fit.** Every R² is *adjusted*, because plain R² rises with predictor count
whether or not a driver carries signal, so a rubric must not clear the gate by being
wider than its neighbours.

| Pillar | Adj. R² | Drivers | MAE | Gate |
|---|---|---|---|---|
| Compliance | 0.9941 | 4 | 0.370 | ≥ 0.75 |
| Cash Flow | 0.9903 | 5 | 0.506 | ≥ 0.75 |
| Business Growth | 0.9423 | 3 | 2.274 | ≥ 0.75 |
| Payment Behaviour | 0.9418 | 4 | 1.065 | ≥ 0.75 |
| Revenue Consistency | 0.8986 | 4 | 2.035 | ≥ 0.75 |
| Business Stability | 0.8755 | 5 | 2.216 | ≥ 0.75 |

Payment Behaviour fits at 0.997 on fully observed rows but 0.942 across the whole
portfolio. The gap is the structural null, and the lower number is the one the portfolio
sees.

**Fairness.** Deviation is measured on model *error*, not on outcome rates. Cohorts may
legitimately differ in creditworthiness, but the model's error must not.

| Segment | Firms | PD MAE | PD bias | Approval (ours) | Approval (dataset) |
|---|---|---|---|---|---|
| Existing-to-Credit | 13,965 | 0.01331 | −0.00004 | 0.741 | 0.776 |
| NTB | 14,967 | 0.01439 | −0.00010 | 0.724 | 0.737 |
| NTC | 21,068 | 0.01476 | −0.00007 | 0.717 | 0.726 |

Largest PD-error deviation across all slices (segment, location, industry): **0.0930
points** against a gate of 3.0. Approval rates span 2.4 points, narrower than the
reference decisions' own 5.0. Thin-file share by segment is NTC 100% · NTB 76.8% ·
ETC 0%, so that is not an artefact of a cohort that happens to look the same.

Read the numbers alongside [docs/MODEL_CARD.md](docs/MODEL_CARD.md): the data is
synthetic, and the eligibility agreement of 93.3% partly reflects a *deliberate*
disagreement where the DSCR cap declines firms the reference decisions approve.

## 9. Testing

**313 tests: 266 unit + 34 integration + 13 eval-gated.** Tests mirror the `src/` tree.

```bash
bash scripts/unit_test.sh          # tests/unit
bash scripts/integration_test.sh   # tests/integration, in-process, no DB/MLflow/model
uv run pytest                      # unit + integration (eval excluded by addopts)
uv run pytest -m eval              # the metric regression gate
```

| Area | Tests | What it covers |
|---|---|---|
| `tests/unit/utils` | 45 | `${ENV}` resolution, config validation, PII redaction |
| `tests/unit/data` | 21 | Validator rejections, derived ratios, structural nulls preserved |
| `tests/unit/scoring` | 116 | Ramps, weight sums, exact decomposition, monotonicity, band edges, limits, DSCR cap, scalar ≡ vectorised, calibration idempotency |
| `tests/unit/model` | 19 | Calibration, KS/AUC helpers, stratified split, fairness deviation |
| `tests/unit/schemas` | 17 | Aliases, ranges, round-trips, bounded scores |
| `tests/unit/service` | 10 | On-demand scoring, determinism, override validation, aggregation |
| `tests/unit/api` | 24 | Auth, middleware order, liveness independence, CORS parsing |
| `tests/unit/test_leakage.py` | 8 | No label can reach a model |
| `tests/unit/test_structural_nulls.py` | 6 | Nulls modelled, weights redistributed |
| `tests/integration/test_api_inprocess.py` | 34 | Every route, both policy branches, HTML views, chart payloads |
| `tests/integration/test_metrics_regression.py` | 13 | Every published metric claim, marker-gated |

The default suite needs no dataset, no Docker, no database and no MLflow:
`conftest.raw_frame` falls back to `tests/stubs.synthetic_raw_frame`, which reproduces the
same columns *and* the same structural-null identity. `pytest -m eval` is the heavy gate:
it scores all 50,000 firms and asserts every threshold in `params.yaml`, PD calibration
per cohort of every slice column, and that the segment approval gap is no wider than the
dataset's own.

## 10. CI/CD

`.github/workflows/ci.yml` runs the **same three scripts** developers run and the same
three the container test targets run, so dev, container and CI cannot diverge.

```
Pre-commit check  ->  Unit tests  ->  Integration tests  ->  Build the api image
```

The image job only runs on a push to `main`/`master`, and only after the checks pass.

**Pre-commit** (12 hooks): `ruff` (E, F, I, N, W, UP, S, B, where `S` is bandit security),
`ruff-format`, `mypy --strict` (clean over 59 files), `detect-secrets` against a committed
baseline, `hadolint`, plus the standard whitespace/YAML/TOML/large-file checks. The mypy
hook runs `uv run mypy` against the project's own environment, so it gives the same
verdict as your terminal.

**Docker targets.** Multi-stage off `python:3.12-slim`, dependencies installed before the
source is copied and cached behind a BuildKit mount.

| Target | What it does | Dev deps |
|---|---|---|
| `api` | Production server on port 8000, non-root uid 10001, liveness healthcheck | No |
| `trainer` | Runs `scripts/train.sh` | No |
| `unit_test` | Runs `scripts/unit_test.sh` | Yes |
| `integration_test` | Runs `scripts/integration_test.sh` | Yes |
| `precommit_check` | Runs `scripts/precommit_check.sh` | Yes |

```bash
docker compose --profile unit_test up --build
docker compose --profile integration_test up --build   # waits for /readiness first
docker compose --profile precommit_check up --build
```

## 11. Project Structure

```
├── params.yaml                    every tunable: weights, ramps, band edges, thresholds
├── dvc.yaml                       5 stages: prepare -> calibrate -> train -> score -> evaluate
├── Dockerfile                     api, trainer, unit/integration/precommit targets
├── docker-compose.yaml            profiled services (dev / train / all / CI targets)
├── src/
│   ├── utils/                     config with ${ENV} resolution, PII redaction, stage tracing
│   ├── data/                      load -> validate -> winsorise -> derive
│   ├── scoring/                   Layers A, C, D, E plus the rubric calibrator
│   ├── model/                     Layer B: leakage firewall, XGBoost heads, reliability, fairness
│   ├── schemas/                   pydantic contracts (33 inputs, card, credit, portfolio)
│   ├── data_access/csv_store.py   read-only store over the batch-scored portfolio
│   ├── service/                   scoring orchestration, simulator, portfolio analytics
│   ├── api/                       app factory, security middleware, 8 routers, form spec
│   ├── templates/                 12 Jinja2 templates (9 views + base + macros + 404)
│   └── evaluation/                offline harness + CLI (no HTTP surface)
├── tests/
│   ├── stubs.py                   the only shared doubles
│   ├── unit/                      mirrors src/, 266 tests
│   └── integration/               in-process API tests + the eval gate, 34 + 13
├── scripts/                       train, develop, start_api, score_portfolio, the 3 gates
├── notebooks/                     the EDA behind the design (on no code path)
├── docs/                          architecture, integration, model card, out-of-scope
└── data/                          never git-tracked; DVC handles raw + reports
```

~7,700 lines across 59 modules, plus ~1,450 lines of templates.

## 12. Limitations

Full treatment in [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

1. **Synthetic data caps the score.** The generator's own pillar formula explains only
   ~83% of its health score, so an FHS MAE near 2.1 points is close to the achievable
   ceiling rather than evidence of a great model.
2. **PD is a generated label.** Retraining on observed default is mandatory before real
   use, and the rubric weights should be expected to move when it happens.
3. **No time series.** Monthly aggregates only, so the card shows no trend charts rather
   than generating a history the data does not contain.
4. **Knock-out thresholds are unvalidated policy.** They sit at the observed floor of the
   eligible population, which is a fitting choice as much as a policy one.
5. **Limit sizing diverges from the reference on 1.7% of firms**, because the DSCR
   headroom cap enforces serviceability and the reference decisions do not.
6. **The 3% rubric weight floor is judgement, not evidence**, the one place where a
   belief overrides the fit.
7. **Anomaly detection is rules-only.** Shapes nobody wrote a rule for are not caught.
8. **Not a fraud engine, and not a bureau substitute.** The anomaly layer orders a review
   queue; it does not adjudicate.

## 13. Further Documentation

* [docs/ARCHITECTURE_DIAGRAM.md](docs/ARCHITECTURE_DIAGRAM.md): deployment topology, the scoring sequence, the data model, and why the engine is hybrid
* [docs/MODEL_CARD.md](docs/MODEL_CARD.md): intended use and misuse, full metrics, top PD drivers, fairness slices, limitations, governance
* [docs/INTEGRATION.md](docs/INTEGRATION.md): ULI / OCEN / AA / GSTN strategy, consent enforcement, adoption path, production gaps
* [docs/BEYOND_SCOPE.md](docs/BEYOND_SCOPE.md): what a production deployment needs that this deliberately does not build, and why
* [notebooks/01_eda_and_feature_design.ipynb](notebooks/01_eda_and_feature_design.ipynb): the EDA behind the design, each finding mapped to the `params.yaml` key it set
