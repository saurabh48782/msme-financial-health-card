# Beyond scope — what a production deployment needs that this does not build

This project is deliberately bounded by what the data can support: 50,000 rows of
one flat snapshot per firm, with no event stream, no consent layer, no repayment
outcomes and no time series. Several components a real lending platform needs
would therefore have been *simulated interfaces* rather than integrations — code
that looks like a capability but is only shaped like one.

Each item below is a considered omission with the design already settled, not an
oversight. Where the design is drawn, it is drawn as designed-not-built in
[ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md).

## Versioned card store and audit log

**Not built.** Cards are not persisted. Listings come from the batch-scored
portfolio CSV; an individual card is scored on demand through the full five-layer
pipeline on every request.

**Why.** A write-through Postgres store plus its migrations, connection pool and
advisory-locked migration step is real infrastructure whose only demonstrable
benefit here would be storing rows nobody reads back. The property that actually
matters for an RBI digital-lending audit — *can you reproduce the decision you
made?* — is delivered instead by stamping every card with `model_version`,
`policy_version`, `rubric_version` and `feature_snapshot_hash`. The same inputs
under the same versions reproduce the card exactly, and re-scoring on demand
means a card is never served from a row that predates the current policy.

**What production needs.** The schema in ARCHITECTURE_DIAGRAM.md §3: an append-only
`health_cards` table with `(msme_id, version)` unique, the served card kept as
`jsonb` so the exact bytes a lender saw are recoverable, and a per-MSME lock so two
concurrent refreshes cannot race on the version number. That last point is why
this repo has no `refresh` endpoint: without a store there is nothing to
serialise, and an endpoint that merely re-did what `GET` already does would be
surface area pretending to be a capability.

## Consent and per-source ingestion

**Not built.** No `POST /api/v1/ingest/{source}`, no consent artifact registry.
`consent_artifact_id` exists as a field on every card and travels through the
scoring path, but nothing populates it.

**Why.** The dataset has no consent layer and no per-source feeds — it is one
pre-joined row per firm. Ingestion endpoints would have accepted fields, filtered
them against a hand-written allowlist, and written them nowhere. An Account
Aggregator consent artifact is a signed object with a real lifecycle; a Pydantic
model with a `covers()` method is a mock of one, and mocking the authorisation
layer of a lending system is worse than leaving the seam visible.

**What production needs.** AA consent as the gate in front of every data pull, with
the artifact id recorded on each card so a decision traces to the authorisation
that permitted it, plus revocation and expiry handling that fails a pull closed.

## OCEN / ULI adapters

**Not built as code.** The integration strategy is documented in
[INTEGRATION.md](INTEGRATION.md).

**Why.** OCEN is a real specification. A hand-invented loan-application and offer
shape would be confidently wrong in ways that read as unfamiliarity with the
standard. The service layer is already HTTP-agnostic — nothing in `src/service`
imports FastAPI — so an adapter is a translation layer over calls that already
exist, which is the point worth making in an interview rather than in guessed
JSON.

## Edge concerns: rate limiting and body-size caps

**Not built.** API-key auth (`secrets.compare_digest`), request-id propagation and
opt-in CORS are in `src/api/security.py`. A sliding-window per-IP limiter and a
pre-buffering body-size cap are not.

**Why.** Both belong in front of the application — an API gateway, an ALB, an
ingress — not in the process being protected. In-process rate limiting is
per-worker, so it is wrong by a factor of `WORKERS` the moment you scale out, and
it resets on deploy. Implementing it in the wrong tier is a liability that looks
like diligence. The application-level batch cap that *does* belong here is kept:
`POST /api/v1/score` rejects a batch over `api.max_batch_size` with 413.

## Model promotion and a second environment

**Not built.** Runs are logged to MLflow and the model is registered, so every
card's `model_version` resolves to a run with its parameters, metrics, plots and a
`params.yaml` snapshot. What is absent is the lifecycle *around* that: no
`staging` → `production` aliases, no transaction-safe promote script, no
boot-time registry download.

**Why.** Promotion machinery exists to move a model between environments, and
there is only one environment here. The training run writes `./model_artifact`
and the API loads it — which also means the API makes no network call to the
tracking server at boot or on the request path, so an MLflow outage cannot stop
scoring. Tracking is the half that earns its place, because those are the numbers
quoted in [MODEL_CARD.md](MODEL_CARD.md).

**What production needs.** Alias-based resolution (`models:/<name>@production`)
pulled at container start, an assign-then-drop ordering on the alias swap so a
failed promotion never leaves the alias unset, and a gate that refuses to promote
a version whose eval report has a threshold breach.

## Unsupervised anomaly detection

**Not built.** Layer C is eight named consistency rules, with no IsolationForest
or other novelty detector over the same metrics.

**Why.** Its output cannot be acted on. "The model found you unusual" is not a
lawful reason to decline credit under RBI digital-lending norms, so the score can
only order a review queue — and `max_severity` already orders that queue by
something a reviewer can open and check. Against that, a forest has to be fitted,
carried in the scoring bundle, and min-max normalised against the training
portfolio so that a real-time score does not depend on who else was in the batch.
Real machinery for a number with no decision attached.

**What production needs.** The rules only catch what someone thought to write
down. With observed defaults you would fit the detector against *outcomes* rather
than against shape, which makes its output actionable — a queue ordered by
realised risk instead of by novelty.
