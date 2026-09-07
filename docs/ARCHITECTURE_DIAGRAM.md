# Architecture

> **Built vs. designed.** This document is the target architecture. Dashed boxes are
> *designed but deliberately not built* here — the consent and
> ingestion layer, the OCEN adapters, and the versioned Postgres card store. The
> dataset is one flat snapshot per firm with no consent layer and no event stream,
> so building those would have meant simulating an interface rather than
> integrating one. [docs/BEYOND_SCOPE.md](BEYOND_SCOPE.md) states what each would
> take. Everything drawn solid runs and is tested.

## 1. Deployment topology

```mermaid
flowchart TB
    subgraph consent["Consent layer"]
        AA["Account Aggregator<br/>(RBI NBFC-AA)"]
        ULI["ULI / Public Tech Platform"]
    end

    subgraph sources["Consent-based data sources"]
        GSTN["GSTN<br/>returns, filing history"]
        NPCI["NPCI / UPI<br/>inflow, ticket size, counts"]
        BANK["Bank statements via AA<br/>credits, debits, balance, OD"]
        EPFO["EPFO<br/>payroll, headcount"]
        INV["Invoice / TReDS<br/>receivable ageing"]
    end

    subgraph ingest["Ingestion · designed, not built"]
        API_IN["POST /api/v1/ingest/{source}<br/>per-source field allowlist"]
        CONSENT["POST /api/v1/consent/artifacts"]
    end

    subgraph pipeline["Feature pipeline · src/data"]
        BRONZE["Raw store<br/>CSV, DVC-tracked"]
        SILVER["Validate → winsorise →<br/>derive ratios"]
    end

    subgraph engine["Scoring engine · src/scoring"]
        A["Layer A · Pillar rubrics<br/>6 sub-scores, exact decomposition"]
        B["Layer B · XGBoost<br/>PD + eligibility, calibrated"]
        C["Layer C · Anomaly<br/>8 named consistency rules"]
        D["Layer D · Policy engine<br/>band, knock-outs, limit, pricing"]
        E["Layer E · Explainability<br/>SHAP + waterfall + reason codes"]
    end

    subgraph serve["Serving"]
        SCORED[("Batch-scored portfolio<br/>CSV, DVC-tracked")]
        STORE[("Postgres<br/>versioned cards + audit")]
        REST["REST API<br/>src/api/routers"]
        OCEN["OCEN adapters<br/>LSP ↔ lender"]
        DASH["Dashboard<br/>Jinja2 + Bootstrap"]
    end

    AA --> sources
    ULI --> sources
    CONSENT -.authorises.-> API_IN
    sources --> API_IN --> BRONZE
    sources --> BRONZE
    BRONZE --> SILVER
    SILVER --> A --> D
    SILVER --> B --> D
    SILVER --> C --> D
    A --> E
    B --> E
    D --> SCORED
    E --> SCORED
    SCORED --> REST
    SCORED --> DASH
    REST -.-> OCEN
    SCORED -.would persist to.-> STORE

    classDef planned stroke-dasharray: 5 4,color:#666
    class API_IN,CONSENT,OCEN,STORE planned
```

The built path is: raw CSV → feature pipeline → the five scoring layers → the
batch-scored portfolio, read by the REST API and the dashboard. An individual
card is re-scored on demand through the same five layers, so a card is never
served from a stale row.

## 2. Scoring pipeline for one MSME

Each stage is wrapped in `traced_stage`, so the p95 latency claim is measured per
layer rather than asserted. A slow card names the layer that was slow.

```mermaid
sequenceDiagram
    autonumber
    participant C as Caller (lender / LSP)
    participant API as FastAPI router
    participant S as HealthCardService
    participant F as Feature pipeline
    participant A as Pillars (Layer A)
    participant M as XGBoost (Layer B)
    participant N as Anomaly (Layer C)
    participant P as Policy (Layer D)
    participant X as Explainer (Layer E)

    C->>API: POST /api/v1/score  (or GET /healthcard/{id})
    API->>API: validate against MSMEFeatures (422 on bad input)
    API->>S: score_records / get_card
    S->>F: winsorise → derive
    Note over F: fitted winsorisation limits reused,<br/>never refitted per request
    F->>A: 6 pillar rubrics
    Note over A: missing repayment driver →<br/>weight redistributed, never imputed
    F->>M: calibrated PD + eligibility probability
    F->>N: 8 named consistency rules
    A->>P: cashflow pillar (sizes the limit)
    M->>P: probability of default (sets the band)
    N->>P: risk flags (critical → knock-out)
    P->>X: decision + reasons
    A->>X: exact pillar decomposition
    M->>X: SHAP contributions
    X-->>S: reason codes in plain English
    Note over S: card stamped with model, policy and rubric<br/>version plus the feature hash — the same inputs<br/>reproduce the same card, so nothing need be stored
    S-->>API: HealthCard
    API-->>C: 200 with the full card
```

## 3. Data model — designed, not built

The service stores no cards: listings come from the batch-scored portfolio and
an individual card is scored on demand, so reproducibility rests on the version
stamps rather than on a stored row. This is the schema a write-through deployment
would use, and it is what the provenance fields on every card already carry.

```mermaid
erDiagram
    MSME_PROFILES ||--o{ HEALTH_CARDS : "scored into"
    CONSENT_ARTIFACTS ||--o{ HEALTH_CARDS : "authorises"

    MSME_PROFILES {
        text msme_id PK
        jsonb features "the 33 consented inputs"
        text consent_artifact_id
        timestamptz updated_at
    }
    CONSENT_ARTIFACTS {
        text consent_id PK
        text msme_id FK
        jsonb sources "gst, upi, bank, epfo, invoices"
        text status "ACTIVE | PAUSED | REVOKED | EXPIRED"
        timestamptz granted_at
        timestamptz expires_at
    }
    HEALTH_CARDS {
        bigint id PK
        text msme_id
        int version "monotonic per MSME, unique with msme_id"
        numeric financial_health_score
        text grade
        numeric probability_of_default
        text risk_band
        boolean eligible
        numeric credit_limit_inr
        boolean thin_file
        text model_version "provenance"
        text policy_version "provenance"
        text rubric_version "provenance"
        text feature_snapshot_hash "reproducibility"
        jsonb card "the card exactly as served"
        timestamptz scored_at
    }
```

## 4. Why the scoring engine is hybrid

```mermaid
flowchart LR
    subgraph problem["The trap"]
        L1["Financial_Health_Score is itself<br/>a weighted formula over 6 pillars"]
        L2["Pure ML on FHS = predicting<br/>a formula. Circular, unexplainable."]
        L1 --> L2
    end

    subgraph answer["The split"]
        R1["Rubric owns the SCORE<br/>transparent, exactly decomposable,<br/>fitted to R² 0.87–0.99"]
        R2["ML owns the RISK<br/>PD and eligibility, where the<br/>signal is genuinely learned"]
        R3["Rules own the DECISION<br/>versioned policy, auditable<br/>knock-outs, sized limits"]
    end

    L2 -.avoided by.-> R1
    R1 --> R3
    R2 --> R3
```

Two independent witnesses reach the same card: the rubric explains the *score*
by arithmetic, the model explains the *risk* by attribution. When they disagree,
that disagreement is itself information and is surfaced
(`credit.model_eligibility_probability` beside the policy decision).
