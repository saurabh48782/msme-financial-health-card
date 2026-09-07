# Model card — MSME Financial Health Card

Numbers below are from model version `18`, policy `policy-1.0.0`, rubric
`rubric-1.0.0`, scored over all 50,000 firms on 2026-09-04. The machine-readable
source is `data/eval/reports/latest.json`; regenerate it with
`python -m src.evaluation report`.

This file is written and reviewed by hand rather than generated: a model card's
value is in the judgement about what the numbers *mean*, and a generator can only
restate them.

## What this system does

Consumes 33 consent-based alternative-data fields (GST, UPI, bank/AA, EPFO,
invoices, firmographics) for an Indian MSME and emits:

* a 0–100 **Financial Health Score** with six pillar sub-scores, exactly decomposable
* a **probability of default** and a risk band
* an **eligibility decision** with machine-readable knock-out reasons
* a **credit limit**, tenor and indicative rate
* **typed risk flags** and plain-English **reason codes** (rubric + SHAP)

Intended for **New-to-Credit and New-to-Bank MSMEs**, where bureau scores, audited
financials and collateral records do not exist.

## Intended use and misuse

**In scope.** Pre-screening and limit-setting for unsecured MSME working capital,
where the lender holds valid Account Aggregator consent for the data used.
Portfolio monitoring. Adverse-action reason generation.

**Out of scope.** Underwriting real credit before retraining on observed
repayment (see **Limitations**). Substituting for a bureau on firms that do have a
credit record. Fraud adjudication — the anomaly layer orders a review queue, it
does not decide.

## Architecture in one line

The rubric owns the score (transparent, exact), XGBoost owns the risk (PD and
eligibility), and a versioned rule engine owns the decision. Pure ML on the health
score would be circular: the label is itself a weighted formula over the pillars.

## Data

| Property | Value |
|---|---|
| Source | `msme_synthetic_50k.csv` |
| Rows | 50,000 (one snapshot per MSME) |
| Columns | 45 = 1 identifier + 33 inputs + 11 supervision targets |
| Segment mix | NTC 21,068 · NTB 14,967 · Existing-to-Credit 13,965 |
| Structural nulls | `EMI_On_Time_Rate_Pct` is null for all 32,556 firms with no existing loan |
| Split | 80/20, stratified on `Customer_Segment` (40,000 train / 10,000 test) |
| Model features | 33 raw inputs only — derived ratios are ablated out (they cost accuracy); label-block columns blocked by allowlist |

**Structural nulls are never imputed.** Two thirds of the portfolio has no
repayment record. Filling it with a mean would fabricate a credit history for
exactly the New-to-Credit firms this system exists to serve. Instead the driver's
rubric weight is redistributed across observed evidence, XGBoost handles the raw NaN
natively, `has_repayment_history` names the state for the rubric and the card, and the
card reports `thin_file: true` out loud.

## Metrics — full portfolio (50,000 firms)

| Metric | Value | Gate |
|---|---|---|
| Financial Health Score adjusted R² | 0.8148 | — |
| Financial Health Score MAE | 2.150 points | ≤ 2.5 |
| PD adjusted R² | 0.9554 | — |
| PD MAE | 0.01424 | ≤ 0.025 |
| PD expected calibration error | 0.00104 | — |
| Eligibility ROC-AUC | 0.9886 | ≥ 0.95 |
| Eligibility KS | 0.8831 | — |
| Risk-band agreement | 0.8645 | — |
| Eligibility agreement | 0.9329 | — |
| Credit limit MAPE | 0.1189 | ≤ 0.15 |

### Holdout metrics (20% test fold, 10,000 firms)

| Head | Metric | Value |
|---|---|---|
| PD regressor | adjusted R² | 0.9053 |
| PD regressor | MAE | 0.02045 |
| PD regressor | calibration bias | −0.00036 |
| Eligibility classifier | ROC-AUC | 0.9776 |
| Eligibility classifier | PR-AUC | 0.9923 |
| Eligibility classifier | KS | 0.8317 |
| Eligibility classifier | precision / recall | 0.947 / 0.954 |

**On PD calibration.** The PD is trustworthy as a number, not only as a ranking.
The head is a regressor on a continuous target, already trained to minimise squared
error against the quantity it reports, and it measures as well calibrated: expected
calibration error 0.00104 and bias −0.00007 on the full portfolio, against a mean PD
of 0.1187. Calibration is measured and asserted — bias is gated per cohort of every
slice column by `pytest -m eval` — rather than corrected after the fact.

### Pillar rubric fit

Driver weights are fitted by non-negative least squares against the dataset's own
pillar column, with a 3% floor so no configured driver is dropped (see
**Governance**). Non-negativity is a fairness property, not a fitting convenience:
it makes it impossible for better GST filing to *lower* a Compliance score.

Every R² reported here is **adjusted** — `1 - (1 - R²)(n - 1)/(n - p - 1)`, with
*p* the number of fitted drivers behind the pillar. Plain R² rises with predictor
count whether or not a driver carries signal, so a rubric could otherwise clear
the gate simply by being wider than its neighbours. At 50,000 rows against ≤ 6
drivers the penalty is ~1e-4, which is exactly the point: the correction is free
to apply and it removes the objection.

| Pillar | Adjusted R² | Drivers (*p*) | MAE | Gate |
|---|---|---|---|---|
| Compliance | 0.9941 | 4 | 0.370 | ≥ 0.75 |
| Cash Flow | 0.9903 | 5 | 0.506 | ≥ 0.75 |
| Business Growth | 0.9423 | 3 | 2.274 | ≥ 0.75 |
| Payment Behaviour | 0.9418 | 4 | 1.065 | ≥ 0.75 |
| Revenue Consistency | 0.8986 | 4 | 2.035 | ≥ 0.75 |
| Business Stability | 0.8755 | 5 | 2.216 | ≥ 0.75 |

Payment Behaviour fits at 0.997 on fully observed rows but 0.942 across the whole
portfolio. The gap is the structural null: two thirds of firms have no EMI record,
so the rubric scores them by redistributing that weight onto observed evidence.
The lower number is the one to trust, because it is the one the portfolio sees.

### Top PD drivers (gain-normalised)

| Rank | Feature | Share of gain |
|---|---|---|
| 1 | `GST_Filing_Timeliness_Pct` | 0.3446 |
| 2 | `Cashflow_Stability_Index` | 0.1939 |
| 3 | `Overdraft_Usage_Ratio` | 0.1522 |
| 4 | `Salary_Consistency_Pct` | 0.0776 |
| 5 | `Transaction_Volatility_Index` | 0.0634 |
| 6 | `Avg_Invoice_Payment_Delay_Days` | 0.0497 |
| 7 | `Revenue_Growth_Rate_Pct` | 0.0299 |
| 8 | `Vendor_Payment_Timeliness_Pct` | 0.0281 |
| 9 | `Customer_Concentration_Ratio` | 0.0238 |
| 10 | `Years_in_Operation` | 0.0093 |

Compliance behaviour dominates, and no bureau-style variable appears anywhere in
the list. That is the headline claim of the project, and it is measured rather
than asserted.

## Fairness

Deviation is measured on model **error**, not on outcome rates. Cohorts may
legitimately differ in creditworthiness; the model's error must not, or the system
is transferring its own uncertainty onto one group of borrowers.

| Segment | Firms | PD MAE | PD bias | Approval (ours) | Approval (dataset) |
|---|---|---|---|---|---|
| Existing-to-Credit | 13,965 | 0.01331 | −0.00004 | 0.741 | 0.776 |
| NTB | 14,967 | 0.01439 | −0.00010 | 0.724 | 0.737 |
| NTC | 21,068 | 0.01476 | −0.00007 | 0.717 | 0.726 |

Largest PD-error deviation across **all** slices (segment, location, industry):
**0.0930 points**, against a gate of 3.0.

**The inclusion result.** Approval rates span 71.7%–74.1% across the three
segments — a spread of 2.4 points, *narrower* than the reference decisions' own
5.0-point spread. A firm with no credit history is not penalised for the absence
of one.

Every claim in this section is asserted by `pytest -m eval`
(`tests/integration/test_metrics_regression.py`). It is a failing test, not a
paragraph.

## Limitations

1. **Synthetic data optimism.** The generator's own pillar formula explains only
   ~83% of its health score, so an FHS MAE near 2.1 points is close to the
   achievable ceiling rather than evidence of a great model. Real data will be
   messier.
2. **PD is a generated label.** Every risk number inherits the generator's
   assumptions. Retraining on observed default is mandatory before real use, and
   the rubric weights should be expected to move when it happens.
3. **No time series.** The dataset carries monthly aggregates only, so trend,
   momentum and within-year volatility are unobservable. `Revenue_Growth_Rate_Pct`,
   `Seasonality_Index` and `Transaction_Volatility_Index` are the only shape
   information available, and the card shows no trend charts rather than generating
   a history the data does not contain.
4. **Knock-out rules are unvalidated policy.** Their thresholds sit at the observed
   floor of the eligible population, so the engine agrees with the reference
   decisions out of the box. That is a fitting choice as much as a policy one, and
   in production the thresholds should come from credit policy, not from the data.
5. **Limit sizing diverges from the reference on 1.7% of firms.** The DSCR headroom
   cap binds on 3,361 firms (6.7%): 2,423 receive a smaller limit than the turnover
   formula would give, and 863 (1.7%) are declined outright because their free
   cashflow cannot service even the ₹20,000 minimum ticket. The reference decisions
   ignore serviceability; we do not. The divergence is deliberate, coded as
   `below_min_ticket_dscr`, and is the main reason eligibility agreement is 0.933
   rather than near-perfect.
6. **The rubric weight floor is judgement, not evidence.** A pure fit drives nine
   drivers below 0.005 — including the DSCR proxy, salary consistency and
   the GST-bank reconciliation gap — and suppresses the cash buffer to 0.011,
   because this generator does not use them. Holding each at 3% asserts that a
   real lender would need them. That is a defensible belief, not a measured fact,
   and it is the one place in the rubric where a belief overrides the data.
7. **Anomaly detection is rules-only.** Eight named consistency rules, no
   unsupervised detector. Shapes nobody wrote a rule for are not caught; see
   [BEYOND_SCOPE.md](BEYOND_SCOPE.md) for why that trade was made deliberately.

## Governance

| Control | Mechanism |
|---|---|
| Leakage | Feature allowlist in code, `validate_config`, and a dedicated unit-test module |
| Reproducibility | `feature_snapshot_hash` on every card; DVC-tracked data and reports |
| Version provenance | `model_version` / `policy_version` / `rubric_version` stored per card |
| Rubric changes | Calibration emits a paste-ready YAML block; it never overwrites `params.yaml` |
| Model lineage | Every run logged to MLflow with params, metrics, plots and a `params.yaml` snapshot |
| Metric regression | `pytest -m eval` fails the build on any threshold breach |
| Borrower privacy | PII redaction at the error-log boundary and in stage tracing; `detect-secrets` in the gate |

Deliberately absent: a staging→production alias lifecycle. There is no second
environment to promote into, so it would have been ceremony. What matters — which
model produced a given card — is on the card itself. See
[BEYOND_SCOPE.md](BEYOND_SCOPE.md).
