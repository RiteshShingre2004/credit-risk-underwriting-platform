# AI-Powered Credit Risk & Underwriting Platform

An explainable credit risk / underwriting platform built as a portfolio
project. Predicts Probability of Default (PD) on the UCI German Credit
dataset, calibrates the predictions, explains individual decisions, and
(eventually) serves them through an API + dashboard with an audit trail.

## Status: Phase 7 core built (LLM explanation + fact-check) — API/dashboard wiring next

## What's built so far

### Phase 0 — Modeling pipeline (`credit_pipeline.py`)
Trains two models to predict default risk:
- **Logistic Regression** — simple, fully transparent (a weighted formula).
- **XGBoost** — 200 small decision trees voting together; more accurate but
  a "black box" without extra tooling (that's what SHAP is for).

Both models are **calibrated** with Platt scaling (`CalibratedClassifierCV`,
sigmoid method) so their output probabilities reflect real-world default
frequency, not just a ranking score.

**Current held-out test metrics (300 applicants, never seen during training):**

| Model | AUC | Gini | KS |
|---|---|---|---|
| Logistic Regression (raw) | 0.802 | 0.603 | 0.514 |
| Logistic Regression (calibrated) | 0.802 | 0.604 | 0.519 |
| XGBoost (raw) | 0.802 | 0.604 | 0.468 |
| XGBoost (calibrated) | 0.810 | 0.620 | 0.502 |

- **Gini** (`2*AUC - 1`): how well the model ranks risky vs. safe applicants.
  Higher is better.
- **KS** (Kolmogorov-Smirnov): at the best cutoff, the biggest gap between
  % of defaulters caught and % of good customers wrongly flagged. >0.4 is
  considered good in credit risk practice; these models are in the 0.5 range.

Run it with:
```bash
source venv/bin/activate
python3 credit_pipeline.py
```
This regenerates `scaler.joblib`, `logreg_model.joblib`, `xgb_model.joblib`,
`logreg_calibrated.joblib`, `xgb_calibrated.joblib`, `test_predictions.csv`,
and `metrics.json`.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**macOS note:** XGBoost needs the OpenMP runtime, which isn't part of macOS.
If you see a `libomp.dylib` load error, run `brew install libomp`.

### Phase 1 — SHAP explainability (`explainability.py`)
Explains individual XGBoost predictions and overall feature importance
using [SHAP](https://shap.readthedocs.io/) (`TreeExplainer`, interventional
perturbation, probability-space output — so a SHAP value of `0.15` means
"+15 percentage points of predicted default risk," not an abstract score).

Two functions, importable independently of training:
- `explain_applicant(row)` — takes a test-set row index (or a dict/Series
  of raw feature values for a new applicant) and returns the predicted
  probability plus the top features pushing it up or down, e.g.
  `"X2 (value=48) increased predicted default risk by 0.055"`.
- `plot_global_importance()` — saves `shap_global_importance.png`, a bar
  chart of each feature's average impact across all 300 test applicants.
  `X1` (checking-account status) dominates, which matches the German
  Credit dataset's well-known top predictor.

Run the demo with:
```bash
source venv/bin/activate
python3 explainability.py
```

**Note:** SHAP explains the *raw* (uncalibrated) XGBoost model — that's
fine since calibration only rescales the output, it doesn't change the
tree structure. `logreg_calibrated.joblib` and `xgb_calibrated.joblib`
(saved by `credit_pipeline.py`) are what Phase 2's `/score` endpoint uses.

**macOS/XGBoost quirk fixed here:** newer XGBoost versions default
`enable_categorical=True` internally even when no feature is categorical,
which made SHAP wrongly refuse to run (`NotImplementedError: Categorical
split is not yet supported`). Fixed by passing `enable_categorical=False`
explicitly in both `XGBClassifier(...)` calls in `credit_pipeline.py`.

### Phase 2 — FastAPI backend (`api.py`)
Wraps the calibrated models in a small HTTP API so other programs (a
dashboard, a script, curl) can request a score without touching Python
or the model files directly.

Request/response validation uses **Pydantic**: `ApplicantFeatures`
describes a valid request (24 numeric fields, `X1`-`X24`). FastAPI checks
every incoming request against it automatically — a missing field or a
wrong type gets rejected with a `422` error before our code runs, no
manual `if` checks needed.

Endpoints:
- `POST /score` — calibrated Probability of Default from both LR and
  XGBoost. LR's input is scaled first (`scaler.joblib`, same as
  training); XGBoost takes raw features.
- `POST /explain` — same request shape, returns the Phase 1 SHAP
  explanation (reuses `explain_applicant()` directly, no duplicated logic).
- `GET /metrics` — returns `metrics.json` (AUC/Gini/KS for all four
  model variants, computed once during training).
- `GET /` — health check.

Run it with:
```bash
source venv/bin/activate
uvicorn api:app --reload
```
Then open `http://127.0.0.1:8000/docs` for interactive, auto-generated
API docs you can test requests against in the browser.

### Phase 3 — Streamlit dashboard (`dashboard.py`)
A UI on top of the API — it holds no models itself and decides nothing
itself either, every number and every decision tier comes from HTTP
calls to `api.py` (see Phase 4 below for how the decision policy works).
Three tabs:
- **Score an Applicant** — pick a test-set applicant (auto-fills all 24
  features, editable) or type your own, hit "Score", and see calibrated
  PD from both models, the APPROVE/REVIEW/REJECT decision, and the SHAP
  explanation as a colored bar chart (red = increases risk, green =
  decreases risk) plus the plain-English sentences from Phase 1.
- **Model Performance** — a grouped bar chart + table of AUC/Gini/KS for
  all four model variants, from `/metrics`.
- **Audit Trail** — every logged decision, from `/decisions` (Phase 4).

Run both pieces (two terminals):
```bash
# terminal 1
source venv/bin/activate && uvicorn api:app --reload

# terminal 2
source venv/bin/activate && streamlit run dashboard.py
```
Then open `http://localhost:8501`.

**Tested with:** Playwright driving a real headless Chrome against the
running app — picked an applicant, clicked "Score", confirmed the PD
values, decision tier, and SHAP chart all rendered correctly, and
checked the "Model Performance" tab. One real bug caught this way: the
SHAP section showed a different probability (2.7%) than the score card
above it (12.1%) — because SHAP explains the *raw* XGBoost model and
`/score` returns the *calibrated* one. Fixed by adding an explicit note
in the UI rather than hiding the (legitimate) discrepancy.

### Phase 4 — Persistence + audit trail (`database.py`, `api.py`'s `/decide`)
Every real underwriting decision now gets permanently logged to SQLite
(`credit_risk.db`, gitignored — it's runtime data, not source).

**The decision policy moved server-side.** Previously the dashboard had
its own APPROVE/REJECT threshold sliders; now `api.py` is the single
source of truth (`DECISION_MODEL`, `POLICY_APPROVE_BELOW`,
`POLICY_REJECT_ABOVE`), so the decision is consistent and reproducible
regardless of who's looking at it. This is also, deliberately, the
"deterministic policy engine" pattern the LLM phase (Phase 7) will need
to plug into later.

**`POST /decide`** — the audited decision path: scores both models,
applies the policy to get a tier, generates the SHAP explanation, logs
one row, and returns everything (including the new log's id).
`/score` and `/explain` are unchanged and stay non-logged, for quick
testing. **`GET /decisions`** returns the most recent logged decisions.

**`scoring_log` table** — one row per decision: `id`, `created_at`,
`X1`-`X24` (as individual queryable columns, not a JSON blob),
`logreg_pd`, `xgb_pd`, `model_version`, `calibration_method`,
`decision_model`, `decision_tier`, `policy_approve_below`,
`policy_reject_above`, `shap_top_features` (JSON — this one's naturally
a small nested list).

The dashboard's "Score an Applicant" tab now calls `/decide` instead of
separately calling `/score` + `/explain`, and a new **Audit Trail** tab
lists logged decisions from `/decisions`, with a detail view per entry.

Verified with curl (checked `/decide` writes a row, `/decisions` returns
it) and independently with the `sqlite3` CLI directly against the
database file, plus Playwright driving the dashboard end-to-end.

### Phase 5 — Docker (`Dockerfile.api`, `Dockerfile.dashboard`, `docker-compose.yml`)
Packages the API and dashboard as containers so the whole stack runs
identically on any machine with Docker — no manually installed Python,
no `libomp` step, no "works on my laptop."

- **`Dockerfile.api`** — Python 3.14-slim + `libgomp1` (Linux's equivalent
  of macOS's `libomp`, which XGBoost needs) + `requirements.txt` +
  the code and model artifacts. Runs `uvicorn --host 0.0.0.0` (required
  in Docker — `127.0.0.1` would only accept connections from inside the
  container itself).
- **`Dockerfile.dashboard`** — same base image, no `libgomp1` needed since
  the dashboard never imports XGBoost/SHAP directly, only calls the API
  over HTTP.
- **`docker-compose.yml`** — wires both containers onto a private network.
  The dashboard reaches the API at `http://api:8000` (containers address
  each other by service name, not `localhost`) via the `API_URL`
  environment variable; locally (no Docker) it still defaults to
  `http://127.0.0.1:8000`, unchanged.
- **Audit trail survives restarts** — `credit_risk.db` is written to a
  named Docker volume (`credit_risk_data`), not the container's own
  throwaway filesystem, controlled by the `CREDIT_RISK_DB_PATH`
  environment variable (falls back to a local file outside Docker).

Run the whole stack with:
```bash
docker compose up --build
```
Then open `http://localhost:8501` (dashboard) or `http://localhost:8000/docs`
(API). Stop with `docker compose down` (add `-v` to also delete the audit
trail volume).

**Verified:** built both images, brought the stack up, confirmed `/`,
`/metrics`, and `/decide` all work through the containerized API; called
`/decide` then restarted the API container and confirmed the logged
decision was still in `/decisions` (proves the volume mount actually
persists data, not just that the container runs); confirmed the dashboard
container can reach the API container by its service name (`api:8000`),
not just `localhost`.

### Phase 6 — Drift monitoring (`drift_monitoring.py`)
A script that flags when a new batch of applicants looks statistically
different from the applicants the model was trained on — using the
**Population Stability Index (PSI)**, per feature, computed without
needing any outcome labels (you only find out who defaulted months
later, but you can check "do these applicants even look similar?" the
moment they arrive).

**Method:** bucket each feature into 10 deciles using *training-data*
cutoffs (fixed reference), compare what % of the training population
fell in each bucket vs. what % of the new batch does, and sum
`(new% - train%) × ln(new% / train%)` across buckets. Standard
thresholds: `<0.10` stable, `0.10–0.25` **WATCH**, `≥0.25` **INVESTIGATE**.

Two functions:
- `compute_psi(train_values, new_values)` — PSI for one feature.
- `psi_report(new_batch)` — PSI + status for all 24 features, sorted
  worst-first.

Run the demo with:
```bash
source venv/bin/activate
python3 drift_monitoring.py
```
It runs two checks and saves `psi_report.png` (a bar chart of the
second one):
1. **Real held-out test set vs. training data** — same underlying
   population, so PSI stays ~0 everywhere (0 of 24 features flagged).
   This is the "no false alarms" sanity check.
2. **Synthetically drifted batch** — credit amount (`X5`) inflated 60%,
   loan duration (`X2`) inflated 40%, to prove the alert actually fires
   on a real shift. Result: `X2` → PSI 0.65 (INVESTIGATE), `X5` → PSI
   0.11 (WATCH), everything else stays stable — exactly the two
   tampered features, nothing else.

### Phase 7 — LLM explanation + fact-check (`policy_docs/`, `policy_retrieval.py`, `llm_explainer.py`)
**The hard constraint:** the LLM never decides anything. `api.py`'s
`apply_policy()` (fixed PD thresholds) is the only thing that ever
produces APPROVE/REVIEW/REJECT. The LLM's only job here is to explain a
decision that's already final, and a *second, independent* LLM call
fact-checks that explanation before it's trusted.

- **`policy_docs/underwriting_policy.md`** — a synthetic underwriting
  policy (there's no real bank policy to use for a portfolio project):
  decision tiers matching `api.py`'s actual thresholds, which model
  drives the decision, SHAP explanation guidance, adverse-action reason
  categories, fair lending / prohibited factors, and an explicit section
  addressed to the automated explanation assistant.
- **`policy_retrieval.py`** — chunks the policy doc by section, retrieves
  the most relevant sections for a query via TF-IDF + cosine similarity
  (`scikit-learn`, no new heavy dependency — appropriate since the corpus
  is one short document, not a large collection). Verified: 7 sections
  indexed, 4 test queries each correctly retrieved their relevant section.
- **`llm_explainer.py`** — two separate LLM calls, on purpose:
  - `generate_explanation()` writes a plain-language explanation from
    the decision facts + retrieved policy text.
  - `verify_explanation()` is a **second, independent** call that
    re-checks the draft against only the same facts and policy text,
    and reports any unsupported claim. It never sees the first call's
    reasoning, only its output.
  - If verification fails, `explain_decision()` regenerates once with
    the specific issues fed back, and always returns an honest
    `verified` flag — a narrative that failed verification is still
    returned, just labeled as such, never silently hidden.

**Runs on Groq's free API** serving OpenAI's open-weight `gpt-oss`
models (Apache 2.0 licensed) — `gpt-oss-120b` for generation,
`gpt-oss-20b` for verification (a grading/checklist task doesn't need
the biggest model, same reasoning as using a cheaper model for an LLM
judge anywhere else). This started as a local-only plan via Ollama, but
large model downloads (~5GB) kept failing on this network; Groq avoids
that entirely — no local download, no GPU needed, free tier is
rate-limited rather than metered for this volume of use. Get a free key
at [console.groq.com/keys](https://console.groq.com/keys) and:
```bash
export GROQ_API_KEY=gsk_...
source venv/bin/activate
python3 llm_explainer.py
```

**A real bug caught and fixed during testing:** the decision JSON handed
to the LLM originally contained two different "probability"-shaped
fields — `xgb_probability_of_default` (the *calibrated* PD that actually
drives the decision) and `explanation.predicted_probability` (SHAP's own
figure, from the *raw, uncalibrated* model — see the Phase 1 caveat
above). The LLM picked the wrong one and stated it as "the" probability
of default (applicant #54: said 0.908, the real decision PD was 0.760) —
and the verifier didn't catch it either, since the number technically
did appear somewhere in the facts, just under the wrong claim. Fixed by
building a `_decision_facts_for_prompt()` helper that exposes only one,
unambiguous `probability_of_default` field to both LLM calls, rather
than trying to prompt-engineer around two similarly-named numbers.
Re-tested after the fix: both a high-risk (REJECT) and low-risk
(APPROVE) applicant now cite the correct PD and verify as grounded on
the first attempt.

**Still to do:** wire this into `api.py` as a new endpoint (e.g.
`POST /narrate`, taking a `log_id` from an existing `/decide` call),
log the narrative + verification result to the audit trail, and
surface it in the dashboard.

## What's next
- Finish Phase 7: wire `llm_explainer.py` into `api.py` + the dashboard
- Phase 8 (not in the original plan, worth considering): a short writeup
  / architecture diagram summarizing the whole project for a portfolio
  audience
