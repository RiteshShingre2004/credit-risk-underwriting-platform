# AI-Powered Credit Risk & Underwriting Platform

An explainable credit risk / underwriting platform built as a portfolio
project. Predicts Probability of Default (PD) on the UCI German Credit
dataset, calibrates the predictions, explains individual decisions, and
(eventually) serves them through an API + dashboard with an audit trail.

## Status: Phase 3 complete (Streamlit dashboard)

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
A UI on top of the API — it holds no models itself, every number comes
from HTTP calls to `api.py`. Two tabs:
- **Score an Applicant** — pick a test-set applicant (auto-fills all 24
  features, editable) or type your own, hit "Score", and see:
  - Calibrated PD from both models.
  - A decision tier (APPROVE / REVIEW / REJECT) driven by sidebar
    thresholds — the thresholds are a policy choice you can drag around,
    not something the model outputs.
  - The SHAP explanation as a colored bar chart (red = increases risk,
    green = decreases risk) plus the plain-English sentences from Phase 1.
- **Model Performance** — a grouped bar chart + table of AUC/Gini/KS for
  all four model variants, from `/metrics`.

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

## What's next
- Phase 4: Persistence + audit trail (SQLite/Postgres)
- Phase 5: Docker
- Phase 6: Drift monitoring (PSI/CSI)
- Phase 7: LLM/RAG policy assistant (design discussion required before
  starting — the LLM must never emit the final decision, only explain and
  fact-check a decision already made by the deterministic model + policy
  engine)
