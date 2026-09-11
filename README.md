# AI-Powered Credit Risk & Underwriting Platform

An explainable credit risk / underwriting platform built as a portfolio
project. Predicts Probability of Default (PD) on the UCI German Credit
dataset, calibrates the predictions, explains individual decisions, and
(eventually) serves them through an API + dashboard with an audit trail.

## Status: Phase 1 in progress (SHAP explainability)

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
and `test_predictions.csv`.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy scikit-learn xgboost shap joblib
```

**macOS note:** XGBoost needs the OpenMP runtime, which isn't part of macOS.
If you see a `libomp.dylib` load error, run `brew install libomp`.

## What's next
- Phase 1: SHAP explainability (`explainability.py`) — in progress
- Phase 2: FastAPI backend (`/score`, `/explain`, `/metrics`)
- Phase 3: Streamlit dashboard
- Phase 4: Persistence + audit trail (SQLite/Postgres)
- Phase 5: Docker
- Phase 6: Drift monitoring (PSI/CSI)
- Phase 7: LLM/RAG policy assistant (design discussion required before
  starting — the LLM must never emit the final decision, only explain and
  fact-check a decision already made by the deterministic model + policy
  engine)
