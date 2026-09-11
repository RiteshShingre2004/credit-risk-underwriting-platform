"""
CREDIT RISK PIPELINE - PHASE 2: FASTAPI BACKEND
========================================================
Goal: expose the trained, calibrated models over HTTP so other programs
(a dashboard, a script, curl) can request a risk score without needing
Python or the model files themselves.

PYDANTIC, IN PLAIN ENGLISH:
Pydantic lets us describe "what a valid request looks like" as a Python
class. Below, ApplicantFeatures says "a valid request has 24 fields,
X1 through X24, and each one must be a number." FastAPI uses that
description automatically: if someone sends a request missing a field,
or sends text where a number belongs, FastAPI rejects it with a clear
error message before our code even runs. We don't write any manual
"if field is missing" checks -- Pydantic does that for us.

Run the server with:
    uvicorn api:app --reload
Then open http://127.0.0.1:8000/docs for interactive, auto-generated
documentation you can test requests against directly in the browser.
"""

import json

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from explainability import explain_applicant

FEATURE_NAMES = [f"X{i}" for i in range(1, 25)]

# -----------------------------------------------------------------
# LOAD EVERYTHING ONCE, AT STARTUP
# -----------------------------------------------------------------
# Loading model files from disk is relatively slow, so we do it once
# when the server starts, not on every request.
scaler = joblib.load("scaler.joblib")
logreg_calibrated = joblib.load("logreg_calibrated.joblib")
xgb_calibrated = joblib.load("xgb_calibrated.joblib")

with open("metrics.json") as f:
    METRICS = json.load(f)

app = FastAPI(
    title="Credit Risk Scoring API",
    description="Serves calibrated Probability of Default (PD) scores, "
    "SHAP-based explanations, and model performance metrics for the "
    "UCI German Credit model.",
)


# -----------------------------------------------------------------
# REQUEST SCHEMA
# -----------------------------------------------------------------
class ApplicantFeatures(BaseModel):
    """One applicant's 24 numeric features, matching the UCI German
    Credit (numeric) dataset's columns X1 through X24."""

    X1: float
    X2: float
    X3: float
    X4: float
    X5: float
    X6: float
    X7: float
    X8: float
    X9: float
    X10: float
    X11: float
    X12: float
    X13: float
    X14: float
    X15: float
    X16: float
    X17: float
    X18: float
    X19: float
    X20: float
    X21: float
    X22: float
    X23: float
    X24: float

    model_config = {
        "json_schema_extra": {
            "example": {f"X{i}": 2.0 for i in range(1, 25)}
        }
    }


def _to_row(applicant: ApplicantFeatures) -> pd.DataFrame:
    """Turn a validated request into a single-row DataFrame with
    columns in the exact order the models were trained on."""
    return pd.DataFrame([applicant.model_dump()])[FEATURE_NAMES]


# -----------------------------------------------------------------
# ENDPOINTS
# -----------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs"}


@app.post("/score")
def score(applicant: ApplicantFeatures):
    """Return calibrated Probability of Default from both models."""
    row = _to_row(applicant)

    # Logistic Regression was trained on SCALED features, so we must
    # scale a new applicant's features the same way before predicting.
    row_scaled = scaler.transform(row)
    lr_pd = float(logreg_calibrated.predict_proba(row_scaled)[0, 1])

    # XGBoost was trained on raw (unscaled) features.
    xgb_pd = float(xgb_calibrated.predict_proba(row)[0, 1])

    return {
        "logreg_probability_of_default": round(lr_pd, 4),
        "xgb_probability_of_default": round(xgb_pd, 4),
    }


@app.post("/explain")
def explain(applicant: ApplicantFeatures):
    """Return a SHAP-based, plain-English explanation of the XGBoost
    model's prediction for this applicant."""
    return explain_applicant(applicant.model_dump())


@app.get("/metrics")
def metrics():
    """Return AUC / Gini / KS for all model variants on the held-out
    test set, computed once during training (see metrics.json)."""
    return METRICS
