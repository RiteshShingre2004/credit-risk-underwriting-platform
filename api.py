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
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import database
import llm_explainer
from explainability import explain_applicant

FEATURE_NAMES = [f"X{i}" for i in range(1, 25)]

# -----------------------------------------------------------------
# DECISION POLICY (the "deterministic policy engine")
# -----------------------------------------------------------------
# This is the one, single-source-of-truth definition of how a
# probability turns into a decision. The dashboard used to have its own
# copy of thresholds -- now it just displays whatever the API decides,
# so two people looking at the same applicant always see the same
# answer, and every decision is reproducible from the logged policy
# values alone.
DECISION_MODEL = "xgb_calibrated"  # which model's PD drives the decision
POLICY_APPROVE_BELOW = 0.20
POLICY_REJECT_ABOVE = 0.50


def apply_policy(pd_value: float) -> str:
    if pd_value < POLICY_APPROVE_BELOW:
        return "APPROVE"
    if pd_value > POLICY_REJECT_ABOVE:
        return "REJECT"
    return "REVIEW"


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

database.init_db()

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


@app.post("/decide")
def decide(applicant: ApplicantFeatures):
    """
    The audited decision path: score both models, apply the fixed
    policy thresholds to get a tier, explain the decision with SHAP,
    and permanently log all of it to SQLite. This is what a real
    underwriting decision -- as opposed to a what-if score -- should
    go through.
    """
    features = applicant.model_dump()
    row = _to_row(applicant)

    row_scaled = scaler.transform(row)
    lr_pd = float(logreg_calibrated.predict_proba(row_scaled)[0, 1])
    xgb_pd = float(xgb_calibrated.predict_proba(row)[0, 1])

    pds = {
        "logreg_probability_of_default": lr_pd,
        "xgb_probability_of_default": xgb_pd,
    }
    decision_pd = pds["xgb_probability_of_default"] if DECISION_MODEL == "xgb_calibrated" else pds["logreg_probability_of_default"]
    tier = apply_policy(decision_pd)

    explanation = explain_applicant(features)

    log_id = database.log_decision(
        features=features,
        logreg_pd=lr_pd,
        xgb_pd=xgb_pd,
        model_version=METRICS["model_version"],
        calibration_method=METRICS["calibration_method"],
        decision_model=DECISION_MODEL,
        decision_tier=tier,
        policy_approve_below=POLICY_APPROVE_BELOW,
        policy_reject_above=POLICY_REJECT_ABOVE,
        shap_top_features=explanation["top_features"],
    )

    return {
        "log_id": log_id,
        "logreg_probability_of_default": round(lr_pd, 4),
        "xgb_probability_of_default": round(xgb_pd, 4),
        "decision_model": DECISION_MODEL,
        "decision_tier": tier,
        "policy_approve_below": POLICY_APPROVE_BELOW,
        "policy_reject_above": POLICY_REJECT_ABOVE,
        "explanation": explanation,
    }


@app.get("/decisions")
def decisions(limit: int = 50):
    """Return the most recently logged decisions, newest first --
    the audit trail."""
    return database.get_recent_decisions(limit=limit)


@app.post("/narrate/{log_id}")
def narrate(log_id: int):
    """
    Phase 7: generates a plain-language, policy-grounded explanation
    for an ALREADY-LOGGED decision, and fact-checks it before saving.
    This never re-decides anything -- decision_tier and the PD that
    drove it are read back exactly as /decide logged them and handed
    to the LLM as fixed facts it cannot change (see llm_explainer.py's
    hard constraint). Narratives are generated on demand rather than
    automatically at /decide time, so scoring stays fast even when the
    LLM step is slow or (if GROQ_API_KEY isn't set) unavailable.
    """
    row = database.get_decision_by_id(log_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No logged decision with id {log_id}")

    decision = {
        "decision_tier": row["decision_tier"],
        "xgb_probability_of_default": row["xgb_pd"],
        "policy_approve_below": row["policy_approve_below"],
        "policy_reject_above": row["policy_reject_above"],
        "explanation": {"top_features": json.loads(row["shap_top_features"])},
    }

    try:
        result = llm_explainer.explain_decision(decision)
    except llm_explainer.LLMServiceError as e:
        # Covers a missing GROQ_API_KEY, a rate limit, or any other
        # upstream failure reaching Groq -- one clear 503 (service
        # temporarily unavailable) instead of each different failure
        # mode surfacing as an unexplained 500.
        raise HTTPException(status_code=503, detail=str(e))

    database.save_narrative(
        log_id=log_id,
        narrative=result["narrative"],
        verified=result["verified"],
        issues=result["verification_issues"],
    )

    return {"log_id": log_id, **result}
