"""
CREDIT RISK PIPELINE - PHASE 3+4: STREAMLIT DASHBOARD
========================================================
Goal: a simple UI to score an applicant and see WHY, without needing to
hand-craft curl commands against the FastAPI backend.

This app is just a client: it doesn't load any models itself, and it
doesn't decide anything itself either. Every number and every
APPROVE/REVIEW/REJECT tier on screen comes from calling api.py's
/decide endpoint over HTTP -- the API is the single source of truth
for the decision policy, and every call to /decide is permanently
logged to the audit trail (see database.py). Run the API first, then
this dashboard, in two terminals:

    # terminal 1
    source venv/bin/activate && uvicorn api:app --reload

    # terminal 2
    source venv/bin/activate && streamlit run dashboard.py
"""

import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

FEATURE_NAMES = [f"X{i}" for i in range(1, 25)]
TIER_COLORS = {"APPROVE": "green", "REVIEW": "orange", "REJECT": "red"}
# Locally this defaults to localhost, same as before. In Docker
# (Phase 5), docker-compose.yml sets API_URL to "http://api:8000" --
# containers reach each other by service name, not localhost.
DEFAULT_API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")
st.title("Credit Risk & Underwriting Dashboard")


# -----------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------
st.sidebar.header("Settings")
api_url = st.sidebar.text_input("FastAPI base URL", value=DEFAULT_API_URL)
st.sidebar.caption(
    "The decision policy (which model drives the decision, and the "
    "APPROVE/REJECT thresholds) is enforced server-side by /decide, "
    "not configurable here -- that's what makes every decision "
    "reproducible from the audit trail alone."
)


# -----------------------------------------------------------------
# LOAD TEST APPLICANTS (so you can pick a real one instead of typing
# 24 numbers from scratch)
# -----------------------------------------------------------------
@st.cache_data
def load_test_applicants():
    return pd.read_csv("test_predictions.csv")


test_df = load_test_applicants()

tab_score, tab_metrics, tab_audit = st.tabs(
    ["Score an Applicant", "Model Performance", "Audit Trail"]
)

# -----------------------------------------------------------------
# TAB 1: SCORE AN APPLICANT
# -----------------------------------------------------------------
with tab_score:
    st.subheader("1. Choose or edit an applicant")

    col_pick, col_actual = st.columns([3, 1])
    with col_pick:
        applicant_idx = st.selectbox(
            "Pick a test-set applicant to pre-fill (or just edit the fields below)",
            options=list(range(len(test_df))),
            index=0,
        )
    with col_actual:
        actual = test_df.loc[applicant_idx, "actual_default"]
        st.metric("Actual outcome", "DEFAULTED" if actual == 1 else "Repaid")

    default_row = test_df.loc[applicant_idx, FEATURE_NAMES]

    with st.expander("Applicant features (edit if needed)", expanded=True):
        feature_values = {}
        cols = st.columns(4)
        for i, name in enumerate(FEATURE_NAMES):
            with cols[i % 4]:
                feature_values[name] = st.number_input(
                    name, value=float(default_row[name]), key=f"feat_{name}_{applicant_idx}"
                )

    st.subheader("2. Score")
    st.caption("This calls /decide, which also permanently logs the decision (see the Audit Trail tab).")
    if st.button("Score this applicant", type="primary"):
        payload = feature_values
        try:
            resp = requests.post(f"{api_url}/decide", json=payload, timeout=10)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            st.error(
                "Could not reach the API. Is it running? Start it with:\n\n"
                "`uvicorn api:app --reload`"
            )
        except requests.exceptions.HTTPError as e:
            st.error(f"API returned an error: {e}")
        else:
            result = resp.json()
            explanation = result["explanation"]
            tier = result["decision_tier"]
            color = TIER_COLORS[tier]

            st.markdown("### Result")
            m1, m2, m3 = st.columns(3)
            m1.metric("Logistic Regression PD", f"{result['logreg_probability_of_default']:.1%}")
            m2.metric("XGBoost PD", f"{result['xgb_probability_of_default']:.1%}")
            m3.markdown(
                f"<div style='text-align:center'>"
                f"<span style='font-size:0.9rem;color:gray'>Decision</span><br>"
                f"<span style='font-size:1.8rem;font-weight:700;color:{color}'>{tier}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.caption(
                f"Policy: APPROVE if {result['decision_model']} PD is below "
                f"{result['policy_approve_below']:.0%}, REJECT if above "
                f"{result['policy_reject_above']:.0%}, else REVIEW. "
                f"Logged as audit entry #{result['log_id']}."
            )

            st.markdown("### Why (SHAP explanation, XGBoost model)")
            st.caption(
                f"Average predicted default rate across all applicants: "
                f"{explanation['base_rate']:.1%}. This applicant's predicted "
                f"probability from the **raw, pre-calibration** XGBoost model: "
                f"{explanation['predicted_probability']:.1%} — SHAP explains this "
                f"raw model's tree structure, so this number won't exactly match "
                f"the **calibrated** XGBoost PD ({result['xgb_probability_of_default']:.1%}) "
                f"shown above, which is what actually drives the decision. "
                f"Calibration rescales the probability to match real-world default "
                f"frequency; it doesn't change which features matter or in which "
                f"direction, so the explanation below is still valid."
            )

            # Parse the plain-English strings back into (feature, direction, value)
            # for the chart, and show them as text too.
            parsed = []
            for line in explanation["top_features"]:
                name = line.split(" ")[0]
                increased = "increased" in line
                magnitude = float(line.rsplit("by ", 1)[1])
                parsed.append((name, magnitude if increased else -magnitude))

            chart_col, text_col = st.columns([2, 1])
            with chart_col:
                fig = go.Figure(
                    go.Bar(
                        x=[v for _, v in parsed],
                        y=[n for n, _ in parsed],
                        orientation="h",
                        marker_color=["#d62728" if v > 0 else "#2ca02c" for _, v in parsed],
                    )
                )
                fig.update_layout(
                    title="Top features (red = increases risk, green = decreases risk)",
                    xaxis_title="Impact on predicted default probability",
                    yaxis=dict(autorange="reversed"),
                    height=320,
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig, width="stretch")
            with text_col:
                for line in explanation["top_features"]:
                    st.write(f"- {line}")

# -----------------------------------------------------------------
# TAB 2: MODEL PERFORMANCE
# -----------------------------------------------------------------
with tab_metrics:
    st.subheader("KS and Gini on the held-out test set")
    st.caption(
        "Gini (2*AUC - 1) measures how well the model ranks risky vs. safe "
        "applicants; KS measures, at the best cutoff, the biggest gap "
        "between defaulters caught and good customers wrongly flagged. "
        "Both are computed once during training and served from /metrics."
    )
    try:
        metrics = requests.get(f"{api_url}/metrics", timeout=10).json()
    except requests.exceptions.ConnectionError:
        st.error(
            "Could not reach the API. Is it running? Start it with:\n\n"
            "`uvicorn api:app --reload`"
        )
    else:
        model_keys = ["logreg_raw", "logreg_calibrated", "xgb_raw", "xgb_calibrated"]
        labels = ["LR (raw)", "LR (calibrated)", "XGB (raw)", "XGB (calibrated)"]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Gini", x=labels, y=[metrics[k]["gini"] for k in model_keys]))
        fig.add_trace(go.Bar(name="KS", x=labels, y=[metrics[k]["ks"] for k in model_keys]))
        fig.update_layout(
            barmode="group",
            yaxis_title="Score (higher is better)",
            height=400,
        )
        st.plotly_chart(fig, width="stretch")

        st.dataframe(
            pd.DataFrame(metrics, index=["auc", "gini", "ks", "ks_best_threshold"])[model_keys].T,
            width="stretch",
        )
        st.caption(
            f"Test set: {metrics['test_set_size']} applicants, "
            f"{metrics['test_set_default_rate']:.1%} actual default rate."
        )

# -----------------------------------------------------------------
# TAB 3: AUDIT TRAIL
# -----------------------------------------------------------------
with tab_audit:
    st.subheader("Every decision, permanently logged")
    st.caption(
        "Each row is one call to /decide, stored in credit_risk.db. This "
        "is what a compliance review or a later model-monitoring check "
        "would query."
    )
    limit = st.number_input("Show most recent N decisions", min_value=1, max_value=500, value=25)
    if st.button("Refresh"):
        st.cache_data.clear()

    try:
        decisions = requests.get(f"{api_url}/decisions", params={"limit": limit}, timeout=10).json()
    except requests.exceptions.ConnectionError:
        st.error(
            "Could not reach the API. Is it running? Start it with:\n\n"
            "`uvicorn api:app --reload`"
        )
    else:
        if not decisions:
            st.info("No decisions logged yet -- score an applicant in the first tab.")
        else:
            audit_df = pd.DataFrame(decisions)
            display_cols = [
                "id", "created_at", "decision_tier", "decision_model",
                "logreg_pd", "xgb_pd", "model_version",
            ]
            st.dataframe(audit_df[display_cols], width="stretch")

            with st.expander("Full detail for one entry"):
                selected_id = st.selectbox("Entry id", audit_df["id"].tolist())
                st.json(audit_df[audit_df["id"] == selected_id].iloc[0].to_dict())
