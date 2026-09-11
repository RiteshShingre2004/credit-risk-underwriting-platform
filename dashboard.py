"""
CREDIT RISK PIPELINE - PHASE 3: STREAMLIT DASHBOARD
========================================================
Goal: a simple UI to score an applicant and see WHY, without needing to
hand-craft curl commands against the FastAPI backend.

This app is just a client: it doesn't load any models itself. Every
number on screen comes from calling api.py's /score, /explain, and
/metrics endpoints over HTTP (the same way a browser talks to a
website). Run the API first, then this dashboard, in two terminals:

    # terminal 1
    source venv/bin/activate && uvicorn api:app --reload

    # terminal 2
    source venv/bin/activate && streamlit run dashboard.py
"""

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

FEATURE_NAMES = [f"X{i}" for i in range(1, 25)]

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")
st.title("Credit Risk & Underwriting Dashboard")


# -----------------------------------------------------------------
# SIDEBAR: API CONNECTION + DECISION POLICY
# -----------------------------------------------------------------
st.sidebar.header("Settings")
api_url = st.sidebar.text_input("FastAPI base URL", value="http://127.0.0.1:8000")

st.sidebar.subheader("Decision policy")
st.sidebar.caption(
    "These thresholds turn a Probability of Default into a decision. "
    "They're a policy choice, not something the model decides."
)
decision_model = st.sidebar.selectbox(
    "Decision driven by",
    ["xgb_probability_of_default", "logreg_probability_of_default"],
    format_func=lambda x: "XGBoost (calibrated)" if "xgb" in x else "Logistic Regression (calibrated)",
)
approve_below = st.sidebar.slider("APPROVE if PD below", 0.0, 1.0, 0.20, 0.01)
reject_above = st.sidebar.slider("REJECT if PD above", 0.0, 1.0, 0.50, 0.01)
if approve_below > reject_above:
    st.sidebar.error("APPROVE threshold must be below REJECT threshold.")


def decide(pd_value: float) -> tuple[str, str]:
    """Map a PD to (tier, color) using the sidebar thresholds."""
    if pd_value < approve_below:
        return "APPROVE", "green"
    if pd_value > reject_above:
        return "REJECT", "red"
    return "REVIEW", "orange"


# -----------------------------------------------------------------
# LOAD TEST APPLICANTS (so you can pick a real one instead of typing
# 24 numbers from scratch)
# -----------------------------------------------------------------
@st.cache_data
def load_test_applicants():
    return pd.read_csv("test_predictions.csv")


test_df = load_test_applicants()

tab_score, tab_metrics = st.tabs(["Score an Applicant", "Model Performance"])

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
    if st.button("Score this applicant", type="primary"):
        payload = feature_values
        try:
            score_resp = requests.post(f"{api_url}/score", json=payload, timeout=10)
            explain_resp = requests.post(f"{api_url}/explain", json=payload, timeout=10)
            score_resp.raise_for_status()
            explain_resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            st.error(
                "Could not reach the API. Is it running? Start it with:\n\n"
                "`uvicorn api:app --reload`"
            )
        except requests.exceptions.HTTPError as e:
            st.error(f"API returned an error: {e}")
        else:
            scores = score_resp.json()
            explanation = explain_resp.json()

            decision_pd = scores[decision_model]
            tier, color = decide(decision_pd)

            st.markdown("### Result")
            m1, m2, m3 = st.columns(3)
            m1.metric("Logistic Regression PD", f"{scores['logreg_probability_of_default']:.1%}")
            m2.metric("XGBoost PD", f"{scores['xgb_probability_of_default']:.1%}")
            m3.markdown(
                f"<div style='text-align:center'>"
                f"<span style='font-size:0.9rem;color:gray'>Decision</span><br>"
                f"<span style='font-size:1.8rem;font-weight:700;color:{color}'>{tier}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.markdown("### Why (SHAP explanation, XGBoost model)")
            st.caption(
                f"Average predicted default rate across all applicants: "
                f"{explanation['base_rate']:.1%}. This applicant's predicted "
                f"probability from the **raw, pre-calibration** XGBoost model: "
                f"{explanation['predicted_probability']:.1%} — SHAP explains this "
                f"raw model's tree structure, so this number won't exactly match "
                f"the **calibrated** XGBoost PD ({scores['xgb_probability_of_default']:.1%}) "
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
