"""
CREDIT RISK PIPELINE - PHASE 1: EXPLAINABILITY (SHAP)
========================================================
Goal: explain WHY the XGBoost model gave a specific applicant their risk
score, and which features matter most across all applicants.

SHAP (SHapley Additive exPlanations) attributes a prediction to each
input feature: for one applicant, it splits "predicted probability of
default" into a starting point (the average prediction across everyone)
plus each feature's individual push up or down. The pushes add up
exactly to the final prediction, so nothing is left unexplained.

For XGBoost (a tree-based model) we use shap.TreeExplainer, which reads
the actual decision trees and computes each feature's contribution
exactly, rather than approximating it.
"""

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------
# STEP 1: RELOAD THE SAME TRAIN/TEST SPLIT credit_pipeline.py USED
# -----------------------------------------------------------------
# We need a small sample of TRAINING data as a "background" reference
# for SHAP (it asks "compared to a typical applicant, how did this
# feature change the prediction?"). Using the same random_state=42
# split as credit_pipeline.py reproduces the identical split, so this
# lines up exactly with the model that was trained and with
# test_predictions.csv.
col_names = [f"X{i}" for i in range(1, 25)] + ["target_raw"]
df = pd.read_csv("german.data-numeric", sep=r"\s+", header=None, names=col_names)
df["default"] = (df["target_raw"] == 2).astype(int)
df = df.drop(columns=["target_raw"])

X = df.drop(columns=["default"])
y = df["default"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# -----------------------------------------------------------------
# STEP 2: LOAD THE TRAINED XGBOOST MODEL
# -----------------------------------------------------------------
xgb_model = joblib.load("xgb_model.joblib")

# -----------------------------------------------------------------
# STEP 3: BUILD THE SHAP EXPLAINER
# -----------------------------------------------------------------
# model_output="probability" + a background sample makes SHAP express
# every contribution in probability terms (e.g. "+0.15" means "+15
# percentage points of predicted default risk"), which is far more
# intuitive than the model's raw log-odds output.
# We use a random sample of 100 training rows as the background --
# using all 700 would be more precise but much slower, and 100 is
# plenty for a stable reference point.
background = shap.sample(X_train, 100, random_state=42)
explainer = shap.TreeExplainer(
    xgb_model,
    data=background,
    model_output="probability",
    feature_perturbation="interventional",
)

# SHAP values for the whole test set, computed LAZILY (on first actual
# use) rather than eagerly at import time. Only the standalone demo
# below and plot_global_importance() ever need this -- the deployed
# API always calls explain_applicant() with a brand-new applicant's
# dict, never a test-set index, so it never touches this at all. Eagerly
# computing it on every import wasted real memory and startup time on
# every single API container start/restart for something the live API
# path never uses -- worth avoiding on a memory-constrained deployment
# (e.g. a free-tier cloud instance).
_shap_values_test_cache = None


def _get_shap_values_test():
    global _shap_values_test_cache
    if _shap_values_test_cache is None:
        _shap_values_test_cache = explainer.shap_values(X_test)
    return _shap_values_test_cache


# -----------------------------------------------------------------
# STEP 4: EXPLAIN ONE APPLICANT
# -----------------------------------------------------------------
def explain_applicant(row, top_n=5):
    """
    Explain a single applicant's predicted default risk.

    `row` can be:
      - an integer position into the test set (e.g. 0 for the first
        test applicant), or
      - a pandas Series / dict of feature values (X1..X24) for a
        brand-new applicant not in the test set.

    Returns a dict with the predicted probability, the average
    ("base") probability across all applicants, and a plain-English
    list of the top features pushing this applicant's risk up or down.
    """
    if isinstance(row, (int, np.integer)):
        features = X_test.iloc[[row]]
        shap_row = _get_shap_values_test()[row]
    else:
        features = pd.DataFrame([row])[X.columns]
        shap_row = explainer.shap_values(features)[0]

    base_value = explainer.expected_value
    predicted_prob = base_value + shap_row.sum()

    # Rank features by how much they moved the prediction, regardless
    # of direction (a big decrease matters just as much as a big increase).
    order = np.argsort(-np.abs(shap_row))[:top_n]

    explanations = []
    for i in order:
        feature_name = X.columns[i]
        feature_value = features.iloc[0, i]
        impact = shap_row[i]
        direction = "increased" if impact > 0 else "decreased"
        explanations.append(
            f"{feature_name} (value={feature_value:g}) {direction} predicted "
            f"default risk by {abs(impact):.3f}"
        )

    return {
        "base_rate": round(float(base_value), 3),
        "predicted_probability": round(float(predicted_prob), 3),
        "top_features": explanations,
    }


# -----------------------------------------------------------------
# STEP 5: GLOBAL FEATURE IMPORTANCE (across the whole test set)
# -----------------------------------------------------------------
def plot_global_importance(save_path="shap_global_importance.png"):
    """
    Plots the average impact of each feature across ALL test
    applicants (mean absolute SHAP value). This answers a different
    question than explain_applicant(): not "why did THIS person get
    THIS score" but "which features does the model lean on most,
    overall?"
    """
    shap.summary_plot(
        _get_shap_values_test(), X_test, plot_type="bar", show=False
    )
    plt.title("Global Feature Importance (mean |SHAP value|, XGBoost)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved global feature importance plot to {save_path}")


# -----------------------------------------------------------------
# STEP 6: DEMO WHEN RUN DIRECTLY
# -----------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("SHAP EXPLAINABILITY DEMO")
    print("=" * 60)

    example_idx = 0
    result = explain_applicant(example_idx)

    print(f"\nApplicant #{example_idx} (test set):")
    print(f"  Average predicted default rate across all applicants: {result['base_rate']:.1%}")
    print(f"  This applicant's predicted default probability:       {result['predicted_probability']:.1%}")
    print(f"  Actual outcome: {'DEFAULTED' if y_test.iloc[example_idx] == 1 else 'repaid'}")
    print("\n  Top features driving this prediction:")
    for line in result["top_features"]:
        print(f"    - {line}")

    print()
    plot_global_importance()
