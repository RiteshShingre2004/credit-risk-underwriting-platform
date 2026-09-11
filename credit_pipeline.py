"""
CREDIT RISK PIPELINE - STEP 1
==============================
Goal: predict Probability of Default (PD), calibrate it properly,
and score how good the model is using KS and Gini (the metrics
banks/regulators actually look at).

Dataset: UCI German Credit Data (numeric version, 1000 applicants, 24 features)
Target: 1 = Good credit risk (did NOT default), 2 = Bad credit risk (defaulted)
We flip this so 1 = DEFAULT (bad), 0 = NO DEFAULT (good) -- this is the
industry-standard convention for a "Probability of Default" model.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier

# -----------------------------------------------------------------
# STEP 1: LOAD THE DATA
# -----------------------------------------------------------------
# The file has 25 columns separated by whitespace: 24 features + 1 label.
# There are no column headers, so we name them generically (X1, X2, ...)
col_names = [f"X{i}" for i in range(1, 25)] + ["target_raw"]
df = pd.read_csv("german.data-numeric", sep=r"\s+", header=None, names=col_names)

# Original labels: 1 = good (repaid), 2 = bad (defaulted)
# We convert to: 1 = defaulted (what we want to PREDICT), 0 = repaid
df["default"] = (df["target_raw"] == 2).astype(int)
df = df.drop(columns=["target_raw"])

print("=" * 60)
print("STEP 1: DATA LOADED")
print("=" * 60)
print(f"Number of applicants: {len(df)}")
print(f"Number of features: {df.shape[1] - 1}")
print(f"Default rate (bad credit %): {df['default'].mean():.1%}")
print()

# -----------------------------------------------------------------
# STEP 2: TRAIN / TEST SPLIT
# -----------------------------------------------------------------
# We hold out 30% of applicants that the model NEVER sees during training.
# This is how we honestly check "would this model work on new applicants?"
X = df.drop(columns=["default"])
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

print("=" * 60)
print("STEP 2: TRAIN/TEST SPLIT")
print("=" * 60)
print(f"Training applicants: {len(X_train)}")
print(f"Test applicants (never seen by model): {len(X_test)}")
print()

# -----------------------------------------------------------------
# STEP 3: TRAIN LOGISTIC REGRESSION (the simple, transparent model)
# -----------------------------------------------------------------
# Logistic Regression needs features on the same numeric scale to
# behave well, so we standardize (mean 0, std 1) first.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)

lr_probs_raw = log_reg.predict_proba(X_test_scaled)[:, 1]

print("=" * 60)
print("STEP 3: LOGISTIC REGRESSION TRAINED")
print("=" * 60)
print("This model learned a simple weighted formula: each feature gets")
print("a weight, and the weighted sum turns into a probability of default.")
print()

# -----------------------------------------------------------------
# STEP 4: TRAIN XGBOOST (the more powerful, less transparent model)
# -----------------------------------------------------------------
# XGBoost builds hundreds of small decision trees, each one correcting
# the mistakes of the ones before it. It can capture patterns Logistic
# Regression can't (like "this only matters for young applicants"),
# but it's a black box unless you add SHAP (which comes in step 6+).
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    eval_metric="logloss",
    random_state=42,
    enable_categorical=False,  # our features are all plain numeric; being
    # explicit here avoids a version quirk where XGBoost's default flips
    # this on, which then confuses SHAP into thinking there are
    # categorical splits to worry about.
)
xgb.fit(X_train, y_train)  # XGBoost does not need scaling
xgb_probs_raw = xgb.predict_proba(X_test)[:, 1]

print("=" * 60)
print("STEP 4: XGBOOST TRAINED")
print("=" * 60)
print("This model built 200 small decision trees that vote together.")
print()

# -----------------------------------------------------------------
# STEP 5: CALIBRATION
# -----------------------------------------------------------------
# WHY THIS MATTERS: a model can rank applicants correctly (riskier
# people get higher scores) while still being "wrong" about the actual
# NUMBER. E.g. it might say "80% chance of default" for someone who
# actually only defaults 50% of the time historically. For a bank,
# that 50% vs 80% difference changes real decisions (loss reserves,
# interest rates), so we RECALIBRATE the raw probabilities to match
# reality. We use Platt scaling (sigmoid) here since our dataset is
# small; isotonic regression is an alternative for larger datasets.
lr_calibrated = CalibratedClassifierCV(
    LogisticRegression(max_iter=1000, random_state=42), method="sigmoid", cv=5
)
lr_calibrated.fit(X_train_scaled, y_train)
lr_probs_cal = lr_calibrated.predict_proba(X_test_scaled)[:, 1]

xgb_calibrated = CalibratedClassifierCV(
    XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                  eval_metric="logloss", random_state=42,
                  enable_categorical=False),
    method="sigmoid", cv=5
)
xgb_calibrated.fit(X_train, y_train)
xgb_probs_cal = xgb_calibrated.predict_proba(X_test)[:, 1]

print("=" * 60)
print("STEP 5: PROBABILITIES CALIBRATED")
print("=" * 60)
print("Both models now output probabilities that should better match")
print("real-world default frequency, not just a ranking score.")
print()

# -----------------------------------------------------------------
# STEP 6: THE METRICS BANKS ACTUALLY CARE ABOUT (KS and Gini)
# -----------------------------------------------------------------
def gini_coefficient(y_true, y_prob):
    """
    Gini = 2 * AUC - 1
    AUC (Area Under the ROC Curve) asks: "if I pick one defaulter and
    one non-defaulter at random, how often does the model correctly
    give the defaulter a higher risk score?" AUC of 0.5 = random
    guessing. AUC of 1.0 = perfect separation.
    Gini just rescales that to a 0-1 (or -1 to 1) range that's the
    traditional credit-industry convention. Higher Gini = better.
    """
    auc = roc_auc_score(y_true, y_prob)
    return 2 * auc - 1, auc

def ks_statistic(y_true, y_prob):
    """
    KS (Kolmogorov-Smirnov) asks a different question: "at the BEST
    possible cutoff score, what's the biggest gap between the
    percentage of defaulters I've caught and the percentage of
    good customers I've wrongly flagged?"
    A bigger gap = the model separates good from bad more cleanly.
    KS above ~0.3-0.4 is generally considered acceptable in credit
    risk practice; above 0.5 is strong.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    # tpr = % of actual defaulters caught at this threshold
    # fpr = % of actual good customers wrongly flagged at this threshold
    ks_values = tpr - fpr
    ks = np.max(ks_values)
    best_threshold = thresholds[np.argmax(ks_values)]
    return ks, best_threshold

print("=" * 60)
print("STEP 6: MODEL PERFORMANCE (KS & GINI)")
print("=" * 60)

for name, probs in [
    ("Logistic Regression (raw)", lr_probs_raw),
    ("Logistic Regression (calibrated)", lr_probs_cal),
    ("XGBoost (raw)", xgb_probs_raw),
    ("XGBoost (calibrated)", xgb_probs_cal),
]:
    gini, auc = gini_coefficient(y_test, probs)
    ks, best_thresh = ks_statistic(y_test, probs)
    print(f"\n{name}:")
    print(f"  AUC  = {auc:.3f}")
    print(f"  Gini = {gini:.3f}")
    print(f"  KS   = {ks:.3f}  (best separating cutoff ~ {best_thresh:.3f})")

# -----------------------------------------------------------------
# STEP 7: SAVE RESULTS FOR THE NEXT STAGE (SHAP + API)
# -----------------------------------------------------------------
# We save BOTH the raw models (SHAP explains these -- calibration doesn't
# change the tree structure, just rescales the output) AND the calibrated
# models (the API serves PDs from these, since a calibrated probability
# is the one that should actually be trusted as "real-world % chance").
import json
import joblib

joblib.dump(scaler, "scaler.joblib")
joblib.dump(log_reg, "logreg_model.joblib")
joblib.dump(xgb, "xgb_model.joblib")
joblib.dump(lr_calibrated, "logreg_calibrated.joblib")
joblib.dump(xgb_calibrated, "xgb_calibrated.joblib")

results_df = X_test.copy()
results_df["actual_default"] = y_test.values
results_df["lr_prob_calibrated"] = lr_probs_cal
results_df["xgb_prob_calibrated"] = xgb_probs_cal
results_df.to_csv("test_predictions.csv", index=False)

# The FastAPI /metrics endpoint (Phase 2) reads this instead of
# retraining/re-evaluating every time the server starts.
metrics = {}
for key, probs in [
    ("logreg_raw", lr_probs_raw),
    ("logreg_calibrated", lr_probs_cal),
    ("xgb_raw", xgb_probs_raw),
    ("xgb_calibrated", xgb_probs_cal),
]:
    gini, auc = gini_coefficient(y_test, probs)
    ks, best_thresh = ks_statistic(y_test, probs)
    metrics[key] = {
        "auc": round(float(auc), 4),
        "gini": round(float(gini), 4),
        "ks": round(float(ks), 4),
        "ks_best_threshold": round(float(best_thresh), 4),
    }
metrics["test_set_size"] = len(y_test)
metrics["test_set_default_rate"] = round(float(y_test.mean()), 4)

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n" + "=" * 60)
print("STEP 7: ARTIFACTS SAVED")
print("=" * 60)
print("Saved: scaler.joblib, logreg_model.joblib, xgb_model.joblib,")
print("       logreg_calibrated.joblib, xgb_calibrated.joblib,")
print("       test_predictions.csv, metrics.json")
