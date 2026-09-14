"""
CREDIT RISK PIPELINE - PHASE 6: DRIFT MONITORING (PSI)
========================================================
Goal: detect when NEW applicants start looking statistically different
from the applicants the model was trained on -- before you have enough
outcomes (defaults/repayments) to notice the model's accuracy slipping.

WHY THIS MATTERS: a model's accuracy metrics (KS, Gini, AUC) need
labels -- you only know if someone defaulted months after they took the
loan. Drift monitoring needs no labels at all: it just compares the
SHAPE of the new applicant population to the training population,
feature by feature, the moment new data arrives.

POPULATION STABILITY INDEX (PSI), IN PLAIN ENGLISH:
1. Take one feature (say, credit amount). Cut the TRAINING data's
   values into 10 equal-sized buckets (deciles) -- bucket boundaries
   come only from training data, and stay fixed after that.
2. Record what % of training applicants fell in each bucket (this is
   the "expected" shape, by definition ~10% per bucket).
3. Sort the NEW batch of applicants into those same fixed buckets, and
   record what % of the new batch falls in each one (the "actual" shape).
4. PSI = sum over buckets of (actual% - expected%) * ln(actual% / expected%)
   This is 0 when the two shapes match exactly, and grows the more they
   diverge in either direction.

Standard credit-industry thresholds:
    PSI < 0.10            -> stable, no action
    0.10 <= PSI < 0.25     -> moderate shift, worth investigating
    PSI >= 0.25            -> major shift, investigate before trusting
                              the model's scores on this population
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

PSI_WATCH_THRESHOLD = 0.10
PSI_ALERT_THRESHOLD = 0.25
N_BUCKETS = 10

# -----------------------------------------------------------------
# STEP 1: RELOAD THE SAME TRAIN/TEST SPLIT AS THE REST OF THE PROJECT
# -----------------------------------------------------------------
# Training data is the reference population every future batch gets
# compared against. Same random_state=42 as credit_pipeline.py and
# explainability.py, so this is the identical split.
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
# STEP 2: PSI FOR ONE FEATURE
# -----------------------------------------------------------------
def compute_psi(train_values, new_values, n_buckets=N_BUCKETS):
    """
    Compares one feature's distribution in `train_values` (the
    reference) against `new_values` (the batch being checked).
    Returns a single PSI number for this feature.
    """
    # Bucket edges come ONLY from training data, using quantiles so
    # each training bucket starts out with roughly equal population
    # (~10% each for 10 buckets). These edges then stay fixed -- the
    # new batch is measured against them, not against its own quantiles.
    quantiles = np.linspace(0, 1, n_buckets + 1)
    edges = np.unique(np.quantile(train_values, quantiles))
    # Guard against a feature with so few distinct values that some
    # quantile edges collide (e.g. a near-constant column) -- widen
    # the outer edges so every value is still captured in some bucket.
    edges[0] = -np.inf
    edges[-1] = np.inf

    train_counts, _ = np.histogram(train_values, bins=edges)
    new_counts, _ = np.histogram(new_values, bins=edges)

    train_pct = train_counts / len(train_values)
    new_pct = new_counts / len(new_values)

    # Buckets with 0% on either side would make ln(0) blow up. A tiny
    # floor (0.0001 = 0.01%) keeps the math defined without materially
    # changing the result -- this is the standard PSI convention.
    train_pct = np.clip(train_pct, 1e-4, None)
    new_pct = np.clip(new_pct, 1e-4, None)

    psi = np.sum((new_pct - train_pct) * np.log(new_pct / train_pct))
    return float(psi)


def psi_label(psi):
    if psi >= PSI_ALERT_THRESHOLD:
        return "INVESTIGATE"
    if psi >= PSI_WATCH_THRESHOLD:
        return "WATCH"
    return "stable"


# -----------------------------------------------------------------
# STEP 3: PSI ACROSS ALL FEATURES
# -----------------------------------------------------------------
def psi_report(new_batch, train_reference=X_train, n_buckets=N_BUCKETS):
    """
    Runs compute_psi() for every feature and returns a DataFrame
    sorted by PSI (worst first), with a plain-English status label.
    """
    rows = []
    for col in train_reference.columns:
        psi = compute_psi(train_reference[col], new_batch[col], n_buckets)
        rows.append({"feature": col, "psi": round(psi, 4), "status": psi_label(psi)})
    report = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    return report


def plot_psi_report(report, save_path="psi_report.png"):
    colors = report["status"].map(
        {"stable": "#2ca02c", "WATCH": "#ff7f0e", "INVESTIGATE": "#d62728"}
    )
    plt.figure(figsize=(9, 8))
    plt.barh(report["feature"], report["psi"], color=colors)
    plt.axvline(PSI_WATCH_THRESHOLD, color="#ff7f0e", linestyle="--", linewidth=1,
                label=f"watch ({PSI_WATCH_THRESHOLD})")
    plt.axvline(PSI_ALERT_THRESHOLD, color="#d62728", linestyle="--", linewidth=1,
                label=f"investigate ({PSI_ALERT_THRESHOLD})")
    plt.gca().invert_yaxis()  # highest PSI at the top
    plt.xlabel("PSI (Population Stability Index)")
    plt.title("Feature Drift vs. Training Population")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved PSI report chart to {save_path}")


# -----------------------------------------------------------------
# STEP 4: DEMO WHEN RUN DIRECTLY
# -----------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("DRIFT CHECK 1: real held-out test set vs. training data")
    print("=" * 60)
    print("These 300 applicants are genuinely unseen, but drawn from the")
    print("same population as training -- PSI should stay low everywhere.")
    print("This is the 'no false alarms' sanity check.\n")

    report_real = psi_report(X_test)
    print(report_real.to_string(index=False))
    n_flagged = (report_real["status"] != "stable").sum()
    print(f"\n{n_flagged} of 24 features flagged (expected: 0, or very few by chance).")

    print("\n" + "=" * 60)
    print("DRIFT CHECK 2: synthetically drifted batch")
    print("=" * 60)
    print("Simulating a population shift: credit amounts (X5) up 60%,")
    print("loan durations (X2) up 40%, to prove the alert actually fires")
    print("when the underlying applicant population really does change.\n")

    X_drifted = X_test.copy()
    X_drifted["X5"] = X_drifted["X5"] * 1.6
    X_drifted["X2"] = X_drifted["X2"] * 1.4

    report_drifted = psi_report(X_drifted)
    print(report_drifted.to_string(index=False))
    flagged = report_drifted[report_drifted["status"] != "stable"]
    print(f"\n{len(flagged)} of 24 features flagged:")
    for _, row in flagged.iterrows():
        print(f"  - {row['feature']}: PSI={row['psi']} ({row['status']})")

    plot_psi_report(report_drifted)
