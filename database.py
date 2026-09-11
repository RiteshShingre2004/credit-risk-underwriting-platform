"""
CREDIT RISK PIPELINE - PHASE 4: AUDIT TRAIL (SQLite)
========================================================
Goal: permanently record every underwriting decision the system makes,
so it can be reviewed later -- "why was this applicant rejected?",
"which model version was live when this decision was made?", "how many
REJECTs did we issue last month?".

We use Python's built-in sqlite3 module (no extra package to install)
writing to a single file, credit_risk.db. That file is enough for a
prototype; swapping to Postgres later would mean changing the
connection string, not the table design.

TABLE DESIGN, IN PLAIN ENGLISH:
One row = one decision. We store the 24 input features as their own
columns (X1..X24) rather than one big JSON blob, so a real SQL query
like "show me every REJECT where X5 was above 5000" works directly.
The SHAP explanation, though, is naturally a small nested list (top 5
features + their impact) -- that goes in as JSON text, since we don't
need to query inside it the way we'd query a feature value.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

DB_PATH = "credit_risk.db"

FEATURE_NAMES = [f"X{i}" for i in range(1, 25)]

_FEATURE_COLUMNS_SQL = ",\n    ".join(f"{name} REAL NOT NULL" for name in FEATURE_NAMES)

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS scoring_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    {_FEATURE_COLUMNS_SQL},
    logreg_pd REAL NOT NULL,
    xgb_pd REAL NOT NULL,
    model_version TEXT NOT NULL,
    calibration_method TEXT NOT NULL,
    decision_model TEXT NOT NULL,
    decision_tier TEXT NOT NULL,
    policy_approve_below REAL NOT NULL,
    policy_reject_above REAL NOT NULL,
    shap_top_features TEXT NOT NULL
);
"""


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    """Create the scoring_log table if it doesn't exist yet. Safe to
    call every time the API starts up -- it's a no-op if the table is
    already there."""
    with get_connection() as conn:
        conn.execute(CREATE_TABLE_SQL)
        conn.commit()


def log_decision(
    features: dict,
    logreg_pd: float,
    xgb_pd: float,
    model_version: str,
    calibration_method: str,
    decision_model: str,
    decision_tier: str,
    policy_approve_below: float,
    policy_reject_above: float,
    shap_top_features: list,
) -> int:
    """Insert one audited decision. Returns the new row's id."""
    columns = (
        ["created_at"]
        + FEATURE_NAMES
        + [
            "logreg_pd",
            "xgb_pd",
            "model_version",
            "calibration_method",
            "decision_model",
            "decision_tier",
            "policy_approve_below",
            "policy_reject_above",
            "shap_top_features",
        ]
    )
    values = (
        [datetime.now(timezone.utc).isoformat()]
        + [features[name] for name in FEATURE_NAMES]
        + [
            logreg_pd,
            xgb_pd,
            model_version,
            calibration_method,
            decision_model,
            decision_tier,
            policy_approve_below,
            policy_reject_above,
            json.dumps(shap_top_features),
        ]
    )
    placeholders = ", ".join("?" for _ in values)
    sql = f"INSERT INTO scoring_log ({', '.join(columns)}) VALUES ({placeholders})"

    with get_connection() as conn:
        cursor = conn.execute(sql, values)
        conn.commit()
        return cursor.lastrowid


def get_recent_decisions(limit: int = 50) -> list[dict]:
    """Return the most recent logged decisions, newest first."""
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM scoring_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]
