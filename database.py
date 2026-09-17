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
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

# Locally this just writes credit_risk.db next to the code, same as
# before. In Docker (Phase 5), docker-compose.yml sets CREDIT_RISK_DB_PATH
# to a path inside a mounted volume, so the audit trail survives
# container restarts/rebuilds instead of vanishing with the container.
DB_PATH = os.environ.get("CREDIT_RISK_DB_PATH", "credit_risk.db")

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
    shap_top_features TEXT NOT NULL,
    narrative TEXT,
    narrative_verified INTEGER,
    narrative_issues TEXT,
    narrative_generated_at TEXT
);
"""

# Phase 7 added the four narrative_* columns after some real
# credit_risk.db files already existed from earlier testing (Phases
# 4-6). CREATE_TABLE_SQL above only runs for a brand-new file, so
# existing databases need these columns added on top -- this is a
# minimal migration, not a full migration framework, appropriate for
# a single additive change to a SQLite prototype.
_NARRATIVE_COLUMNS = {
    "narrative": "TEXT",
    "narrative_verified": "INTEGER",
    "narrative_issues": "TEXT",
    "narrative_generated_at": "TEXT",
}


@contextmanager
def get_connection():
    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    """Create the scoring_log table if it doesn't exist yet, and add
    any columns a newer version of this file expects that an older
    database file doesn't have. Safe to call every time the API starts
    up -- both steps are no-ops once the schema is already current."""
    with get_connection() as conn:
        conn.execute(CREATE_TABLE_SQL)
        existing_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(scoring_log)")
        }
        for name, sql_type in _NARRATIVE_COLUMNS.items():
            if name not in existing_columns:
                conn.execute(f"ALTER TABLE scoring_log ADD COLUMN {name} {sql_type}")
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


def get_decision_by_id(log_id: int) -> dict | None:
    """Return one logged decision by its id, or None if it doesn't
    exist -- used by /narrate to look up the decision it's explaining."""
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM scoring_log WHERE id = ?", (log_id,)
        ).fetchone()
        return dict(row) if row else None


def save_narrative(log_id: int, narrative: str, verified: bool, issues: list) -> None:
    """Attach an LLM-generated explanation (Phase 7) to an existing
    logged decision. Separate from log_decision() because the
    narrative is generated on demand, after the decision itself is
    already recorded -- not every logged decision needs one."""
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE scoring_log
            SET narrative = ?,
                narrative_verified = ?,
                narrative_issues = ?,
                narrative_generated_at = ?
            WHERE id = ?
            """,
            (
                narrative,
                int(verified),
                json.dumps(issues),
                datetime.now(timezone.utc).isoformat(),
                log_id,
            ),
        )
        conn.commit()
