"""Prediction logging DB.

Phase I: SQLite locally (zero setup) withsame schema that migrates to RDS
(Postgres/MySQL) by just changing DATABASE_URL. Every /predict call is logged
with input features, prediction, probability, and nullable actual outcome
(filled later via POST /feedback — feeds Susendar's accuracy.xlsx + retraining).
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone

DB_PATH = os.getenv("PREDICTION_DB", "predictions.db")
# For RDS migration: set DATABASE_URL, e.g. postgresql://user:pass@host:5432/cicd
DATABASE_URL = os.getenv("DATABASE_URL", "")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    build_id TEXT,
    ts TEXT NOT NULL,
    repo TEXT DEFAULT 'unknown',
    team TEXT DEFAULT 'unknown',
    pipeline_stage TEXT DEFAULT 'build',
    features TEXT NOT NULL,
    prediction TEXT NOT NULL,
    probability REAL NOT NULL,
    model_version TEXT DEFAULT '',
    actual TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
"""

RDS_SCHEMA_SQL = """
-- RDS (Postgres) equivalent — run once on RDS instance:
-- CREATE TABLE predictions (
--   id SERIAL PRIMARY KEY,
--   build_id TEXT, ts TIMESTAMPTZ NOT NULL,
--   repo TEXT, team TEXT, pipeline_stage TEXT,
--   features JSONB NOT NULL, prediction TEXT NOT NULL,
--   probability DOUBLE PRECISION NOT NULL,
--   model_version TEXT, actual TEXT, created_at TIMESTAMPTZ DEFAULT now()
-- );
-- CREATE INDEX idx_predictions_repo ON predictions(repo);
-- CREATE INDEX idx_predictions_team ON predictions(team);
-- CREATE INDEX idx_predictions_stage ON predictions(pipeline_stage);
"""


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA_SQL)
    conn.commit()
    return conn


def log_prediction(build_id, repo, team, stage, features, prediction, probability, model_version) -> int:
    conn = get_conn()
    ts = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        "INSERT INTO predictions (build_id, ts, repo, team, pipeline_stage, features, prediction, probability, model_version)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        (build_id, ts, repo, team, stage, json.dumps(features), prediction, probability, model_version),
    )
    conn.commit()
    row_id = cur.lastrowid or 0
    conn.close()
    return row_id


def set_actual(build_id: str, actual: str) -> int:
    conn = get_conn()
    cur = conn.execute("UPDATE predictions SET actual=? WHERE build_id=?", (actual, build_id))
    conn.commit()
    n = cur.rowcount
    conn.close()
    return n


def query_analytics(group_by: str = "repo"):
    """Fleet-level DevOps analytics aggregation. group_by: repo|team|pipeline_stage."""
    allowed = {"repo": "repo", "team": "team", "pipeline_stage": "pipeline_stage"}
    col = allowed.get(group_by, "repo")
    conn = get_conn()
    rows = conn.execute(
        f"SELECT {col}, COUNT(*), SUM(CASE WHEN prediction='failed' THEN 1 ELSE 0 END), AVG(probability)"
        f" FROM predictions GROUP BY {col} ORDER BY 2 DESC"
    ).fetchall()
    conn.close()
    return [
        {"group": r[0], "total": r[1], "predicted_failures": r[2] or 0, "avg_failure_prob": round(r[3] or 0, 4)}
        for r in rows
    ]
