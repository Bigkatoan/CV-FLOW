"""
Sync SQLite3 wrapper for execution session logging.

WHY THIS EXISTS
---------------
execution_service.py is pure sync code (subprocess management).
The main database layer uses SQLAlchemy async + aiosqlite, which cannot
be awaited from a sync context without creating a new event loop — fragile
and risky when called from inside a running uvicorn event loop.

Solution: use stdlib sqlite3 directly for the narrow write path
(insert_session, update_session_stopped, insert_node_metrics).
All read queries go through the normal async SQLAlchemy layer via datahub API.

The same physical cv_flow.db file is shared; SQLite handles concurrent
access via its built-in WAL mode / locking.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings


# Resolve once at import time so CWD changes during execution don't shift the path.
_DB_PATH: str = str(
    Path(settings.database_url.replace("sqlite+aiosqlite:///", "")).resolve()
)


def _db_path() -> str:
    return _DB_PATH


def _get_conn() -> sqlite3.Connection:
    """Open a new connection with WAL journal mode for better concurrency."""
    conn = sqlite3.connect(_db_path(), check_same_thread=False, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# ── Public API ─────────────────────────────────────────────────────────────────

def insert_session(
    session_id: str,
    pipeline_id: str,
    mode: str = "sequential",
) -> None:
    """Insert a new execution_sessions row with status='running'.

    Uses INSERT OR IGNORE so duplicate calls are safe (idempotent).
    """
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO execution_sessions
                (id, pipeline_id, started_at, status, frame_count, mode)
            VALUES (?, ?, ?, 'running', 0, ?)
            """,
            (session_id, pipeline_id, now, mode),
        )


def update_session_stopped(
    session_id: str,
    status: str,
    frame_count: int = 0,
    error_msg: str | None = None,
) -> None:
    """Update ended_at, status, frame_count, error_msg for a completed/stopped/errored session.

    status: 'completed' | 'stopped' | 'error'
    """
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        conn.execute(
            """
            UPDATE execution_sessions
               SET ended_at   = ?,
                   status     = ?,
                   frame_count= ?,
                   error_msg  = ?
             WHERE id = ?
            """,
            (now, status, frame_count, error_msg, session_id),
        )


def insert_node_metrics(session_id: str, metrics: dict[str, dict]) -> None:
    """Insert per-node timing stats into node_metrics.

    metrics format (from stats JSON written by pipeline_runner):
        {
            "node_id_1": {"avg_ms": 12.3, "p95_ms": 18.4, "fps": 22.0, "errors": 0},
            "node_id_2": {...},
        }
    """
    if not metrics:
        return
    now = datetime.now(timezone.utc).isoformat()
    rows = [
        (
            session_id,
            node_id,
            float(m.get("avg_ms", 0.0)),
            float(m.get("p95_ms", 0.0)),
            float(m.get("fps",    0.0)),
            int(  m.get("errors", 0)),
            now,
        )
        for node_id, m in metrics.items()
    ]
    with _get_conn() as conn:
        conn.executemany(
            """
            INSERT INTO node_metrics
                (session_id, node_id, avg_ms, p95_ms, fps, errors, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def insert_detection_event(
    session_id: str,
    pipeline_id: str,
    node_id: str,
    event_type: str,
    payload: dict,
) -> None:
    """Insert a single detection event.

    event_type: 'object_detected' | 'face_matched' | 'face_unknown' | 'counter_update'
    payload: arbitrary JSON-serialisable dict (bbox, confidence, identity, count, etc.)
    """
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO detection_events
                (session_id, pipeline_id, node_id, timestamp, event_type, payload_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, pipeline_id, node_id, now, event_type, json.dumps(payload)),
        )
