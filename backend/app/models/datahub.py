"""
Data Hub ORM models — execution history, node metrics, detection events.

These three tables are created alongside the existing tables (pipelines,
model_registry, compiled_nodes) in the same cv_flow.db via create_all().

Write path: backend/app/services/execution_db.py (sync sqlite3)
Read path:  backend/app/api/datahub.py (async SQLAlchemy)
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ExecutionSession(Base):
    """One row per pipeline execution session."""

    __tablename__ = "execution_sessions"

    id:          Mapped[str]            = mapped_column(String(36),  primary_key=True)
    pipeline_id: Mapped[str]            = mapped_column(String(36),  nullable=False, index=True)
    started_at:  Mapped[datetime]       = mapped_column(DateTime(timezone=True), default=_utcnow)
    ended_at:    Mapped[datetime | None]= mapped_column(DateTime(timezone=True), nullable=True)
    status:      Mapped[str]            = mapped_column(String(16),  default="running")
    # status: "running" | "completed" | "stopped" | "error"
    frame_count: Mapped[int]            = mapped_column(Integer,     default=0)
    error_msg:   Mapped[str | None]     = mapped_column(Text,        nullable=True)
    mode:        Mapped[str]            = mapped_column(String(16),  default="sequential")
    # mode: "sequential" | "multiprocess"


class NodeMetric(Base):
    """Per-node timing stats flushed at session end from .stats.json."""

    __tablename__ = "node_metrics"

    id:          Mapped[int]      = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id:  Mapped[str]      = mapped_column(
        String(36), ForeignKey("execution_sessions.id"), index=True
    )
    node_id:     Mapped[str]      = mapped_column(String(64))
    avg_ms:      Mapped[float]    = mapped_column(Float)
    p95_ms:      Mapped[float]    = mapped_column(Float)
    fps:         Mapped[float]    = mapped_column(Float)
    errors:      Mapped[int]      = mapped_column(Integer)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class DetectionEvent(Base):
    """Individual detection / recognition events emitted during execution."""

    __tablename__ = "detection_events"

    id:           Mapped[int]      = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id:   Mapped[str]      = mapped_column(
        String(36), ForeignKey("execution_sessions.id"), index=True
    )
    pipeline_id:  Mapped[str]      = mapped_column(String(36), index=True)
    node_id:      Mapped[str]      = mapped_column(String(64))
    timestamp:    Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    event_type:   Mapped[str]      = mapped_column(String(32), index=True)
    # event_type: "object_detected" | "face_matched" | "face_unknown" | "counter_update"
    payload_json: Mapped[str]      = mapped_column(Text)
    # payload: JSON dict — bbox, class_name, confidence, identity_id, count, etc.
