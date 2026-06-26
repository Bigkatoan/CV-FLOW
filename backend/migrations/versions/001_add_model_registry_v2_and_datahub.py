"""Add model_registry v2 columns + datahub tables

Adds:
  - 12 new nullable columns to model_registry (slug, tag, is_latest, parent_id,
    changelog, description, ports_json, last_used_at, size_bytes, author,
    license, extra_meta)
  - 3 new tables: execution_sessions, node_metrics, detection_events

All model_registry columns are nullable so existing rows are unaffected.
Backfills slug from existing name+version where slug IS NULL.

Revision ID: 001
Revises: (initial)
Create Date: 2026-06-26
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── execution_sessions ────────────────────────────────────────────────────
    op.create_table(
        "execution_sessions",
        sa.Column("id",          sa.String(36),  primary_key=True),
        sa.Column("pipeline_id", sa.String(36),  nullable=False),
        sa.Column("started_at",  sa.DateTime(timezone=True), nullable=True),
        sa.Column("ended_at",    sa.DateTime(timezone=True), nullable=True),
        sa.Column("status",      sa.String(16),  server_default="running"),
        sa.Column("frame_count", sa.Integer(),   server_default="0"),
        sa.Column("error_msg",   sa.Text(),      nullable=True),
        sa.Column("mode",        sa.String(16),  server_default="sequential"),
    )
    op.create_index("ix_execution_sessions_pipeline_id", "execution_sessions", ["pipeline_id"])

    # ── node_metrics ──────────────────────────────────────────────────────────
    op.create_table(
        "node_metrics",
        sa.Column("id",          sa.Integer(),   primary_key=True, autoincrement=True),
        sa.Column("session_id",  sa.String(36),  sa.ForeignKey("execution_sessions.id")),
        sa.Column("node_id",     sa.String(64),  nullable=False),
        sa.Column("avg_ms",      sa.Float(),     nullable=False),
        sa.Column("p95_ms",      sa.Float(),     nullable=False),
        sa.Column("fps",         sa.Float(),     nullable=False),
        sa.Column("errors",      sa.Integer(),   nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_node_metrics_session_id", "node_metrics", ["session_id"])

    # ── detection_events ──────────────────────────────────────────────────────
    op.create_table(
        "detection_events",
        sa.Column("id",           sa.Integer(),  primary_key=True, autoincrement=True),
        sa.Column("session_id",   sa.String(36), sa.ForeignKey("execution_sessions.id")),
        sa.Column("pipeline_id",  sa.String(36), nullable=False),
        sa.Column("node_id",      sa.String(64), nullable=False),
        sa.Column("timestamp",    sa.DateTime(timezone=True), nullable=True),
        sa.Column("event_type",   sa.String(32), nullable=False),
        sa.Column("payload_json", sa.Text(),     nullable=False),
    )
    op.create_index("ix_detection_events_session_id",  "detection_events", ["session_id"])
    op.create_index("ix_detection_events_pipeline_id", "detection_events", ["pipeline_id"])
    op.create_index("ix_detection_events_event_type",  "detection_events", ["event_type"])
    op.create_index("ix_detection_events_timestamp",   "detection_events", ["timestamp"])

    # ── model_registry new columns (render_as_batch → batch context for SQLite) ──
    with op.batch_alter_table("model_registry") as batch_op:
        batch_op.add_column(sa.Column("slug",         sa.String(128), nullable=True))
        batch_op.add_column(sa.Column("tag",          sa.String(16),  nullable=True, server_default="stable"))
        batch_op.add_column(sa.Column("is_latest",    sa.Boolean(),   nullable=True, server_default="1"))
        batch_op.add_column(sa.Column("parent_id",    sa.String(36),  nullable=True))
        batch_op.add_column(sa.Column("changelog",    sa.Text(),      nullable=True))
        batch_op.add_column(sa.Column("description",  sa.Text(),      nullable=True))
        batch_op.add_column(sa.Column("ports_json",   sa.Text(),      nullable=True))
        batch_op.add_column(sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("size_bytes",   sa.Integer(),   nullable=True))
        batch_op.add_column(sa.Column("author",       sa.String(128), nullable=True))
        batch_op.add_column(sa.Column("license",      sa.String(64),  nullable=True))
        batch_op.add_column(sa.Column("extra_meta",   sa.Text(),      nullable=True))
        batch_op.create_index("ix_model_registry_slug", ["slug"])

    # Backfill slug for existing records
    op.execute(
        "UPDATE model_registry "
        "SET slug = lower(replace(name, ' ', '-') || '-' || version) "
        "WHERE slug IS NULL"
    )


def downgrade() -> None:
    # Remove model_registry new columns
    with op.batch_alter_table("model_registry") as batch_op:
        batch_op.drop_index("ix_model_registry_slug")
        for col in [
            "slug", "tag", "is_latest", "parent_id", "changelog", "description",
            "ports_json", "last_used_at", "size_bytes", "author", "license", "extra_meta",
        ]:
            batch_op.drop_column(col)

    # Drop new tables
    op.drop_table("detection_events")
    op.drop_table("node_metrics")
    op.drop_table("execution_sessions")
