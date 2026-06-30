"""
backend.app.db — SQLAlchemy engine + table definition for pipeline persistence.

A single `pipelines` table storing each PipelineSpec as a JSON blob keyed by
id. Pipeline specs are small JSON documents, not large binary data, so one
flat table is sufficient — no relational schema, no Alembic migrations.
"""
from __future__ import annotations

from sqlalchemy import Column, MetaData, String, Table, Text
from sqlalchemy.engine import Engine, create_engine

metadata = MetaData()

pipelines_table = Table(
    "pipelines", metadata,
    Column("id", String, primary_key=True),
    Column("data", Text, nullable=False),
)


def make_engine(db_path: str) -> Engine:
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        future=True,
    )
    metadata.create_all(engine)
    return engine
