"""
backend.app.pipeline_store — SQLite-backed pipeline persistence.

Each PipelineSpec is stored as a JSON blob keyed by id in a single
`pipelines` table (see backend.app.db) — small JSON documents, not large
binary blobs, so no relational schema is needed. Survives process restarts,
unlike the previous in-memory dict store.

DB location resolution order: explicit db_path argument > CV_FLOW_PIPELINE_DB_PATH
env var (used by the test suite to isolate state per session) > default
~/.cv_flow/pipelines.db.
"""
from __future__ import annotations

import os
import threading
import uuid
from pathlib import Path

from sqlalchemy import delete, select

from backend.app.db import make_engine, pipelines_table
from backend.app.schemas import PipelineRecord, PipelineSpec

DEFAULT_DB_PATH = Path.home() / ".cv_flow" / "pipelines.db"


class PipelineStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        path = Path(db_path or os.environ.get("CV_FLOW_PIPELINE_DB_PATH") or DEFAULT_DB_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._engine = make_engine(str(path))

    def create(self, spec: PipelineSpec) -> PipelineRecord:
        pipeline_id = uuid.uuid4().hex[:12]
        record = PipelineRecord(id=pipeline_id, **spec.model_dump())
        with self._lock, self._engine.begin() as conn:
            conn.execute(pipelines_table.insert().values(
                id=pipeline_id, data=record.model_dump_json(),
            ))
        return record

    def get(self, pipeline_id: str) -> PipelineRecord | None:
        with self._lock, self._engine.connect() as conn:
            row = conn.execute(
                select(pipelines_table.c.data).where(pipelines_table.c.id == pipeline_id)
            ).first()
        return PipelineRecord.model_validate_json(row[0]) if row else None

    def update(self, pipeline_id: str, spec: PipelineSpec) -> PipelineRecord | None:
        record = PipelineRecord(id=pipeline_id, **spec.model_dump())
        with self._lock, self._engine.begin() as conn:
            result = conn.execute(
                pipelines_table.update()
                .where(pipelines_table.c.id == pipeline_id)
                .values(data=record.model_dump_json())
            )
            if result.rowcount == 0:
                return None
        return record

    def delete(self, pipeline_id: str) -> bool:
        with self._lock, self._engine.begin() as conn:
            result = conn.execute(
                delete(pipelines_table).where(pipelines_table.c.id == pipeline_id)
            )
        return result.rowcount > 0

    def list_all(self) -> list[PipelineRecord]:
        with self._lock, self._engine.connect() as conn:
            rows = conn.execute(select(pipelines_table.c.data)).fetchall()
        return [PipelineRecord.model_validate_json(row[0]) for row in rows]

    def clear(self) -> None:
        with self._lock, self._engine.begin() as conn:
            conn.execute(delete(pipelines_table))


# Process-wide singleton used by the API routers.
store = PipelineStore()
