"""
backend.app.pipeline_store — In-memory pipeline persistence.

A lightweight thread-safe dict-backed store. The visual editor's pipeline
specs are small JSON documents (no large binary blobs), so a process-local
store is sufficient; swap for a database-backed implementation later
without changing the API layer's contract.
"""
from __future__ import annotations

import threading
import uuid

from backend.app.schemas import PipelineRecord, PipelineSpec


class PipelineStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[str, PipelineRecord] = {}

    def create(self, spec: PipelineSpec) -> PipelineRecord:
        pipeline_id = uuid.uuid4().hex[:12]
        record = PipelineRecord(id=pipeline_id, **spec.model_dump())
        with self._lock:
            self._records[pipeline_id] = record
        return record

    def get(self, pipeline_id: str) -> PipelineRecord | None:
        with self._lock:
            return self._records.get(pipeline_id)

    def update(self, pipeline_id: str, spec: PipelineSpec) -> PipelineRecord | None:
        with self._lock:
            if pipeline_id not in self._records:
                return None
            record = PipelineRecord(id=pipeline_id, **spec.model_dump())
            self._records[pipeline_id] = record
            return record

    def delete(self, pipeline_id: str) -> bool:
        with self._lock:
            return self._records.pop(pipeline_id, None) is not None

    def list_all(self) -> list[PipelineRecord]:
        with self._lock:
            return list(self._records.values())

    def clear(self) -> None:
        with self._lock:
            self._records.clear()


# Process-wide singleton used by the API routers.
store = PipelineStore()
