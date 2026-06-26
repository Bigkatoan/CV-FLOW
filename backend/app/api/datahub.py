"""Data Hub API — browse execution history, node metrics, detection events, and vector collections.

Security:
  - Relational table access is restricted to a hardcoded whitelist (ALLOWED_TABLES).
  - CSV export sanitises filenames from the whitelist only.
  - Vector collection names are sanitised to alphanumeric + underscore/hyphen.

Endpoints:
  GET  /api/datahub/relational/tables
  GET  /api/datahub/relational/sessions
  GET  /api/datahub/relational/sessions/{session_id}
  GET  /api/datahub/relational/events
  GET  /api/datahub/relational/export/{table}

  GET    /api/datahub/vector/collections
  POST   /api/datahub/vector/collections
  DELETE /api/datahub/vector/collections/{name}
  GET    /api/datahub/vector/collections/{name}/records
  POST   /api/datahub/vector/collections/{name}/records
  DELETE /api/datahub/vector/collections/{name}/{record_id}
  POST   /api/datahub/vector/collections/{name}/search
  GET    /api/datahub/vector/collections/{name}/export
"""
from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.datahub import DetectionEvent, ExecutionSession, NodeMetric

router = APIRouter(prefix="/datahub", tags=["datahub"])

# ── Security whitelist ────────────────────────────────────────────────────────
ALLOWED_TABLES = frozenset({
    "execution_sessions",
    "node_metrics",
    "detection_events",
    "pipelines",
    "model_registry",
    "compiled_nodes",
})

_COLLECTION_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def _validate_table(table: str) -> str:
    if table not in ALLOWED_TABLES:
        raise HTTPException(400, f"Table {table!r} is not accessible. Allowed: {sorted(ALLOWED_TABLES)}")
    return table


def _validate_collection(name: str) -> str:
    if not _COLLECTION_RE.match(name):
        raise HTTPException(400, "Collection name must be alphanumeric, underscores, or hyphens (1–64 chars)")
    return name


# ── Vector store registry (singleton per collection) ─────────────────────────
_vector_stores: dict[str, "VectorStore"] = {}


def _get_vector_store(name: str) -> "VectorStore":
    from engine.data.vector_store import VectorStore  # lazy import (engine package)
    if name not in _vector_stores:
        _vector_stores[name] = VectorStore(name, settings.storage_path / "vectordb")
    return _vector_stores[name]


def _list_vector_collections() -> list[dict]:
    base = settings.storage_path / "vectordb"
    if not base.exists():
        return []
    result = []
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        try:
            store = _get_vector_store(d.name)
            result.append({"name": d.name, "count": store.count, "dim": store.dim})
        except Exception:
            result.append({"name": d.name, "count": 0, "dim": 0})
    return result


# ── Relational endpoints ──────────────────────────────────────────────────────

@router.get("/relational/tables")
async def list_tables(db: AsyncSession = Depends(get_db)) -> list[dict]:
    """Return whitelisted tables with row counts."""
    result = []
    for table in sorted(ALLOWED_TABLES):
        try:
            row = await db.execute(text(f"SELECT COUNT(*) FROM {table}"))  # safe: whitelisted
            count = row.scalar()
        except Exception:
            count = None  # table may not exist yet
        result.append({"name": table, "row_count": count})
    return result


@router.get("/relational/sessions")
async def list_sessions(
    pipeline_id: Optional[str] = Query(None),
    status:      Optional[str] = Query(None),
    limit:       int           = Query(50, ge=1, le=500),
    offset:      int           = Query(0,  ge=0),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """List execution sessions with optional filters."""
    stmt = select(ExecutionSession).order_by(ExecutionSession.started_at.desc())
    if pipeline_id:
        stmt = stmt.where(ExecutionSession.pipeline_id == pipeline_id)
    if status:
        stmt = stmt.where(ExecutionSession.status == status)

    total_row = await db.execute(
        select(func.count()).select_from(stmt.subquery())
    )
    total = total_row.scalar() or 0

    rows = await db.execute(stmt.limit(limit).offset(offset))
    sessions = rows.scalars().all()

    return {
        "total":  total,
        "limit":  limit,
        "offset": offset,
        "items":  [
            {
                "id":          s.id,
                "pipeline_id": s.pipeline_id,
                "started_at":  s.started_at.isoformat() if s.started_at  else None,
                "ended_at":    s.ended_at.isoformat()   if s.ended_at    else None,
                "status":      s.status,
                "frame_count": s.frame_count,
                "error_msg":   s.error_msg,
                "mode":        s.mode,
            }
            for s in sessions
        ],
    }


@router.get("/relational/sessions/{session_id}")
async def get_session(session_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    """Get a single session with its node metrics."""
    row = await db.execute(
        select(ExecutionSession).where(ExecutionSession.id == session_id)
    )
    session = row.scalar_one_or_none()
    if not session:
        raise HTTPException(404, f"Session {session_id!r} not found")

    metrics_rows = await db.execute(
        select(NodeMetric).where(NodeMetric.session_id == session_id)
    )
    metrics = metrics_rows.scalars().all()

    return {
        "id":          session.id,
        "pipeline_id": session.pipeline_id,
        "started_at":  session.started_at.isoformat() if session.started_at else None,
        "ended_at":    session.ended_at.isoformat()   if session.ended_at   else None,
        "status":      session.status,
        "frame_count": session.frame_count,
        "error_msg":   session.error_msg,
        "mode":        session.mode,
        "node_metrics": [
            {
                "node_id":     m.node_id,
                "avg_ms":      m.avg_ms,
                "p95_ms":      m.p95_ms,
                "fps":         m.fps,
                "errors":      m.errors,
                "recorded_at": m.recorded_at.isoformat() if m.recorded_at else None,
            }
            for m in metrics
        ],
    }


@router.get("/relational/events")
async def list_events(
    event_type:  Optional[str] = Query(None),
    pipeline_id: Optional[str] = Query(None),
    session_id:  Optional[str] = Query(None),
    limit:       int           = Query(100, ge=1, le=1000),
    offset:      int           = Query(0,   ge=0),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """List detection events with optional filters."""
    stmt = select(DetectionEvent).order_by(DetectionEvent.timestamp.desc())
    if event_type:
        stmt = stmt.where(DetectionEvent.event_type == event_type)
    if pipeline_id:
        stmt = stmt.where(DetectionEvent.pipeline_id == pipeline_id)
    if session_id:
        stmt = stmt.where(DetectionEvent.session_id == session_id)

    total_row = await db.execute(select(func.count()).select_from(stmt.subquery()))
    total = total_row.scalar() or 0

    rows = await db.execute(stmt.limit(limit).offset(offset))
    events = rows.scalars().all()

    return {
        "total":  total,
        "limit":  limit,
        "offset": offset,
        "items":  [
            {
                "id":           e.id,
                "session_id":   e.session_id,
                "pipeline_id":  e.pipeline_id,
                "node_id":      e.node_id,
                "timestamp":    e.timestamp.isoformat() if e.timestamp else None,
                "event_type":   e.event_type,
                "payload":      json.loads(e.payload_json) if e.payload_json else {},
            }
            for e in events
        ],
    }


@router.get("/relational/export/{table}")
async def export_table_csv(
    table: str,
    limit: int = Query(10_000, ge=1, le=100_000),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Export a whitelisted table as CSV download."""
    _validate_table(table)
    rows = await db.execute(text(f"SELECT * FROM {table} LIMIT :lim"), {"lim": limit})
    all_rows = rows.fetchall()
    columns  = list(rows.keys()) if rows.keys() else []

    def _generate():
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(columns)
        yield buf.getvalue()
        buf.seek(0); buf.truncate()
        for row in all_rows:
            writer.writerow(row)
            yield buf.getvalue()
            buf.seek(0); buf.truncate()

    filename = f"{table}.csv"
    return StreamingResponse(
        _generate(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Vector endpoints ──────────────────────────────────────────────────────────

class CreateCollectionRequest(BaseModel):
    name: str
    dim:  int = 512


class AddRecordRequest(BaseModel):
    embedding: list[float]
    label:     str  = ""
    metadata:  dict = {}


class VectorSearchRequest(BaseModel):
    embedding: list[float]
    top_k:     int = 5


@router.get("/vector/collections")
async def list_vector_collections_api() -> list[dict]:
    """List all vector collections with count and dimension."""
    return _list_vector_collections()


@router.post("/vector/collections", status_code=201)
async def create_vector_collection(req: CreateCollectionRequest) -> dict:
    """Create a new (empty) vector collection."""
    _validate_collection(req.name)
    if req.dim < 1 or req.dim > 65536:
        raise HTTPException(400, "dim must be between 1 and 65536")
    from engine.data.vector_store import VectorStore
    store = VectorStore(req.name, settings.storage_path / "vectordb", dim=req.dim)
    _vector_stores[req.name] = store
    return {"name": req.name, "count": 0, "dim": req.dim}


@router.delete("/vector/collections/{name}", status_code=204)
async def delete_vector_collection(name: str) -> None:
    """Delete a vector collection and all its files."""
    import shutil
    _validate_collection(name)
    coll_dir = settings.storage_path / "vectordb" / name
    if not coll_dir.exists():
        raise HTTPException(404, f"Collection {name!r} not found")
    shutil.rmtree(str(coll_dir))
    _vector_stores.pop(name, None)


@router.get("/vector/collections/{name}/records")
async def list_vector_records(
    name:   str,
    limit:  int = Query(50,  ge=1, le=500),
    offset: int = Query(0,   ge=0),
) -> dict:
    """List records in a vector collection (id, label, metadata preview)."""
    _validate_collection(name)
    coll_dir = settings.storage_path / "vectordb" / name
    if not coll_dir.exists():
        raise HTTPException(404, f"Collection {name!r} not found")
    store = _get_vector_store(name)
    meta  = store._meta[offset: offset + limit]
    return {
        "collection": name,
        "total":      store.count,
        "limit":      limit,
        "offset":     offset,
        "records":    meta,
    }


@router.post("/vector/collections/{name}/records", status_code=201)
async def add_vector_record(name: str, req: AddRecordRequest) -> dict:
    """Add a single vector record to a collection."""
    _validate_collection(name)
    import uuid
    import numpy as np
    store = _get_vector_store(name)
    record_id = str(uuid.uuid4())
    meta = {"label": req.label, **req.metadata}
    store.add(record_id, np.array(req.embedding, dtype=np.float32), meta)
    return {"id": record_id, "label": req.label}


@router.delete("/vector/collections/{name}/{record_id}", status_code=204)
async def delete_vector_record(name: str, record_id: str) -> None:
    """Delete a single record from a vector collection by id."""
    _validate_collection(name)
    store = _get_vector_store(name)
    if not store.delete(record_id):
        raise HTTPException(404, f"Record {record_id!r} not found in collection {name!r}")


@router.post("/vector/collections/{name}/search")
async def search_vector_collection(name: str, req: VectorSearchRequest) -> list[dict]:
    """Cosine similarity search in a vector collection."""
    _validate_collection(name)
    coll_dir = settings.storage_path / "vectordb" / name
    if not coll_dir.exists():
        raise HTTPException(404, f"Collection {name!r} not found")
    import numpy as np
    store = _get_vector_store(name)
    q = np.array(req.embedding, dtype=np.float32)
    results = store.search(q, top_k=req.top_k)
    return [{"id": r.id, "score": round(r.score, 4), "metadata": r.metadata} for r in results]


@router.get("/vector/collections/{name}/export")
async def export_vector_collection(name: str) -> StreamingResponse:
    """Export a vector collection as a zip file (index.npy + meta.json)."""
    import zipfile
    import io as _io
    _validate_collection(name)
    coll_dir = settings.storage_path / "vectordb" / name
    if not coll_dir.exists():
        raise HTTPException(404, f"Collection {name!r} not found")

    buf = _io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in ("index.npy", "meta.json"):
            fpath = coll_dir / fname
            if fpath.exists():
                zf.write(str(fpath), fname)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{name}.zip"'},
    )
