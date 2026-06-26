"""Face Logging DB API — store and browse face crops with dedup and cooldown.

Endpoints:
  POST   /api/logging/{name}/add              — atomic check-and-add
  GET    /api/logging/{name}/faces            — paginated list with base64 thumbnails
  GET    /api/logging/{name}/faces/{id}/image — raw JPEG image
  GET    /api/logging/{name}/stats            — count, max_count, size_mb
  DELETE /api/logging/{name}/faces/{id}       — delete a single record
"""
from __future__ import annotations

import base64
import io
import re
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import settings

router = APIRouter(prefix="/logging", tags=["logging"])

_COLL_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

_logging_dbs: dict[str, "LoggingDB"] = {}


def _validate_name(name: str) -> str:
    if not _COLL_RE.match(name):
        raise HTTPException(400, "Collection name must be alphanumeric, underscores, or hyphens (1–64 chars)")
    return name


def _get_db(name: str) -> "LoggingDB":
    from engine.data.logging_db import LoggingDB
    if name not in _logging_dbs:
        max_f = 1000
        _logging_dbs[name] = LoggingDB(name, settings.storage_path / "logging", max_faces=max_f)
    return _logging_dbs[name]


# ── Request / response models ─────────────────────────────────────────────────

class AddFaceRequest(BaseModel):
    embedding:     list[float]
    image_b64:     str
    metadata:      dict = {}
    cooldown_sec:  int   = 300
    sim_threshold: float = 0.6


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/{name}/add")
async def add_face(name: str, req: AddFaceRequest) -> dict:
    """Atomic check-and-add: returns {added, id, reason}."""
    _validate_name(name)
    import numpy as np
    import cv2

    db  = _get_db(name)
    emb = np.array(req.embedding, dtype=np.float32)

    can, reason = db.can_add(emb, cooldown_sec=req.cooldown_sec, sim_threshold=req.sim_threshold)
    if not can:
        return {"added": False, "id": None, "reason": reason}

    # Decode base64 image
    try:
        img_bytes = base64.b64decode(req.image_b64)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("image decode failed")
    except Exception as e:
        raise HTTPException(400, f"Invalid image_b64: {e}")

    record_id = db.add(emb, img, metadata=req.metadata)
    return {"added": True, "id": record_id, "reason": "ok"}


@router.get("/{name}/faces")
async def list_faces(
    name:   str,
    limit:  int = Query(20, ge=1, le=200),
    offset: int = Query(0,  ge=0),
    thumb:  bool = Query(True),
) -> dict:
    """List logged faces, newest first. Includes base64 thumbnails when thumb=true."""
    _validate_name(name)
    db = _get_db(name)
    items = db.list_faces(limit=limit, offset=offset)

    faces_out = []
    for item in items:
        entry = dict(item)
        if thumb:
            img_path = db.get_image_path(item["id"])
            if img_path:
                entry["image_b64"] = base64.b64encode(img_path.read_bytes()).decode()
        faces_out.append(entry)

    return {
        "count":     db.count,
        "max_count": db.max_faces,
        "size_mb":   db.estimate_size_mb,
        "faces":     faces_out,
    }


@router.get("/{name}/faces/{record_id}/image")
async def get_face_image(name: str, record_id: str) -> StreamingResponse:
    """Return raw JPEG for a logged face."""
    _validate_name(name)
    # Validate record_id is UUID-like to prevent path traversal
    if not re.match(r"^[a-f0-9-]{36}$", record_id):
        raise HTTPException(400, "Invalid record id format")
    db = _get_db(name)
    img_path = db.get_image_path(record_id)
    if img_path is None:
        raise HTTPException(404, f"Image {record_id!r} not found in collection {name!r}")
    return StreamingResponse(
        io.BytesIO(img_path.read_bytes()),
        media_type="image/jpeg",
    )


@router.get("/{name}/stats")
async def get_stats(name: str) -> dict:
    """Return count, max_count, size_mb for a logging collection."""
    _validate_name(name)
    db = _get_db(name)
    return {
        "name":      name,
        "count":     db.count,
        "max_count": db.max_faces,
        "size_mb":   db.estimate_size_mb,
        "dim":       db.dim,
    }


@router.delete("/{name}/faces/{record_id}", status_code=204)
async def delete_face(name: str, record_id: str) -> None:
    """Delete a single logged face by id."""
    _validate_name(name)
    if not re.match(r"^[a-f0-9-]{36}$", record_id):
        raise HTTPException(400, "Invalid record id format")
    db = _get_db(name)
    if not db.delete(record_id):
        raise HTTPException(404, f"Record {record_id!r} not found in {name!r}")
