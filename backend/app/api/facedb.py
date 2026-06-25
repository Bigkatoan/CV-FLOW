"""REST endpoints for the Face Vector Database.

Since the engine runs as a subprocess, communication with face nodes uses:
  - Disk reads for queries  (storage/facedb/identities.json, seen_log.json)
  - File-based IPC for commands (.pending_confirms.json, .manual_trigger)

Endpoints:
  GET    /api/facedb/identities            — list all identities
  GET    /api/facedb/identities/{id}       — get one identity
  DELETE /api/facedb/identities/{id}       — delete identity (disk only)
  POST   /api/facedb/enroll               — confirm a pending enrollment
  POST   /api/facedb/enroll/trigger       — trigger manual enrollment
  GET    /api/facedb/seen_log             — recent seen-faces log
"""
from __future__ import annotations
import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/facedb", tags=["facedb"])

_DB_ROOT = Path("storage/facedb")
_CONFIRMS_FILE = _DB_ROOT / ".pending_confirms.json"
_TRIGGER_FILE  = _DB_ROOT / ".manual_trigger"
_LOCK = threading.Lock()


def _db_root() -> Path:
    _DB_ROOT.mkdir(parents=True, exist_ok=True)
    return _DB_ROOT


# ── schemas ───────────────────────────────────────────────────────────────────

class EnrollConfirmRequest(BaseModel):
    pending_id: str
    name: str
    attributes: dict[str, Any] = {}


class EnrollTriggerRequest(BaseModel):
    pass


# ── routes ────────────────────────────────────────────────────────────────────

@router.get("/identities")
async def list_identities():
    p = _db_root() / "identities.json"
    if not p.exists():
        return {"identities": []}
    try:
        return {"identities": json.loads(p.read_text())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/identities/{identity_id}")
async def get_identity(identity_id: str):
    p = _db_root() / "identities.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="No database found")
    try:
        entries = json.loads(p.read_text())
        entry = next((e for e in entries if e.get("id") == identity_id), None)
        if entry is None:
            raise HTTPException(status_code=404, detail="Identity not found")
        return entry
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/identities/{identity_id}")
async def delete_identity(identity_id: str):
    """Delete an identity. Writes a tombstone command the engine picks up on next frame."""
    p = _db_root() / "identities.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="No database found")
    try:
        with _LOCK:
            entries = json.loads(p.read_text())
            found = any(e.get("id") == identity_id for e in entries)
            if not found:
                raise HTTPException(status_code=404, detail="Identity not found")
            entries = [e for e in entries if e.get("id") != identity_id]
            p.write_text(json.dumps(entries, indent=2))
            # Remove embeddings file too
            emb = _db_root() / "embeddings" / f"{identity_id}.npy"
            if emb.exists():
                emb.unlink()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"deleted": identity_id}


@router.post("/enroll")
async def confirm_enroll(req: EnrollConfirmRequest):
    """Write confirmation to .pending_confirms.json — engine picks it up on next frame."""
    confirms_file = _db_root() / ".pending_confirms.json"
    with _LOCK:
        try:
            existing = json.loads(confirms_file.read_text()) if confirms_file.exists() else {}
        except Exception:
            existing = {}
        existing[req.pending_id] = {
            "name":       req.name,
            "attributes": req.attributes,
            "written_at": time.time(),
        }
        confirms_file.write_text(json.dumps(existing, indent=2))
    return {"status": "confirmed", "pending_id": req.pending_id, "name": req.name}


@router.post("/enroll/trigger")
async def trigger_enroll():
    """Write a trigger file — engine picks it up and starts manual enrollment."""
    trigger_file = _db_root() / ".manual_trigger"
    trigger_file.write_text(str(time.time()))
    return {"status": "triggered"}


@router.get("/seen_log")
async def get_seen_log(limit: int = 50):
    p = _db_root() / "seen_log.json"
    if not p.exists():
        return {"entries": [], "total": 0}
    try:
        entries = json.loads(p.read_text())
        # Strip crop_b64 (potentially large) from log entries for list view
        clean = [{k: v for k, v in e.items() if k != "crop_b64"} for e in entries]
        return {"entries": clean[-limit:], "total": len(entries)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
