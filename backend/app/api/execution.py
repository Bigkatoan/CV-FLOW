import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.pipeline import Pipeline
from app.schemas.execution import ExecutionStartRequest, ExecutionStartResponse, SessionStatusResponse
from app.services import execution_service
from app.config import settings
import uuid

router = APIRouter(prefix="/execution", tags=["execution"])


@router.post("/start", response_model=ExecutionStartResponse, status_code=201)
async def start_pipeline(body: ExecutionStartRequest, db: AsyncSession = Depends(get_db)):
    row = await db.get(Pipeline, body.pipeline_id)
    if not row:
        raise HTTPException(404, "Pipeline not found")

    pipeline_json = json.loads(row.config_json)
    session_id = str(uuid.uuid4())

    try:
        execution_service.start_session(
            session_id, pipeline_json, body.params_override, mode=body.mode
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to start engine: {e}")

    return ExecutionStartResponse(session_id=session_id)


@router.post("/stop/{session_id}", status_code=200)
async def stop_pipeline(session_id: str):
    stopped = execution_service.stop_session(session_id)
    if not stopped:
        raise HTTPException(404, "Session not found or already stopped")
    return {"stopped": True}


@router.get("/status/{session_id}", response_model=SessionStatusResponse)
async def get_status(session_id: str):
    status = execution_service.session_status(session_id)
    meta = execution_service.session_meta(session_id)
    if not meta:
        raise HTTPException(404, "Session not found")
    return SessionStatusResponse(
        session_id=session_id,
        pipeline_id=meta.get("pipeline_id", ""),
        status=status,
        started_at=datetime.fromisoformat(meta["started_at"]),
    )


@router.get("/sessions", response_model=list[str])
async def list_sessions():
    return list(execution_service.get_running_sessions().keys())


@router.get("/logs/{session_id}", response_model=list[str])
async def get_logs(session_id: str, tail: int = 200):
    return list(execution_service.iter_logs(session_id, tail=tail))


@router.get("/stats/{session_id}")
async def get_stats(session_id: str):
    """Return per-node timing stats written by the engine runner."""
    meta = execution_service.session_meta(session_id)
    if not meta:
        raise HTTPException(404, "Session not found")
    stats_path = Path(meta.get("stats_path", str(settings.pipelines_tmp_dir / f"{session_id}.stats.json")))
    status = execution_service.session_status(session_id)
    if not stats_path.exists():
        return {"session_id": session_id, "status": status, "nodes": {}}
    try:
        data = json.loads(stats_path.read_text(errors="replace"))
        return {"session_id": session_id, "status": status, "nodes": data}
    except Exception as e:
        raise HTTPException(500, f"Error reading stats: {e}")


@router.get("/logs/{session_id}/stream")
async def stream_logs(session_id: str):
    """SSE endpoint — streams engine log lines as they are written, then signals done."""
    log_path = settings.pipelines_tmp_dir / f"{session_id}.log"

    async def _generate():
        # Wait up to 5 s for the log file to appear (engine may still be starting)
        for _ in range(50):
            if log_path.exists():
                break
            await asyncio.sleep(0.1)

        if not log_path.exists():
            yield b"data: (log file not found)\n\nevent: done\ndata: stop\n\n"
            return

        pos = 0
        # First: dump all existing content
        text = log_path.read_text(errors="replace")
        for line in text.splitlines():
            yield f"data: {line}\n\n".encode()
        pos = log_path.stat().st_size

        # Then: tail new content until the process exits
        while True:
            status = execution_service.session_status(session_id)
            try:
                size = log_path.stat().st_size
            except OSError:
                break
            if size > pos:
                chunk = log_path.read_bytes()[pos:size].decode(errors="replace")
                for line in chunk.splitlines():
                    if line:
                        yield f"data: {line}\n\n".encode()
                pos = size
            if status in ("stopped", "completed", "error"):
                yield b"event: done\ndata: " + status.encode() + b"\n\n"
                break
            await asyncio.sleep(0.3)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
