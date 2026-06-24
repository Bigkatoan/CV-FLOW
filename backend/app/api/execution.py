import json
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.pipeline import Pipeline
from app.schemas.execution import ExecutionStartRequest, ExecutionStartResponse, SessionStatusResponse
from app.services import execution_service
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
        execution_service.start_session(session_id, pipeline_json, body.params_override)
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
