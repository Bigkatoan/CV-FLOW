from datetime import datetime
from typing import Any
from pydantic import BaseModel


class ExecutionStartRequest(BaseModel):
    pipeline_id: str
    params_override: dict[str, Any] | None = None
    mode: str = "sequential"  # "sequential" | "multiprocess"


class ExecutionStartResponse(BaseModel):
    session_id: str
    ws_port: int = 8765


class SessionStatusResponse(BaseModel):
    session_id: str
    pipeline_id: str
    status: str        # running | stopped | error | completed
    started_at: datetime
    stopped_at: datetime | None = None
    error_msg: str | None = None
