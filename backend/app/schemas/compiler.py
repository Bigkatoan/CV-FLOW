from datetime import datetime
from pydantic import BaseModel


class CompileRequest(BaseModel):
    node_id: str
    source_code: str
    compile_flags: list[str] = ["-O2", "-march=native"]
    extra_libs: list[str] = []


class CompileResponse(BaseModel):
    status: str           # ok | error
    so_hash: str | None = None
    stderr_output: str | None = None
    compiled_at: datetime | None = None
