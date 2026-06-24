import io
import zipfile
from pathlib import Path
from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import StreamingResponse

from app.schemas.compiler import CompileRequest, CompileResponse
from app.services import compiler_service

router = APIRouter(prefix="/compile", tags=["compiler"])


@router.post("", response_model=CompileResponse)
async def compile_node(body: CompileRequest, bg: BackgroundTasks):
    result = compiler_service.compile_node(
        source_code=body.source_code,
        compile_flags=body.compile_flags,
        extra_libs=body.extra_libs,
    )
    return CompileResponse(
        status=result["status"],
        so_hash=result.get("so_hash"),
        stderr_output=result.get("stderr_output"),
        compiled_at=result.get("compiled_at"),
    )


@router.get("/sdk")
async def download_sdk():
    """Return a .zip of the C++ SDK headers and CMakeLists.txt template."""
    sdk_root = Path(__file__).parent.parent.parent.parent / "shared" / "cpp"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sdk_root.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(sdk_root.parent))
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=cv_flow_sdk.zip"},
    )
