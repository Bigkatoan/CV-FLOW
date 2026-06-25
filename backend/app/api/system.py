"""System utilities — package installer, etc."""
import subprocess
import sys
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter(prefix="/system", tags=["system"])


class PipInstallRequest(BaseModel):
    command: str  # e.g. "pip install ultralytics" or "torch torchvision"


def _parse_packages(raw: str) -> list[str]:
    """Accept 'pip install X Y' or just 'X Y' — return list of package tokens."""
    s = raw.strip()
    # Strip leading 'pip install' or 'pip install --upgrade' etc.
    if s.lower().startswith("pip"):
        parts = s.split()
        # Drop 'pip' and 'install' tokens; keep flags and package names
        parts = [p for i, p in enumerate(parts)
                 if not (i == 0 and p.lower() == "pip")
                 and not (i <= 1 and p.lower() == "install")]
        return parts
    return s.split()


def _stream_pip(packages: list[str]):
    """Run pip in a subprocess and yield SSE lines as bytes."""
    cmd = [sys.executable, "-m", "pip", "install", *packages]

    yield f"data: $ pip install {' '.join(packages)}\n\n".encode()

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line:
                yield f"data: {line}\n\n".encode()

        proc.wait()
        if proc.returncode == 0:
            yield b"data: \n\n"
            yield b"data: Installation complete.\n\n"
            yield b"event: done\ndata: ok\n\n"
        else:
            yield f"data: \ndata: pip exited with code {proc.returncode}\n\n".encode()
            yield b"event: done\ndata: error\n\n"
    except Exception as exc:
        yield f"data: ERROR: {exc}\n\n".encode()
        yield b"event: done\ndata: error\n\n"


@router.post("/pip-install")
async def pip_install(req: PipInstallRequest):
    """Stream pip install output as Server-Sent Events."""
    packages = _parse_packages(req.command)
    if not packages:
        return {"error": "No packages specified"}

    return StreamingResponse(
        _stream_pip(packages),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
