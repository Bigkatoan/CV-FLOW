"""System utilities — package installer, system info."""
import subprocess
import sys
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/info")
async def system_info() -> dict:
    """Return CPU, RAM, and GPU information for the host running CV-FLOW."""
    try:
        import psutil
        cpu_count   = psutil.cpu_count(logical=True)
        cpu_percent = psutil.cpu_percent(interval=0.1)
        ram         = psutil.virtual_memory()
        result: dict = {
            "cpu_count":    cpu_count,
            "cpu_percent":  round(cpu_percent, 1),
            "ram_total_gb": round(ram.total / (1024 ** 3), 2),
            "ram_used_gb":  round(ram.used  / (1024 ** 3), 2),
            "ram_percent":  round(ram.percent, 1),
            "gpu":          [],
        }
    except ImportError:
        result = {
            "cpu_count":   None,
            "cpu_percent": None,
            "ram_total_gb": None,
            "ram_used_gb":  None,
            "ram_percent":  None,
            "gpu":          [],
            "warning":      "psutil not installed — run: pip install psutil",
        }

    # Optional GPU info via pynvml (NVIDIA only)
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            h    = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            mem  = pynvml.nvmlDeviceGetMemoryInfo(h)
            result["gpu"].append({
                "index":         i,
                "name":          name,
                "vram_total_mb": mem.total // (1024 ** 2),
                "vram_used_mb":  mem.used  // (1024 ** 2),
                "vram_percent":  round(mem.used / mem.total * 100, 1) if mem.total else 0,
            })
    except Exception:
        pass  # No GPU or pynvml not installed — result["gpu"] stays []

    return result




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
