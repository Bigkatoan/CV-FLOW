"""Manages engine subprocess lifecycle."""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from app.config import settings

# session_id → subprocess.Popen
_sessions: dict[str, subprocess.Popen] = {}
# session_id → metadata
_meta: dict[str, dict] = {}


def get_running_sessions() -> dict[str, subprocess.Popen]:
    return _sessions


def start_session(session_id: str, pipeline_json: dict, params_override: dict | None = None) -> subprocess.Popen:
    settings.pipelines_tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = settings.pipelines_tmp_dir / f"{session_id}.json"
    tmp_path.write_text(json.dumps(pipeline_json))

    engine_main = Path(__file__).parent.parent.parent.parent / "engine" / "main.py"

    cmd = [
        settings.engine_python, str(engine_main),
        "--pipeline-json", str(tmp_path),
        "--session-id", session_id,
        "--ws-port", str(settings.engine_ws_port),
    ]
    if params_override:
        cmd += ["--params-override", json.dumps(params_override)]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    _sessions[session_id] = proc
    _meta[session_id] = {
        "pipeline_id": pipeline_json.get("id", ""),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    return proc


def stop_session(session_id: str) -> bool:
    proc = _sessions.get(session_id)
    if not proc:
        return False
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    _sessions.pop(session_id, None)
    return True


def session_status(session_id: str) -> str:
    proc = _sessions.get(session_id)
    if not proc:
        return "stopped"
    rc = proc.poll()
    if rc is None:
        return "running"
    return "error" if rc != 0 else "completed"


def session_meta(session_id: str) -> dict:
    return _meta.get(session_id, {})


def iter_logs(session_id: str, tail: int = 200) -> Iterator[str]:
    """Read last `tail` lines from the log file if it exists."""
    log_path = settings.pipelines_tmp_dir / f"{session_id}.log"
    if not log_path.exists():
        return
    lines = log_path.read_text(errors="replace").splitlines()
    yield from lines[-tail:]
