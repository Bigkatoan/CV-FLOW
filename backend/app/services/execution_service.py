"""Manages engine subprocess lifecycle."""
import json
import os
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


def _kill_all_running() -> None:
    """Terminate every running engine process so the WS port is free."""
    for sid, proc in list(_sessions.items()):
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=4)
            except subprocess.TimeoutExpired:
                proc.kill()
    _sessions.clear()
    _meta.clear()


def start_session(
    session_id: str,
    pipeline_json: dict,
    params_override: dict | None = None,
    mode: str = "sequential",
) -> subprocess.Popen:
    # In sequential mode only one engine can own the WS port — kill stale sessions.
    # In multiprocess mode each session gets a separate port, so we allow concurrency.
    if mode == "sequential":
        _kill_all_running()

    settings.pipelines_tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = settings.pipelines_tmp_dir / f"{session_id}.json"
    tmp_path.write_text(json.dumps(pipeline_json))

    # Write engine stdout+stderr to a log file so iter_logs() can read it.
    # Using PIPE and never draining it causes a pipe-buffer deadlock for long runs.
    log_path = settings.pipelines_tmp_dir / f"{session_id}.log"
    log_file = open(log_path, "w", buffering=1, encoding="utf-8", errors="replace")  # line-buffered

    engine_main = Path(__file__).parent.parent.parent.parent / "engine" / "main.py"

    # Assign a unique WS port: base + number of currently active sessions
    active_count = sum(1 for p in _sessions.values() if p.poll() is None)
    ws_port = settings.engine_ws_port + (active_count if mode != "sequential" else 0)

    cmd = [
        settings.engine_python, str(engine_main),
        "--pipeline-json", str(tmp_path),
        "--session-id", session_id,
        "--ws-port", str(ws_port),
        "--mode", mode,
    ]
    if params_override:
        cmd += ["--params-override", json.dumps(params_override)]

    stats_path = settings.pipelines_tmp_dir / f"{session_id}.stats.json"

    # Pass storage paths to the engine subprocess via environment variables
    # so engine nodes don't need to import from the backend `app` package.
    engine_env = {
        **os.environ,
        "CVFLOW_MODELS_DIR":   str(settings.models_dir),
        "CVFLOW_COMPILED_DIR": str(settings.compiled_dir),
        "CVFLOW_STATS_PATH":   str(stats_path),
    }

    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=log_file,
        env=engine_env,
    )
    _sessions[session_id] = proc
    _meta[session_id] = {
        "pipeline_id": pipeline_json.get("id", ""),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "log_path": str(log_path),
        "stats_path": str(stats_path),
        "ws_port": ws_port,
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
    """Read last `tail` lines from the engine log file."""
    log_path = Path(_meta.get(session_id, {}).get("log_path", ""))
    if not log_path.exists():
        # Fallback: try the conventional path
        log_path = settings.pipelines_tmp_dir / f"{session_id}.log"
    if not log_path.exists():
        yield "(no log file yet — engine may still be starting)"
        return
    try:
        lines = log_path.read_text(errors="replace").splitlines()
        yield from lines[-tail:]
    except OSError as e:
        yield f"(error reading log: {e})"
