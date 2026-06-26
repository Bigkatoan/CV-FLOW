"""Manages engine subprocess lifecycle.

Thread safety: a module-level threading.Lock protects _sessions and _meta
dicts from race conditions when concurrent API calls hit start/stop.

DB persistence: execution_db.py (sync sqlite3) is called OUTSIDE the lock
to avoid blocking other requests during I/O.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from app.config import settings

logger = logging.getLogger(__name__)

# session_id → subprocess.Popen
_sessions: dict[str, subprocess.Popen] = {}
# session_id → metadata dict
_meta: dict[str, dict] = {}

# Lock protecting _sessions and _meta
_lock = threading.Lock()


def get_running_sessions() -> dict[str, subprocess.Popen]:
    with _lock:
        return dict(_sessions)


def _kill_all_running_locked() -> None:
    """Terminate every running engine process. MUST be called with _lock held."""
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
    """Start a new engine subprocess.

    In sequential mode only one engine can own the WS port — kills stale sessions.
    In multiprocess mode each session gets a separate port (base + active_count).
    """
    settings.pipelines_tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = settings.pipelines_tmp_dir / f"{session_id}.json"
    tmp_path.write_text(json.dumps(pipeline_json))

    log_path   = settings.pipelines_tmp_dir / f"{session_id}.log"
    stats_path = settings.pipelines_tmp_dir / f"{session_id}.stats.json"

    log_file = open(log_path, "w", buffering=1, encoding="utf-8", errors="replace")

    engine_main = Path(__file__).parent.parent.parent.parent / "engine" / "main.py"

    with _lock:
        # Never auto-kill — every pipeline gets its own port; user stops sessions explicitly
        active_count = sum(1 for p in _sessions.values() if p.poll() is None)
        ws_port = settings.engine_ws_port + active_count

        engine_env = {
            **os.environ,
            "CVFLOW_MODELS_DIR":   str(settings.models_dir),
            "CVFLOW_COMPILED_DIR": str(settings.compiled_dir),
            "CVFLOW_STATS_PATH":   str(stats_path),
            "CVFLOW_SESSION_ID":   session_id,   # used by BenchmarkNode for file naming
        }

        cmd = [
            settings.engine_python, str(engine_main),
            "--pipeline-json", str(tmp_path),
            "--session-id",    session_id,
            "--ws-port",       str(ws_port),
            "--mode",          mode,
        ]
        if params_override:
            cmd += ["--params-override", json.dumps(params_override)]

        proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, env=engine_env)

        _sessions[session_id] = proc
        _meta[session_id] = {
            "pipeline_id": pipeline_json.get("id", ""),
            "started_at":  datetime.now(timezone.utc).isoformat(),
            "log_path":    str(log_path),
            "stats_path":  str(stats_path),
            "ws_port":     ws_port,
            "mode":        mode,
        }

    # DB write outside lock — non-blocking, no deadlock risk
    try:
        from app.services.execution_db import insert_session
        insert_session(session_id, pipeline_json.get("id", ""), mode=mode)
    except Exception as exc:
        logger.warning("execution_db.insert_session failed: %s", exc)

    return proc


def stop_session(session_id: str) -> bool:
    """Stop a running session, flush stats to DB, clean up state."""
    with _lock:
        proc = _sessions.get(session_id)
        if not proc:
            return False
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        rc = proc.returncode
        meta = _meta.pop(session_id, {})
        _sessions.pop(session_id, None)

    # Determine status
    if rc is None or rc == 0:
        status = "stopped"
    else:
        status = "error"

    # Read stats file and persist node metrics
    stats_path = Path(meta.get("stats_path", ""))
    frame_count = 0
    try:
        from app.services.execution_db import update_session_stopped, insert_node_metrics
        metrics: dict = {}
        if stats_path.exists():
            try:
                metrics = json.loads(stats_path.read_text())
                # frame_count: sum of frame counts if present, else max fps * rough duration
                frame_count = max(
                    (int(m.get("frame_count", 0)) for m in metrics.values()),
                    default=0,
                )
            except Exception as exc:
                logger.warning("Failed to read stats file %s: %s", stats_path, exc)

        if metrics:
            insert_node_metrics(session_id, metrics)
        update_session_stopped(session_id, status, frame_count=frame_count)
    except Exception as exc:
        logger.warning("execution_db.stop_session DB update failed: %s", exc)

    return True


def session_status(session_id: str) -> str:
    with _lock:
        proc = _sessions.get(session_id)
    if not proc:
        return "stopped"
    rc = proc.poll()
    if rc is None:
        return "running"
    return "error" if rc != 0 else "completed"


def session_meta(session_id: str) -> dict:
    with _lock:
        return dict(_meta.get(session_id, {}))


def iter_logs(session_id: str, tail: int = 200) -> Iterator[str]:
    """Read last `tail` lines from the engine log file."""
    with _lock:
        meta = dict(_meta.get(session_id, {}))
    log_path = Path(meta.get("log_path", ""))
    if not log_path.exists():
        log_path = settings.pipelines_tmp_dir / f"{session_id}.log"
    if not log_path.exists():
        yield "(no log file yet — engine may still be starting)"
        return
    try:
        lines = log_path.read_text(errors="replace").splitlines()
        yield from lines[-tail:]
    except OSError as e:
        yield f"(error reading log: {e})"
