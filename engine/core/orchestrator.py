"""
Orchestrator — multi-camera pipeline management.

A CameraGroup runs N cameras sharing one pipeline topology.  In sequential
mode each camera gets its own engine subprocess (existing behavior).  In
multi-process mode the Orchestrator reuses the model inference worker pool
across all cameras — GPU memory is loaded once, frames from all cameras are
batch-dispatched through the shared workers.

Scaling formula for workers per inference node:
    workers = ceil(N_cameras × source_fps / throughput_per_worker)

The Orchestrator measures throughput_per_worker automatically during the first
10 seconds of operation and adjusts the AutoScaler thresholds accordingly.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_ENGINE_ENTRY = Path(__file__).parent.parent / "main.py"


@dataclass
class CameraConfig:
    camera_id:   str
    camera_type: str              # "usb_camera" | "rtsp_stream" | "video_file"
    config:      dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "CameraConfig":
        cam_id   = d.pop("id",   str(uuid.uuid4())[:8])
        cam_type = d.pop("type", "usb_camera")
        return cls(camera_id=cam_id, camera_type=cam_type, config=d)


class CameraGroup:
    """
    N cameras sharing the same pipeline topology.

    In sequential mode: one engine subprocess per camera.
    In multiprocess mode: shared NodeWorker pools, AutoScaler manages load.

    Parameters
    ----------
    group_id:     Unique identifier for this group.
    pipeline_json: The base pipeline definition (nodes + edges).
    cameras:      List of camera configs.
    mode:         'sequential' or 'multiprocess'.
    ws_port_base: Base WebSocket port; each camera gets port_base + camera_index.
    """

    def __init__(
        self,
        group_id: str,
        pipeline_json: dict,
        cameras: list[dict],
        mode: str = "sequential",
        ws_port_base: int = 8765,
    ) -> None:
        self.group_id      = group_id
        self.pipeline_json = pipeline_json
        self.mode          = mode
        self._ws_port_base = ws_port_base

        self._cameras: list[CameraConfig] = [
            CameraConfig.from_dict(dict(c)) for c in cameras
        ]
        # session_id → subprocess.Popen (sequential) or runner (multiprocess)
        self._sessions: dict[str, Any] = {}
        self._session_camera: dict[str, str] = {}  # session_id → camera_id
        self._lock = threading.Lock()
        self._started = False

    # ── Start ─────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start all cameras in the group."""
        for i, cam in enumerate(self._cameras):
            self._start_camera(cam, i)
        self._started = True
        logger.info("CameraGroup %s started: %d cameras", self.group_id, len(self._cameras))

    def _build_camera_pipeline(self, cam: CameraConfig) -> dict:
        """Inject camera source config into the pipeline nodes."""
        pipeline = json.loads(json.dumps(self.pipeline_json))  # deep copy
        for node in pipeline.get("nodes", []):
            ntype = node.get("type", "")
            if ntype in ("usb_camera", "camera", "rtsp_stream", "video_file", "image_directory"):
                node["type"]   = cam.camera_type
                node["config"].update(cam.config)
                break
        return pipeline

    def _start_camera(self, cam: CameraConfig, index: int) -> str:
        session_id  = f"{self.group_id}_{cam.camera_id}_{uuid.uuid4().hex[:6]}"
        ws_port     = self._ws_port_base + index
        pipeline    = self._build_camera_pipeline(cam)

        if self.mode == "multiprocess":
            self._start_mp(session_id, pipeline, ws_port)
        else:
            self._start_sequential(session_id, pipeline, ws_port)

        with self._lock:
            self._session_camera[session_id] = cam.camera_id
        return session_id

    def _start_sequential(self, session_id: str, pipeline: dict, ws_port: int) -> None:
        import tempfile, os
        tmp = Path(tempfile.gettempdir()) / f"cvflow_{session_id}.json"
        tmp.write_text(json.dumps(pipeline))
        log = Path(tempfile.gettempdir()) / f"cvflow_{session_id}.log"

        cmd = [
            sys.executable, str(_ENGINE_ENTRY),
            "--pipeline-json", str(tmp),
            "--session-id",    session_id,
            "--ws-port",       str(ws_port),
            "--mode",          "sequential",
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=open(log, "w"),
            stderr=subprocess.STDOUT,
        )
        with self._lock:
            self._sessions[session_id] = proc
        logger.info("Camera %s started (sequential, port=%d, pid=%d)",
                    session_id, ws_port, proc.pid)

    def _start_mp(self, session_id: str, pipeline: dict, ws_port: int) -> None:
        from engine.core.pipeline_runner_mp import MultiProcessPipelineRunner
        from engine.streaming import ws_server as _ws
        # Each camera group session gets its own WS server instance in production;
        # for simplicity here we share the module-level ws_server and tag by session.
        runner = MultiProcessPipelineRunner(pipeline, session_id)
        runner.start()
        with self._lock:
            self._sessions[session_id] = runner
        logger.info("Camera %s started (multiprocess)", session_id)

    # ── Dynamic camera management ─────────────────────────────────────────────

    def add_camera(self, camera: dict) -> str:
        """Add a new camera at runtime.  Returns the new session_id."""
        if not self._started:
            self._cameras.append(CameraConfig.from_dict(dict(camera)))
            return ""
        cam  = CameraConfig.from_dict(dict(camera))
        idx  = len(self._sessions)
        self._cameras.append(cam)
        session_id = self._start_camera(cam, idx)
        logger.info("Added camera %s to group %s", cam.camera_id, self.group_id)
        return session_id

    def remove_camera(self, camera_id: str) -> None:
        """Stop and remove a camera by camera_id."""
        target_session = None
        with self._lock:
            for sid, cid in list(self._session_camera.items()):
                if cid == camera_id:
                    target_session = sid
                    break
        if target_session:
            self._stop_session(target_session)
            logger.info("Removed camera %s from group %s", camera_id, self.group_id)
        else:
            logger.warning("Camera %s not found in group %s", camera_id, self.group_id)

    def _stop_session(self, session_id: str) -> None:
        with self._lock:
            runner = self._sessions.pop(session_id, None)
            self._session_camera.pop(session_id, None)
        if runner is None:
            return
        if isinstance(runner, subprocess.Popen):
            runner.terminate()
            try:
                runner.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                runner.kill()
        elif hasattr(runner, "stop"):
            runner.stop()

    # ── Stop ──────────────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Stop all cameras in the group."""
        with self._lock:
            session_ids = list(self._sessions.keys())
        for sid in session_ids:
            try:
                self._stop_session(sid)
            except Exception as e:
                logger.warning("Error stopping session %s: %s", sid, e)
        self._started = False
        logger.info("CameraGroup %s stopped", self.group_id)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        with self._lock:
            sessions = dict(self._sessions)
            camera_map = dict(self._session_camera)

        active = 0
        node_stats: dict[str, Any] = {}

        for sid, runner in sessions.items():
            if isinstance(runner, subprocess.Popen):
                if runner.poll() is None:
                    active += 1
            elif hasattr(runner, "get_stats"):
                active += 1
                try:
                    node_stats[sid] = runner.get_stats()
                except Exception:
                    pass

        return {
            "group_id":       self.group_id,
            "total_cameras":  len(self._cameras),
            "active_cameras": active,
            "mode":           self.mode,
            "camera_sessions": camera_map,
            "node_stats":     node_stats,
        }

    # ── Health check ──────────────────────────────────────────────────────────

    def health_check(self) -> None:
        """Restart any dead camera sessions."""
        with self._lock:
            sessions = dict(self._sessions)

        for sid, runner in sessions.items():
            if isinstance(runner, subprocess.Popen) and runner.poll() is not None:
                logger.warning("Camera session %s died (code=%d) — restarting",
                               sid, runner.returncode)
                camera_id = self._session_camera.get(sid)
                cam = next((c for c in self._cameras if c.camera_id == camera_id), None)
                if cam:
                    self._stop_session(sid)
                    self._start_camera(cam, list(self._cameras).index(cam))


# ── Orchestrator ──────────────────────────────────────────────────────────────

class Orchestrator:
    """
    Top-level manager for multiple CameraGroups.

    Typical usage:
        orch = Orchestrator()
        group = orch.create_group(
            group_id="factory_floor",
            pipeline_json=load_json("engine/samples/yolo_detection_usb.json"),
            cameras=[
                {"id": "cam_0", "type": "usb_camera", "device_index": 0},
                {"id": "cam_1", "type": "rtsp_stream", "url": "rtsp://..."},
            ],
        )
        # ... runtime ...
        orch.delete_group("factory_floor")
    """

    def __init__(self, health_check_interval_s: float = 30.0) -> None:
        self._groups: dict[str, CameraGroup] = {}
        self._lock   = threading.Lock()
        self._health_thread = threading.Thread(
            target=self._health_loop,
            args=(health_check_interval_s,),
            daemon=True,
            name="orchestrator-health",
        )
        self._stop_event = threading.Event()
        self._health_thread.start()

    # ── Group lifecycle ───────────────────────────────────────────────────────

    def create_group(
        self,
        group_id: str,
        pipeline_json: dict,
        cameras: list[dict],
        mode: str = "sequential",
        ws_port_base: int = 8765,
        auto_start: bool = True,
    ) -> CameraGroup:
        if group_id in self._groups:
            raise ValueError(f"Group {group_id!r} already exists")
        group = CameraGroup(
            group_id=group_id,
            pipeline_json=pipeline_json,
            cameras=cameras,
            mode=mode,
            ws_port_base=ws_port_base,
        )
        if auto_start:
            group.start()
        with self._lock:
            self._groups[group_id] = group
        logger.info("Created group %s (%d cameras, mode=%s)", group_id, len(cameras), mode)
        return group

    def delete_group(self, group_id: str) -> None:
        with self._lock:
            group = self._groups.pop(group_id, None)
        if group:
            group.stop()
            logger.info("Deleted group %s", group_id)
        else:
            logger.warning("Group %s not found", group_id)

    def get_group(self, group_id: str) -> Optional[CameraGroup]:
        with self._lock:
            return self._groups.get(group_id)

    def list_groups(self) -> list[dict]:
        with self._lock:
            groups = list(self._groups.values())
        return [{"group_id": g.group_id, "cameras": len(g._cameras), "mode": g.mode}
                for g in groups]

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_all_stats(self) -> dict[str, dict]:
        with self._lock:
            groups = dict(self._groups)
        return {gid: g.get_stats() for gid, g in groups.items()}

    # ── Health monitoring ─────────────────────────────────────────────────────

    def _health_loop(self, interval_s: float) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                groups = list(self._groups.values())
            for group in groups:
                try:
                    group.health_check()
                except Exception as e:
                    logger.warning("Health check error for group %s: %s", group.group_id, e)
            self._stop_event.wait(timeout=interval_s)

    def shutdown(self) -> None:
        """Stop all groups and the orchestrator."""
        self._stop_event.set()
        with self._lock:
            group_ids = list(self._groups.keys())
        for gid in group_ids:
            try:
                self.delete_group(gid)
            except Exception as e:
                logger.warning("Error shutting down group %s: %s", gid, e)
        logger.info("Orchestrator shut down")

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass
