"""
MultiProcessPipelineRunner — async DAG runner for multi-process mode.

Each node runs in its own OS process (Python) or thread (C++).
Communication is exclusively via PortBus (Direct Access Memory).
The runner:
  1. Parses the pipeline DAG (topological sort from pipeline_builder).
  2. Allocates one PortBus per edge.
  3. Spawns NodeWorkers for all nodes.
  4. Starts a monitor thread that feeds AutoScaler at ~1 Hz.
  5. Starts a WS relay thread that polls the terminal output bus
     and pushes JPEG frames to the WebSocket server.

Backward compatibility: this runner is only used when
engine/main.py is called with --mode multiprocess.
The existing sequential PipelineRunner remains the default.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from engine.core.autoscaler import AutoScaler, ScalingPolicy
from engine.core.dam import MergeBus, PortBus, RoundRobinBus, make_port_bus
from engine.core.node_worker import NodeWorker, ResourceLimits
from engine.core.pipeline_builder import build_pipeline
from engine.streaming import ws_server

logger = logging.getLogger(__name__)

_WS_STREAM_FPS   = 15
_WS_STREAM_QUAL  = 75
_MONITOR_HZ      = 1.0     # AutoScaler tick rate
_STATS_REPORT_HZ = 5.0     # WS stats event rate


def _parse_scaling(node_json: dict) -> ScalingPolicy:
    return ScalingPolicy.from_dict(node_json.get("scaling", {}))


def _parse_limits(node_json: dict) -> Optional[ResourceLimits]:
    r = node_json.get("resources")
    return ResourceLimits.from_dict(r) if r else None


def _bus_name(edge_id: str, session_id: str) -> str:
    """Generate a unique, valid shared memory name (no slashes)."""
    safe = edge_id.replace("/", "_").replace("-", "_")
    return f"cvflow_{session_id[:8]}_{safe}"


class MultiProcessPipelineRunner:
    """
    Async DAG pipeline runner.

    Parameters
    ----------
    pipeline_json:  Full pipeline dict {nodes, edges}.
    session_id:     Unique session identifier.
    ws_stream_port: Not used directly (ws_server is module-level singleton).
    """

    def __init__(
        self,
        pipeline_json: dict,
        session_id: str,
    ) -> None:
        self.pipeline_json = pipeline_json
        self.session_id    = session_id

        self._nodes_json: list[dict] = pipeline_json.get("nodes", [])
        self._edges_json: list[dict] = pipeline_json.get("edges", [])

        self._buses:       dict[str, PortBus]            = {}  # edge_id → bus
        self._rr_buses:    dict[str, RoundRobinBus]      = {}  # node_id → fan-out
        self._merge_buses: dict[str, MergeBus]           = {}  # node_id → fan-in
        self._workers:     dict[str, list[NodeWorker]]   = {}  # node_id → [worker, ...]
        self._autoscaler   = AutoScaler()

        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._ws_relay_thread: Optional[threading.Thread] = None

        # The "terminal" node is the last in topo order — its output is streamed
        self._terminal_bus: Optional[PortBus] = None

        # Stats file written periodically so backend /stats endpoint can read it
        self._stats_path: Optional[Path] = (
            Path(os.environ["CVFLOW_STATS_PATH"])
            if "CVFLOW_STATS_PATH" in os.environ else None
        )

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build_edge_map(self) -> dict[str, dict]:
        """
        Returns {target_node_id: {port_name: edge}} for fast lookup.
        Also {source_node_id: {port_name: edge}}.
        """
        edges_by_target: dict[str, dict] = {}
        edges_by_source: dict[str, dict] = {}
        for e in self._edges_json:
            src_id   = e["source"]
            dst_id   = e["target"]
            src_port = e.get("sourceHandle", "out")
            dst_port = e.get("targetHandle", "in")
            edge_id  = f"{src_id}:{src_port}__{dst_id}:{dst_port}"

            edges_by_target.setdefault(dst_id, {})[dst_port] = {
                "edge_id": edge_id, "src": src_id, "src_port": src_port,
            }
            edges_by_source.setdefault(src_id, {})[src_port] = {
                "edge_id": edge_id, "dst": dst_id, "dst_port": dst_port,
            }
        return edges_by_target, edges_by_source

    def _resolve_node_class(self, node_json: dict):
        """Resolve node_type string to a Python class (reuses existing registry)."""
        from engine.core.pipeline_builder import build_pipeline as _bp
        node_type = node_json["type"]
        # Build a mini pipeline with just this node to get its class
        # (re-uses existing logic in pipeline_builder.build_pipeline)
        import importlib, pathlib

        # Map of built-in node types → module paths
        _BUILTIN = {
            "python_node": ("engine.nodes.python_code_node", "PythonCodeNode"),
            "cpp_node":    ("engine.nodes.cpp.cpp_node",     "CppCodeNode"),
            "usb_camera":  ("engine.nodes.input.usb_camera", "USBCameraNode"),
            "camera":      ("engine.nodes.input.camera",     "CameraNode"),
            "rtsp_stream": ("engine.nodes.input.rtsp_stream","RTSPStreamNode"),
            "video_file":  ("engine.nodes.input.video_file", "VideoFileNode"),
            "image_directory": ("engine.nodes.input.image_directory", "ImageDirectoryNode"),
            "preprocess":  ("engine.nodes.processing.preprocess", "PreprocessNode"),
            "model_inference": ("engine.nodes.processing.model_inference", "ModelInferenceNode"),
            "nms":         ("engine.nodes.processing.postprocess_nms", "PostprocessNMSNode"),
            "draw_bbox":   ("engine.nodes.processing.draw_bbox", "DrawBboxNode"),
            "crop_bbox":   ("engine.nodes.processing.crop_bbox", "CropBBoxNode"),
            "stream_viewer": ("engine.nodes.output.stream_viewer", "StreamViewerNode"),
            "video_writer":  ("engine.nodes.output.video_writer", "VideoWriterNode"),
            "object_tracker": ("engine.nodes.spatial.object_tracker", "ObjectTrackerNode"),
            "counter":     ("engine.nodes.spatial.counter", "CounterNode"),
            "filter":      ("engine.nodes.utility.filter_node", "FilterNode"),
            "python_function": ("engine.nodes.utility.python_function", "PythonFunctionNode"),
        }
        if node_type in _BUILTIN:
            mod_path, cls_name = _BUILTIN[node_type]
            mod = importlib.import_module(mod_path)
            return getattr(mod, cls_name)
        raise ValueError(f"Unknown node type: {node_type!r}")

    def _default_bus_dims(self) -> tuple[int, int, int]:
        """Infer sensible PortBus dimensions from pipeline config."""
        for n in self._nodes_json:
            cfg = n.get("config", {})
            w = cfg.get("width", 0) or cfg.get("max_w", 0)
            h = cfg.get("height", 0) or cfg.get("max_h", 0)
            if w and h:
                return w, h, 3
        return 1280, 720, 3

    # ── Start ─────────────────────────────────────────────────────────────────

    def start(self) -> None:
        logger.info("MultiProcessPipelineRunner starting (session=%s)", self.session_id)

        edges_by_target, edges_by_source = self._build_edge_map()
        default_w, default_h, default_c  = self._default_bus_dims()

        # Topo-sorted node order from existing builder
        try:
            ordered_nodes = build_pipeline(self.pipeline_json)
        except Exception as e:
            logger.error("Pipeline build failed: %s", e)
            raise

        # Build a lookup: node_id → node_json
        node_json_map = {n["id"]: n for n in self._nodes_json}

        # ── Allocate PortBuses for every edge ─────────────────────────────────
        for e in self._edges_json:
            src_id   = e["source"]
            dst_id   = e["target"]
            src_port = e.get("sourceHandle", "out")
            dst_port = e.get("targetHandle", "in")
            edge_id  = f"{src_id}:{src_port}__{dst_id}:{dst_port}"
            bname    = _bus_name(edge_id, self.session_id)

            nj   = node_json_map.get(src_id, {})
            cfg  = nj.get("config", {})
            w    = cfg.get("width", default_w) or default_w
            h    = cfg.get("height", default_h) or default_h

            bus = make_port_bus(bname, max_w=w, max_h=h, max_c=default_c, create=True)
            self._buses[edge_id] = bus
            logger.debug("Allocated PortBus %s (%d×%d)", bname, w, h)

        # Mark terminal bus (output of last node that has a stream_viewer)
        stream_nodes = [n for n in self._nodes_json if "stream" in n.get("type", "")]
        terminal_node_id = stream_nodes[-1]["id"] if stream_nodes else (
            ordered_nodes[-1].node_id if ordered_nodes else None
        )

        # ── Spawn workers ─────────────────────────────────────────────────────
        for node_obj in ordered_nodes:
            nid  = node_obj.node_id
            nj   = node_json_map.get(nid, {})
            cfg  = nj.get("config", {})
            policy = _parse_scaling(nj)
            limits = _parse_limits(nj)

            # Collect input buses for this node
            node_in_buses: dict[str, PortBus | MergeBus] = {}
            for dst_port, edge_info in edges_by_target.get(nid, {}).items():
                eid = edge_info["edge_id"]
                if eid in self._buses:
                    node_in_buses[dst_port] = self._buses[eid]

            # Collect output buses for this node
            node_out_buses: dict[str, PortBus | RoundRobinBus] = {}
            for src_port, edge_info in edges_by_source.get(nid, {}).items():
                eid = edge_info["edge_id"]
                if eid in self._buses:
                    bus = self._buses[eid]
                    node_out_buses[src_port] = bus
                    if nid == terminal_node_id and src_port in ("out", "frame"):
                        self._terminal_bus = bus

            # Alias "in"→"frame" and "out"→"frame" for simpler workers
            if "in" in node_in_buses and "frame" not in node_in_buses:
                node_in_buses["frame"] = node_in_buses["in"]
            if "out" in node_out_buses and "frame" not in node_out_buses:
                node_out_buses["frame"] = node_out_buses["out"]

            node_class = None
            try:
                node_class = self._resolve_node_class(nj)
            except Exception as e:
                logger.warning("Cannot resolve class for %s: %s — skipping MP mode", nid, e)
                continue

            worker = NodeWorker(
                node_class=node_class,
                node_id=nid,
                node_type=nj.get("type", ""),
                config=cfg,
                input_buses=node_in_buses,
                output_buses=node_out_buses,
                resource_limits=limits,
                worker_index=0,
            )
            worker.start()
            self._workers[nid] = [worker]

            # Register with AutoScaler if scaling is configured
            if policy.max_workers > 1:
                in_bus = node_in_buses.get("frame") or next(iter(node_in_buses.values()), None)
                if in_bus and isinstance(in_bus, PortBus):
                    rr = RoundRobinBus([in_bus])
                    self._rr_buses[nid] = rr

                    def _make_spawn_fn(n_class, n_id, n_type, n_cfg, in_b, out_b, lim):
                        def spawn(idx):
                            w = NodeWorker(
                                node_class=n_class, node_id=n_id, node_type=n_type,
                                config=n_cfg, input_buses=in_b, output_buses=out_b,
                                resource_limits=lim, worker_index=idx,
                            )
                            return w
                        return spawn

                    self._autoscaler.register(
                        node_id=nid,
                        workers=self._workers[nid],
                        input_bus=rr,
                        spawn_fn=_make_spawn_fn(
                            node_class, nid, nj.get("type", ""), cfg,
                            node_in_buses, node_out_buses, limits,
                        ),
                        policy=policy,
                    )

        # ── Background threads ────────────────────────────────────────────────
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="mp-monitor"
        )
        self._monitor_thread.start()

        self._ws_relay_thread = threading.Thread(
            target=self._ws_relay_loop, daemon=True, name="mp-ws-relay"
        )
        self._ws_relay_thread.start()

        logger.info("MultiProcessPipelineRunner started: %d nodes", len(self._workers))

    # ── Monitor loop ──────────────────────────────────────────────────────────

    def _monitor_loop(self) -> None:
        stats_interval = 1.0 / _STATS_REPORT_HZ
        last_stats_t   = 0.0
        while not self._stop_event.is_set():
            # AutoScaler tick
            try:
                self._autoscaler.tick()
            except Exception as e:
                logger.warning("AutoScaler tick error: %s", e)

            # Restart dead workers
            for nid, workers in self._workers.items():
                for i, w in enumerate(workers):
                    if not w.is_alive():
                        logger.warning("Worker %s[%d] died — restarting", nid, i)
                        try:
                            w.start()
                        except Exception as re:
                            logger.error("Restart failed for %s[%d]: %s", nid, i, re)

            # Publish stats to WS and write to stats file
            now = time.monotonic()
            if now - last_stats_t >= stats_interval:
                last_stats_t = now
                try:
                    stats = self.get_stats()
                    ws_server.send_event(self.session_id, {
                        "type": "node_stats",
                        "stats": stats,
                    })
                    if self._stats_path:
                        self._stats_path.write_text(
                            json.dumps(stats, default=str), encoding="utf-8"
                        )
                except Exception:
                    pass

            time.sleep(1.0 / _MONITOR_HZ)

    # ── WS relay loop ─────────────────────────────────────────────────────────

    def _ws_relay_loop(self) -> None:
        """Poll terminal output bus, encode JPEG, push to WS."""
        if self._terminal_bus is None:
            logger.debug("No terminal bus — WS relay disabled")
            return

        interval = 1.0 / _WS_STREAM_FPS
        last_t   = 0.0

        while not self._stop_event.is_set():
            now = time.monotonic()
            if now - last_t < interval:
                time.sleep(0.001)
                continue

            result = self._terminal_bus.read(timeout_ms=50)
            if result is None:
                continue

            frame, detections, metadata = result
            last_t = time.monotonic()

            try:
                ok, buf = cv2.imencode(
                    ".jpg", frame,
                    [cv2.IMWRITE_JPEG_QUALITY, _WS_STREAM_QUAL],
                )
                if ok:
                    ws_server.send_frame(self.session_id, buf.tobytes())
            except Exception as e:
                logger.debug("WS relay encode error: %s", e)

    # ── Stop ──────────────────────────────────────────────────────────────────

    def stop(self) -> None:
        logger.info("MultiProcessPipelineRunner stopping")
        self._stop_event.set()

        # Stop all workers
        self._autoscaler.stop_all()
        for nid, workers in self._workers.items():
            for w in workers:
                try:
                    w.stop(timeout=5.0)
                except Exception as e:
                    logger.warning("Worker %s stop error: %s", nid, e)

        # Wait for threads
        for t in (self._monitor_thread, self._ws_relay_thread):
            if t and t.is_alive():
                t.join(timeout=3.0)

        # Release PortBuses (creator side → unlink=True)
        for bus in self._buses.values():
            try:
                bus.close(unlink=True)
            except Exception:
                pass

        ws_server.cleanup_session(self.session_id)
        logger.info("MultiProcessPipelineRunner stopped")

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        # Workers stats
        for nid, workers in self._workers.items():
            if nid in self._autoscaler._nodes:
                result[nid] = self._autoscaler.get_stats().get(nid, {})
            else:
                wstats = [w.get_stats().to_dict() for w in workers]
                result[nid] = {
                    "workers": len(workers),
                    "buffer_depth": 0,
                    "per_worker": wstats,
                }
        # Bus depths
        bus_depths = {}
        for eid, bus in self._buses.items():
            bus_depths[eid] = bus.get_buffer_depth()
        result["__bus_depths__"] = bus_depths
        return result

    def request_stop(self) -> None:
        self._stop_event.set()
