"""Main frame loop — calls each node's process() in topological order."""
from __future__ import annotations
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext
from engine.streaming import ws_server

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Max FPS for per-node preview frames sent to the frontend
_NODE_PREVIEW_FPS     = 10
_STATS_WRITE_INTERVAL = 5.0  # seconds between stats file writes
_NODE_PREVIEW_QUALITY = 65
_AUTO_STREAM_FPS      = 15   # fallback auto-stream when no stream_viewer in pipeline
_AUTO_STREAM_QUALITY  = 75
_NODE_DATA_FPS        = 5    # JSON metadata preview for non-frame outputs
_VIZ_FPS              = 5    # max show_image/show_text updates per second per node

# Detection overlay palette (matches stream_viewer.py)
_COLORS = [
    (56, 182, 255), (255, 100, 56), (56, 255, 100), (255, 220, 56),
    (200, 56, 255), (56, 230, 230), (255, 56, 150), (130, 255, 56),
    (255, 160, 0),  (0, 160, 255),
]


def _safe(v):
    """Make a value JSON-serialisable."""
    if isinstance(v, np.ndarray):
        if v.ndim == 0:
            return float(v)
        if v.size <= 20:
            return v.tolist()
        return {"shape": list(v.shape), "dtype": str(v.dtype),
                "preview": v.flat[:6].tolist(), "norm": float(np.linalg.norm(v))}
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    if isinstance(v, list):
        return [_safe(x) for x in v]
    if isinstance(v, dict):
        return {k: _safe(vv) for k, vv in v.items()}
    return v


def _node_data_summary(node: "BaseNode", ctx: "FrameContext") -> dict | None:
    """Return a JSON-serialisable summary of what this node produced, or None."""
    nt = node.node_type
    m  = ctx.metadata
    d  = ctx.detections

    # ── Face ID ───────────────────────────────────────────────────────────────
    if nt == "face_detect":
        lms = m.get("face_landmarks", [])
        return {
            "faces_detected": len(d),
            "detections": [
                {"bbox": [round(det.x1), round(det.y1), round(det.x2), round(det.y2)],
                 "confidence": round(det.confidence, 3)}
                for det in d[:8]
            ],
            "landmarks": [
                [[round(float(pt[0]), 1), round(float(pt[1]), 1)] for pt in lm]
                for lm in lms[:4]
            ],
        }

    if nt == "embedding":
        embs = m.get("face_embeddings", [])
        summaries = []
        for e in embs[:4]:
            e_arr = np.array(e, dtype=np.float32)
            summaries.append({
                "shape":   list(e_arr.shape),
                "norm":    round(float(np.linalg.norm(e_arr)), 4),
                "preview": [round(float(x), 4) for x in e_arr[:6]],
            })
        return {"embedding_count": len(embs), "embeddings": summaries}

    if nt == "face_db":
        return {
            "matched":    m.get("face_match", False),
            "similarity": round(float(m.get("face_similarity", 0.0)), 4),
            "name":       m.get("face_match_name", ""),
            "db_count":   m.get("face_db_count", 0),
            "threshold":  node.config.get("threshold", 0.5),
        }

    # ── Spatial / tracking ────────────────────────────────────────────────────
    if nt == "object_tracker":
        return {
            "tracked_count": len(d),
            "tracks": [
                {"id": det.track_id, "class": det.class_name,
                 "conf": round(det.confidence, 3),
                 "bbox": [round(det.x1), round(det.y1), round(det.x2), round(det.y2)]}
                for det in d[:8]
            ],
        }

    if nt == "track_db":
        db = m.get("track_db", {})
        return {
            "active_tracks": len(db),
            "tracks": [
                {"id": tid,
                 "class": info.get("class_name", "?"),
                 "age": info.get("age", 0),
                 "pos": info.get("position_history", [[]])[-1] if info.get("position_history") else []}
                for tid, info in list(db.items())[:8]
            ],
        }

    if nt == "counter":
        node_key = f"counter_{node.node_id}"
        return {"count": m.get(node_key, 0)}

    # ── Detection nodes ───────────────────────────────────────────────────────
    if nt in ("nms", "filter", "draw_bbox"):
        return {
            "detection_count": len(d),
            "detections": [
                {"class": det.class_name, "conf": round(det.confidence, 3),
                 "bbox": [round(det.x1), round(det.y1), round(det.x2), round(det.y2)],
                 **({"id": det.track_id} if det.track_id >= 0 else {})}
                for det in d[:8]
            ],
        }

    if nt == "crop_bbox":
        count = m.get("crop_count", 0)
        ious  = m.get("crop_ious", [])
        # Flatten upper-triangle pairs for display (skip diagonal)
        pairs = []
        for i in range(len(ious)):
            for j in range(i + 1, len(ious)):
                pairs.append({"i": i, "j": j, "iou": ious[i][j]})
        return {
            "crop_count": count,
            "image_size": node.config.get("image_size", 112),
            "iou_pairs": pairs[:12],  # cap display at 12 pairs
        }

    if nt == "model_inference":
        raw = m.get("raw_output")
        if raw is not None and isinstance(raw, np.ndarray):
            return {"raw_shape": list(raw.shape), "dtype": str(raw.dtype)}
        return None

    # ── Vision (OpenCV) ───────────────────────────────────────────────────────
    if nt == "corner_detect":
        pts = m.get("corner_points", [])
        return {"corner_count": len(pts), "corners": [[round(float(p[0])), round(float(p[1]))] for p in pts[:12]]}

    return None


def _overlay_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes + labels onto a copy of frame."""
    out = frame.copy()
    if out.ndim == 2 or (out.ndim == 3 and out.shape[2] == 1):
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    for det in detections:
        color = _COLORS[det.class_id % len(_COLORS)]
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        if det.track_id >= 0:
            label = f"#{det.track_id} {label}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, cv2.FILLED)
        cv2.putText(out, label, (x1 + 3, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


class PipelineRunner:
    def __init__(self, nodes: list[BaseNode], session_id: str):
        self.nodes      = nodes
        self.session_id = session_id
        self._stop_event  = threading.Event()
        self._reload_flag = threading.Event()
        self._frame_number: int = 0
        # Per-node throttle: last timestamp a preview was sent
        self._preview_last: dict[str, float] = {}
        self._data_last:    dict[str, float] = {}
        self._viz_last:     dict[str, float] = {}
        self._auto_stream_last: float = 0.0
        self._stats_last: float = 0.0
        self._stats_path: Path | None = (
            Path(os.environ["CVFLOW_STATS_PATH"])
            if "CVFLOW_STATS_PATH" in os.environ else None
        )
        # Detect whether a stream_viewer node is in the pipeline
        self._has_stream_viewer = any(
            type(n).__name__ == "StreamViewerNode" for n in nodes
        )

    def request_stop(self) -> None:
        self._stop_event.set()

    def request_reload(self) -> None:
        """Called from SIGUSR1 handler; engine reloads model weights after current frame."""
        self._reload_flag.set()

    # ── Per-node live preview ─────────────────────────────────────────────────

    def _try_send_node_preview(self, node: BaseNode, ctx: FrameContext) -> None:
        """
        If a frontend client is watching this node's output, encode the current
        frame as a low-quality JPEG (with detections overlaid) and push it.
        Throttled to _NODE_PREVIEW_FPS.
        """
        if not ws_server.is_node_watched(self.session_id, node.node_id):
            return

        now  = time.monotonic()
        last = self._preview_last.get(node.node_id, 0.0)
        if now - last < 1.0 / _NODE_PREVIEW_FPS:
            return
        self._preview_last[node.node_id] = now

        try:
            frame = ctx.ensure_cpu().copy()
            if ctx.detections:
                frame = _overlay_detections(frame, ctx.detections)
            ok, buf = cv2.imencode(
                ".jpg", frame,
                [cv2.IMWRITE_JPEG_QUALITY, _NODE_PREVIEW_QUALITY],
            )
            if ok:
                ws_server.send_node_preview(self.session_id, node.node_id, buf.tobytes())
        except Exception as exc:
            logger.debug("Preview encode failed for node %s: %s", node.node_id, exc)

    def _try_send_node_data(self, node: BaseNode, ctx: FrameContext) -> None:
        """Send a JSON metadata snapshot for nodes whose output isn't just a frame."""
        if not ws_server.is_node_watched(self.session_id, node.node_id):
            return
        now  = time.monotonic()
        last = self._data_last.get(node.node_id, 0.0)
        if now - last < 1.0 / _NODE_DATA_FPS:
            return
        self._data_last[node.node_id] = now
        try:
            payload = _node_data_summary(node, ctx)
            if payload:
                ws_server.send_event(self.session_id, {
                    "type":      "node_data",
                    "node_id":   node.node_id,
                    "node_type": node.node_type,
                    "data":      payload,
                })
        except Exception as exc:
            logger.debug("node_data summary failed for %s: %s", node.node_id, exc)

    def _try_send_viz(self, node: BaseNode) -> None:
        """Send show_image/show_text data collected during this frame's process() call."""
        items = getattr(node, "_latest_viz", None)
        if not items:
            return
        now  = time.monotonic()
        last = self._viz_last.get(node.node_id, 0.0)
        if now - last < 1.0 / _VIZ_FPS:
            return
        self._viz_last[node.node_id] = now
        try:
            ws_server.send_event(self.session_id, {
                "type":    "node_viz",
                "node_id": node.node_id,
                "items":   items,
            })
        except Exception as exc:
            logger.debug("viz send error for node %s: %s", node.node_id, exc)

    def _try_auto_stream(self, ctx: FrameContext) -> None:
        """Send the final frame to the main stream channel when no stream_viewer exists.
        Allows pipelines without an explicit stream_viewer to still show output."""
        now = time.monotonic()
        if now - self._auto_stream_last < 1.0 / _AUTO_STREAM_FPS:
            return
        self._auto_stream_last = now
        try:
            frame = ctx.ensure_cpu().copy()
            if ctx.detections:
                frame = _overlay_detections(frame, ctx.detections)
            ok, buf = cv2.imencode(
                ".jpg", frame,
                [cv2.IMWRITE_JPEG_QUALITY, _AUTO_STREAM_QUALITY],
            )
            if ok:
                ws_server.send_frame(self.session_id, buf.tobytes())
        except Exception as exc:
            logger.debug("Auto-stream failed: %s", exc)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Blocking frame loop. Returns when stop is requested or source exhausted."""
        logger.info("Pipeline started with %d nodes", len(self.nodes))
        try:
            while not self._stop_event.is_set():
                ctx = FrameContext(
                    frame=None,
                    frame_number=self._frame_number,
                    timestamp=time.time(),
                    session_id=self.session_id,
                )

                stop_pipeline = False
                for node in self.nodes:
                    try:
                        import time as _time
                        _t0 = _time.perf_counter()
                        result = node.process(ctx)
                        node._record_timing((_time.perf_counter() - _t0) * 1000)
                        if result is None:
                            # Node dropped this frame — stop processing remaining nodes
                            stop_pipeline = False
                            break
                        ctx = result
                        self._try_send_node_preview(node, ctx)
                        self._try_send_node_data(node, ctx)
                        self._try_send_viz(node)
                    except StopIteration:
                        stop_pipeline = True
                        break
                    except Exception as e:
                        node._error_count += 1
                        logger.error(
                            "Node %s error on frame %d: %s",
                            node.node_id, self._frame_number, e,
                            exc_info=True,
                        )

                if stop_pipeline:
                    logger.info("Input source exhausted — pipeline complete")
                    break

                # Auto-stream fallback: show output even without stream_viewer
                if not self._has_stream_viewer and ctx.frame is not None:
                    self._try_auto_stream(ctx)

                self._frame_number += 1
                self._try_write_stats()

                if self._reload_flag.is_set():
                    self._reload_flag.clear()
                    self._do_reload()

        finally:
            logger.info("Pipeline stopping, calling teardown on %d nodes", len(self.nodes))
            for node in reversed(self.nodes):
                try:
                    node.teardown()
                except Exception as e:
                    logger.warning("Teardown error for node %s: %s", node.node_id, e)

            ws_server.cleanup_session(self.session_id)

    def _try_write_stats(self) -> None:
        if not self._stats_path:
            return
        now = time.monotonic()
        if now - self._stats_last < _STATS_WRITE_INTERVAL:
            return
        self._stats_last = now
        try:
            stats = {n.node_id: n.get_stats() for n in self.nodes}
            self._stats_path.write_text(json.dumps(stats, default=str), encoding="utf-8")
        except Exception:
            pass

    def _do_reload(self) -> None:
        """Reload model weights in any ModelInference nodes without stopping the loop."""
        from engine.nodes.processing.model_inference import ModelInferenceNode
        for node in self.nodes:
            if isinstance(node, ModelInferenceNode):
                try:
                    node.reload_model()
                    logger.info("Hot-reloaded model in node %s", node.node_id)
                except Exception as e:
                    logger.error("Hot-reload failed for %s: %s", node.node_id, e)
