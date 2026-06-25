"""Track Database — per-session in-memory store of tracked object records.

Connects after object_tracker. For every tracked detection (track_id >= 0) it
maintains a record with:
  - class name, age (frames alive), cumulative travel distance (px)
  - rolling position history (last N frames)
  - last known bounding box
  - a lightweight BGR colour histogram as a visual signature

Writes ctx.metadata["track_db"] = { track_id: {...} } every frame so downstream
nodes (draw_bbox, counter, python_function) can access per-track stats.

This node passes frame and detections through unchanged.
"""
import logging
import cv2
import numpy as np
from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext
from engine.core.node_registry import register

logger = logging.getLogger(__name__)


@register("track_db")
class TrackDatabaseNode(BaseNode):
    def initialize(self):
        self._max_tracks    = int(self.config.get("max_tracks", 200))
        self._history_len   = int(self.config.get("history_frames", 30))
        self._draw_trails   = bool(self.config.get("draw_trails", True))
        self._db: dict[int, dict] = {}   # track_id → record
        logger.info("[TrackDB] node %s — max_tracks=%d history=%d",
                    self.node_id, self._max_tracks, self._history_len)

    def process(self, ctx: FrameContext) -> FrameContext:
        now = ctx.frame_number

        for det in ctx.detections:
            tid = det.track_id
            if tid < 0:
                continue

            cx, cy = det.center

            if tid not in self._db:
                if len(self._db) >= self._max_tracks:
                    oldest = min(self._db, key=lambda k: self._db[k]["last_frame"])
                    del self._db[oldest]
                self._db[tid] = {
                    "class_name":  det.class_name,
                    "class_id":    det.class_id,
                    "first_frame": now,
                    "last_frame":  now,
                    "positions":   [],
                    "distance":    0.0,
                    "color_hist":  None,
                }

            rec = self._db[tid]
            rec["last_frame"]  = now
            rec["class_name"]  = det.class_name
            rec["bbox"]        = (det.x1, det.y1, det.x2, det.y2)
            rec["confidence"]  = det.confidence

            positions = rec["positions"]
            if positions:
                dx = cx - positions[-1][0]
                dy = cy - positions[-1][1]
                rec["distance"] += (dx * dx + dy * dy) ** 0.5
            positions.append((int(cx), int(cy)))
            if len(positions) > self._history_len:
                positions.pop(0)

            # Update colour histogram from the crop (lightweight visual signature)
            if ctx.frame is not None:
                rec["color_hist"] = self._compute_hist(ctx.frame, det)

        # Evict stale tracks (not seen for > history_len frames)
        stale = [tid for tid, rec in self._db.items()
                 if now - rec["last_frame"] > self._history_len]
        for tid in stale:
            del self._db[tid]

        # Draw motion trails on frame
        if self._draw_trails and ctx.frame is not None:
            self._draw_trail_lines(ctx.frame)

        # Publish summary to metadata bus
        ctx.metadata["track_db"] = {
            tid: {
                "class":       rec["class_name"],
                "age_frames":  rec["last_frame"] - rec["first_frame"] + 1,
                "distance_px": round(rec["distance"], 1),
                "pos":         rec["positions"][-1] if rec["positions"] else None,
                "bbox":        rec.get("bbox"),
                "confidence":  rec.get("confidence", 0.0),
            }
            for tid, rec in self._db.items()
        }
        return ctx

    def _compute_hist(self, frame: np.ndarray, det) -> np.ndarray | None:
        """Compute a compact 8-bin BGR histogram for the detection crop."""
        x1, y1 = max(0, int(det.x1)), max(0, int(det.y1))
        x2, y2 = min(frame.shape[1], int(det.x2)), min(frame.shape[0], int(det.y2))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        if crop.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        hist = []
        for ch in range(3):
            h = cv2.calcHist([crop], [ch], None, [8], [0, 256])
            hist.append(cv2.normalize(h, h).flatten())
        return np.concatenate(hist)

    def _draw_trail_lines(self, frame: np.ndarray):
        for tid, rec in self._db.items():
            pts = rec["positions"]
            if len(pts) < 2:
                continue
            # Colour derived from track_id for consistency
            hue = (tid * 37) % 180
            col_hsv = np.uint8([[[hue, 220, 220]]])
            color = tuple(int(x) for x in cv2.cvtColor(col_hsv, cv2.COLOR_HSV2BGR)[0][0])
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                thickness = max(1, int(alpha * 3))
                cv2.line(frame, pts[i - 1], pts[i], color, thickness)
