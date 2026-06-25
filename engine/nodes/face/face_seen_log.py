"""Face Seen Log node — records every recognised (or unknown) face appearance.

Stores a rolling log in memory and on disk. Emits WS events so the frontend
"Seen Faces" panel can display thumbnails and allow enrollment of Unknowns.
"""
from __future__ import annotations
import base64
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext
from engine.core.node_registry import register
from engine.streaming import ws_server

logger = logging.getLogger(__name__)


@register("face_seen_log")
class FaceSeenLogNode(BaseNode):
    """Log and stream every detected face appearance with identity, thumbnail, and metadata."""

    def initialize(self):
        self._log_unknowns   = bool(self.config.get("log_unknowns", True))
        self._interval_s     = float(self.config.get("snapshot_interval_s", 5.0))
        self._max_entries    = int(self.config.get("max_log_entries", 500))
        self._save_crops     = bool(self.config.get("save_crops", True))
        self._ws_push_every  = float(self.config.get("ws_push_interval_s", 1.0))

        raw_path = self.config.get("db_path", "storage/facedb")
        self._crops_dir = Path(raw_path) / "seen_crops"
        self._log_path  = Path(raw_path) / "seen_log.json"
        self._crops_dir.mkdir(parents=True, exist_ok=True)

        # In-memory log (newest at end)
        self._log: list[dict] = self._load_log()
        # Throttle: identity → last snapshot time
        self._last_snap: dict[str, float] = {}
        self._last_ws_push = 0.0
        logger.info("[FaceSeenLog] %d entries loaded, logging to %s",
                    len(self._log), self._log_path)

    def _load_log(self) -> list[dict]:
        if self._log_path.exists():
            try:
                return json.loads(self._log_path.read_text())[-self._max_entries:]
            except Exception:
                pass
        return []

    def _save_log(self):
        try:
            self._log_path.write_text(json.dumps(self._log[-self._max_entries:], indent=2))
        except Exception as e:
            logger.warning("[FaceSeenLog] save failed: %s", e)

    def process(self, ctx: FrameContext) -> FrameContext:
        matches = ctx.metadata.get("face_matches", [])
        aligned = ctx.metadata.get("aligned_faces", [])
        now     = time.time()
        new_entries = []

        for i, match in enumerate(matches):
            name   = match.get("name", "Unknown")
            in_db  = match.get("in_db", False)

            if not in_db and not self._log_unknowns:
                continue

            key = match.get("identity_id") or f"unk_{i}"
            if now - self._last_snap.get(key, 0) < self._interval_s:
                continue
            self._last_snap[key] = now

            crop = aligned[i] if i < len(aligned) else None
            crop_path = ""
            crop_b64  = ""

            if crop is not None:
                ts_str = str(int(now * 1000))
                fname  = f"{key}_{ts_str}.jpg"
                if self._save_crops:
                    crop_path = str(self._crops_dir / fname)
                    cv2.imwrite(crop_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 70])
                crop_b64 = base64.b64encode(buf.tobytes()).decode()

            entry = {
                "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%S"),
                "identity_id": match.get("identity_id"),
                "name":        name,
                "similarity":  match.get("similarity", 0.0),
                "in_db":       in_db,
                "bbox":        [int(ctx.detections[i].x1), int(ctx.detections[i].y1),
                                int(ctx.detections[i].x2), int(ctx.detections[i].y2)]
                               if i < len(ctx.detections) else [],
                "crop_path":   crop_path,
                "crop_b64":    crop_b64,
                "frame_number": ctx.frame_number,
            }
            self._log.append(entry)
            new_entries.append(entry)

        if len(self._log) > self._max_entries:
            self._log = self._log[-self._max_entries:]

        # Write log to disk on any new entry
        if new_entries:
            self._save_log()

        # Push recent entries to frontend via WS (rate-limited)
        if now - self._last_ws_push >= self._ws_push_every:
            self._last_ws_push = now
            recent = [
                {k: v for k, v in e.items() if k != "crop_b64"}   # omit large b64 in list
                for e in self._log[-20:]
            ]
            ws_server.send_event(ctx.session_id, {
                "type":    "face_seen_update",
                "entries": recent,
                "total":   len(self._log),
            })
            # Also push new entries with their crops for the notification toasts
            for e in new_entries:
                ws_server.send_event(ctx.session_id, {
                    "type":       "face_seen",
                    "name":       e["name"],
                    "in_db":      e["in_db"],
                    "similarity": e["similarity"],
                    "crop_b64":   e["crop_b64"],
                    "timestamp":  e["timestamp"],
                    "identity_id": e["identity_id"],
                })

        ctx.metadata["face_seen_events"] = new_entries
        return ctx

    def teardown(self):
        self._save_log()

    def get_log(self, limit: int = 100) -> list[dict]:
        return [
            {k: v for k, v in e.items() if k != "crop_b64"}
            for e in self._log[-limit:]
        ]
