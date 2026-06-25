"""Face Enrollment node — registers new faces into the FaceVectorDB.

Modes:
  manual      — enrollment triggered via Properties panel button or REST API
                (sends face_enroll_prompt WS event; waits for REST confirm)
  auto        — any Unknown face with quality score > min_quality creates a new identity
  interactive — same as manual but prompts on every Unknown face
"""
from __future__ import annotations
import base64
import json
import logging
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext
from engine.core.node_registry import register
from engine.streaming import ws_server

logger = logging.getLogger(__name__)


@register("face_enroll")
class FaceEnrollNode(BaseNode):
    """Enroll face identities into a connected FaceVectorDBNode."""

    def initialize(self):
        self._mode            = self.config.get("mode", "manual")
        self._auto_prefix     = self.config.get("auto_label_prefix", "Person")
        self._min_quality     = float(self.config.get("min_quality_score", 0.6))
        self._timeout_s       = float(self.config.get("timeout_s", 10.0))
        self._auto_counter    = 1
        self._last_prompt_t   = 0.0
        self._prompt_cooldown = 3.0

        # File-based IPC paths (shared with backend REST API)
        db_path = self.config.get("db_path", "storage/facedb")
        self._db_dir          = Path(db_path)
        self._db_dir.mkdir(parents=True, exist_ok=True)
        self._confirms_file   = self._db_dir / ".pending_confirms.json"
        self._trigger_file    = self._db_dir / ".manual_trigger"

        # Pending prompts we've sent: pending_id → {embedding, crop, expires_at}
        self._pending: dict[str, dict] = {}
        logger.info("[FaceEnroll] mode=%s db=%s", self._mode, self._db_dir)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _read_confirms(self) -> dict:
        """Read and clear the file-based confirmation queue."""
        if not self._confirms_file.exists():
            return {}
        try:
            data = json.loads(self._confirms_file.read_text())
            self._confirms_file.unlink(missing_ok=True)
            return data
        except Exception:
            return {}

    def _check_trigger(self) -> bool:
        """Return True if a manual trigger file was written since last check."""
        if not self._trigger_file.exists():
            return False
        try:
            mtime = self._trigger_file.stat().st_mtime
            self._trigger_file.unlink(missing_ok=True)
            # Only act on triggers written within the last 10 s
            return (time.time() - mtime) < 10.0
        except Exception:
            return False

    def _find_db_node(self, ctx: FrameContext):
        return ctx.metadata.get("_face_vector_db_ref")

    # ── process ──────────────────────────────────────────────────────────────

    def process(self, ctx: FrameContext) -> FrameContext:
        embeddings = ctx.metadata.get("face_embeddings", [])
        matches    = ctx.metadata.get("face_matches", [])
        aligned    = ctx.metadata.get("aligned_faces", [])
        db_node    = self._find_db_node(ctx)
        now        = time.time()

        # Expire stale pending entries
        self._pending = {k: v for k, v in self._pending.items() if v["expires_at"] > now}

        # Process confirmations from REST API
        confirms = self._read_confirms()
        for pending_id, conf in confirms.items():
            pending = self._pending.pop(pending_id, None)
            if pending and db_node is not None:
                db_node.enroll(conf["name"], pending["embedding"],
                               attributes=conf.get("attributes", {}),
                               crop_img=pending.get("crop"))
                ws_server.send_event(ctx.session_id, {
                    "type":       "face_enrolled",
                    "name":       conf["name"],
                    "pending_id": pending_id,
                })
                logger.info("[FaceEnroll] enrolled '%s' (pending_id=%s)", conf["name"], pending_id)

        if not embeddings:
            return ctx

        if self._mode == "auto":
            self._process_auto(ctx, embeddings, matches, aligned, db_node, now)
        elif self._mode in ("manual", "interactive"):
            manual_trigger = self._check_trigger()
            self._process_manual(ctx, embeddings, matches, aligned, db_node, now,
                                 force=manual_trigger)

        ctx.metadata["_face_enroll_node"] = self
        return ctx

    # ── enrollment modes ─────────────────────────────────────────────────────

    def _process_auto(self, ctx, embeddings, matches, aligned, db_node, now):
        for i, match in enumerate(matches):
            if match.get("in_db"):
                continue
            if i >= len(embeddings):
                continue
            emb  = np.array(embeddings[i], dtype=np.float32)
            conf = ctx.detections[i].confidence if i < len(ctx.detections) else 0.0
            if conf < self._min_quality:
                continue
            name = f"{self._auto_prefix}_{self._auto_counter}"
            crop = aligned[i] if i < len(aligned) else None
            if db_node is not None:
                db_node.enroll(name, emb, crop_img=crop)
            self._auto_counter += 1
            logger.info("[FaceEnroll] auto-enrolled '%s'", name)

    def _process_manual(self, ctx, embeddings, matches, aligned, db_node, now, force=False):
        if self._mode == "manual" and not force:
            return
        if now - self._last_prompt_t < self._prompt_cooldown:
            return

        for i, match in enumerate(matches):
            # In manual mode, prompt for any face when triggered
            # In interactive mode, only prompt for unknowns
            if self._mode == "interactive" and match.get("in_db") and not force:
                continue
            if i >= len(embeddings):
                continue
            conf = ctx.detections[i].confidence if i < len(ctx.detections) else 0.0
            if conf < self._min_quality and not force:
                continue

            emb    = np.array(embeddings[i], dtype=np.float32)
            crop   = aligned[i] if i < len(aligned) else None
            pid    = str(uuid.uuid4())[:8]
            self._pending[pid] = {
                "embedding":  emb,
                "crop":       crop,
                "expires_at": now + self._timeout_s,
            }

            crop_b64 = ""
            if crop is not None:
                _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                crop_b64 = base64.b64encode(buf.tobytes()).decode()

            ws_server.send_event(ctx.session_id, {
                "type":           "face_enroll_prompt",
                "pending_id":     pid,
                "crop_b64":       crop_b64,
                "suggested_name": match.get("name", "Unknown"),
                "similarity":     match.get("similarity", 0.0),
                "expires_in_s":   self._timeout_s,
            })
            logger.info("[FaceEnroll] prompt sent pending_id=%s", pid)
            self._last_prompt_t = now
            break   # one face at a time
