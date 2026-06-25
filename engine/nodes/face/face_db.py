"""Face DB node — save and compare face embeddings against a persistent database.

Inputs (via ctx.metadata):
  face_embeddings     — list of N numpy embeddings (from embedding node)
  face_has_detection  — active gate (idle if explicitly False)
  face_db_save        — save trigger (True = save current embeddings to DB)
  face_db_compare     — compare trigger (True = compare against DB; default True)

Outputs (via ctx.metadata):
  face_match          — bool: True if any embedding matched above threshold
  face_similarity     — float: highest cosine similarity found
  face_match_name     — str: name of the best-matching identity
"""
from __future__ import annotations
import json
import logging
import uuid
from pathlib import Path

import numpy as np

from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext
from engine.core.node_registry import register

logger = logging.getLogger(__name__)


@register("face_db")
class FaceDBNode(BaseNode):
    """Persistent face embedding database with save and compare triggers."""

    def initialize(self):
        self._threshold = float(self.config.get("threshold", 0.5))
        self._name      = self.config.get("name", "Person")
        self._max_save  = int(self.config.get("max_save", 10))
        db_path = Path(self.config.get("db_path", "storage/facedb"))
        self._emb_dir   = db_path / "embeddings"
        self._db_file   = db_path / "db.json"
        self._trigger_file = db_path / ".save_trigger"
        db_path.mkdir(parents=True, exist_ok=True)
        self._emb_dir.mkdir(parents=True, exist_ok=True)
        self._entries: list[dict] = self._load_db()

    # ── DB persistence ─────────────────────────────────────────────────────────

    def _load_db(self) -> list[dict]:
        if self._db_file.exists():
            try:
                return json.loads(self._db_file.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("[FaceDB] Could not load db.json: %s", e)
        return []

    def _write_db(self):
        self._db_file.write_text(
            json.dumps(self._entries, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ── Save ───────────────────────────────────────────────────────────────────

    def _count_by_name(self, name: str) -> int:
        return sum(1 for e in self._entries if e.get("name") == name)

    def _save_embeddings(self, embeddings: list[np.ndarray]):
        saved = 0
        for emb in embeddings:
            if self._max_save > 0 and self._count_by_name(self._name) >= self._max_save:
                logger.info(
                    "[FaceDB] max_save=%d reached for '%s' — skipping remaining.",
                    self._max_save, self._name,
                )
                break
            eid   = uuid.uuid4().hex[:8]
            fname = f"{self._name}_{eid}.npy"
            np.save(str(self._emb_dir / fname), emb.astype(np.float32))
            self._entries.append({"name": self._name, "file": fname})
            saved += 1
        if saved:
            self._write_db()
        logger.info("[FaceDB] Saved %d/%d embedding(s) as '%s'", saved, len(embeddings), self._name)

    # ── Compare ────────────────────────────────────────────────────────────────

    def _compare(self, embeddings: list[np.ndarray]) -> tuple[bool, float, str]:
        """Cosine-similarity comparison against all stored embeddings."""
        db_embs: list[tuple[str, np.ndarray]] = []
        for entry in self._entries:
            path = self._emb_dir / entry["file"]
            if path.exists():
                try:
                    db_embs.append((entry["name"], np.load(str(path)).astype(np.float32)))
                except Exception:
                    pass

        if not db_embs:
            return False, 0.0, ""

        best_sim  = -1.0
        best_name = ""
        for q_emb in embeddings:
            norm_q = np.linalg.norm(q_emb)
            if norm_q < 1e-8:
                continue
            q = q_emb / norm_q
            for name, db_emb in db_embs:
                norm_d = np.linalg.norm(db_emb)
                if norm_d < 1e-8:
                    continue
                d   = db_emb / norm_d
                sim = float(np.dot(q, d))
                if sim > best_sim:
                    best_sim  = sim
                    best_name = name

        matched = best_sim >= self._threshold
        return matched, round(best_sim, 4), best_name

    # ── Process ────────────────────────────────────────────────────────────────

    def process(self, ctx: FrameContext) -> FrameContext:
        ctx.metadata["face_match"]       = False
        ctx.metadata["face_similarity"]  = 0.0
        ctx.metadata["face_match_name"]  = ""
        ctx.metadata["face_db_count"]    = len(self._entries)

        # Active gate
        if ctx.metadata.get("face_has_detection") is False:
            return ctx

        embeddings: list = ctx.metadata.get("face_embeddings", [])
        if not embeddings:
            return ctx

        # Enroll/save trigger — from enroll dot, save dot, or file-based IPC
        enroll = (
            ctx.metadata.get("face_db_enroll", False)
            or ctx.metadata.get("face_db_save", False)
        )
        if not enroll and self._trigger_file.exists():
            try:
                self._trigger_file.unlink()
                enroll = True
            except Exception:
                pass
        if enroll:
            self._save_embeddings(embeddings)

        # Compare trigger — defaults to True when face is detected
        compare = ctx.metadata.get("face_db_compare", True)
        if compare and self._entries:
            matched, similarity, name = self._compare(embeddings)
            ctx.metadata["face_match"]      = matched
            ctx.metadata["face_similarity"] = similarity
            ctx.metadata["face_match_name"] = name
            logger.debug(
                "[FaceDB] compare → matched=%s sim=%.3f name=%s",
                matched, similarity, name,
            )

        return ctx
