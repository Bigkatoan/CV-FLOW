"""Face Vector Database node — persistent in-memory store of face identities.

Storage layout (relative to db_path):
  identities.json       — identity metadata list
  embeddings/{id}.npy   — stacked (K × 512) embedding matrix per identity
  crops/{id}_{n}.jpg    — face crop thumbnails (optional)

Receives embeddings from face_embed, performs cosine matching, updates
ctx.detections[i].class_name with the matched identity name.
"""
from __future__ import annotations
import json
import logging
import os
import shutil
import threading
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


@register("face_vector_db")
class FaceVectorDBNode(BaseNode):
    """Cosine-matching face identity database with disk persistence."""

    def initialize(self):
        raw_path = self.config.get("db_path", "storage/facedb")
        self._db_dir  = Path(raw_path)
        self._thresh  = float(self.config.get("similarity_threshold", 0.35))
        self._max_emb = int(self.config.get("max_embeddings_per_id", 10))
        self._top_k   = int(self.config.get("return_top_k", 3))
        self._lock    = threading.Lock()
        self._last_event_t = 0.0

        # In-memory store: id → {meta, embeddings ndarray (K×512)}
        self._identities: dict[str, dict] = {}
        self._embeddings: dict[str, np.ndarray] = {}

        self._db_dir.mkdir(parents=True, exist_ok=True)
        (self._db_dir / "embeddings").mkdir(exist_ok=True)
        (self._db_dir / "crops").mkdir(exist_ok=True)

        self._load_db()
        logger.info("[FaceVectorDB] %d identit(ies) loaded from %s",
                    len(self._identities), self._db_dir)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_db(self):
        meta_path = self._db_dir / "identities.json"
        if not meta_path.exists():
            return
        try:
            entries = json.loads(meta_path.read_text())
            for entry in entries:
                iid = entry["id"]
                self._identities[iid] = entry
                emb_path = self._db_dir / "embeddings" / f"{iid}.npy"
                if emb_path.exists():
                    self._embeddings[iid] = np.load(str(emb_path))
        except Exception as e:
            logger.error("[FaceVectorDB] Load failed: %s", e)

    def _save_db(self):
        try:
            meta_path = self._db_dir / "identities.json"
            meta_path.write_text(json.dumps(list(self._identities.values()), indent=2))
            for iid, embs in self._embeddings.items():
                np.save(str(self._db_dir / "embeddings" / f"{iid}.npy"), embs)
        except Exception as e:
            logger.error("[FaceVectorDB] Save failed: %s", e)

    # ── Public API (called by face_enroll & REST API) ─────────────────────────

    def enroll(self, name: str, embedding: np.ndarray,
               attributes: dict | None = None,
               crop_img: np.ndarray | None = None) -> str:
        """Add a new identity or add an embedding to an existing name. Returns identity_id."""
        with self._lock:
            # Find existing identity by name
            iid = next((k for k, v in self._identities.items() if v["name"] == name), None)
            if iid is None:
                iid = str(uuid.uuid4())
                self._identities[iid] = {
                    "id":          iid,
                    "name":        name,
                    "enrolled_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "last_seen":   None,
                    "attributes":  attributes or {},
                    "crop_count":  0,
                }

            # Append embedding (keep last max_embeddings_per_id)
            existing = self._embeddings.get(iid)
            new_emb  = embedding.reshape(1, -1).astype(np.float32)
            if existing is None:
                self._embeddings[iid] = new_emb
            else:
                stacked = np.vstack([existing, new_emb])
                if len(stacked) > self._max_emb:
                    stacked = stacked[-self._max_emb:]
                self._embeddings[iid] = stacked

            # Save crop thumbnail
            if crop_img is not None:
                n = self._identities[iid]["crop_count"]
                crop_path = self._db_dir / "crops" / f"{iid}_{n}.jpg"
                cv2.imwrite(str(crop_path), crop_img)
                self._identities[iid]["crop_count"] = n + 1

            self._save_db()
            logger.info("[FaceVectorDB] Enrolled '%s' (id=%s, total_emb=%d)",
                        name, iid, len(self._embeddings[iid]))
            return iid

    def delete_identity(self, identity_id: str) -> bool:
        with self._lock:
            if identity_id not in self._identities:
                return False
            del self._identities[identity_id]
            self._embeddings.pop(identity_id, None)
            emb_path = self._db_dir / "embeddings" / f"{identity_id}.npy"
            if emb_path.exists():
                emb_path.unlink()
            self._save_db()
            return True

    def list_identities(self) -> list[dict]:
        with self._lock:
            return [
                {**meta, "embedding_count": len(self._embeddings.get(iid, []))}
                for iid, meta in self._identities.items()
            ]

    # ── Matching ──────────────────────────────────────────────────────────────

    def _match_one(self, emb: np.ndarray) -> dict:
        """Return best match dict. Thread-safe."""
        best = {"identity_id": None, "name": "Unknown", "similarity": 0.0,
                "in_db": False, "attributes": {}, "top_k": []}
        if not self._embeddings:
            return best

        emb = emb / (np.linalg.norm(emb) + 1e-8)
        scores: list[tuple[float, str]] = []

        with self._lock:
            for iid, emb_mat in self._embeddings.items():
                sims = emb_mat @ emb          # (K,)
                best_sim = float(sims.max())
                scores.append((best_sim, iid))

        scores.sort(key=lambda x: x[0], reverse=True)
        top_k = scores[:self._top_k]

        best["top_k"] = [
            {"identity_id": iid,
             "name": self._identities[iid]["name"],
             "similarity": round(sim, 4)}
            for sim, iid in top_k if iid in self._identities
        ]

        if top_k and top_k[0][0] >= (1.0 - self._thresh):
            best_sim, best_iid = top_k[0]
            meta = self._identities.get(best_iid, {})
            best.update({
                "identity_id": best_iid,
                "name":        meta.get("name", "Unknown"),
                "similarity":  round(best_sim, 4),
                "in_db":       True,
                "attributes":  meta.get("attributes", {}),
            })
            with self._lock:
                self._identities[best_iid]["last_seen"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        return best

    # ── Process ───────────────────────────────────────────────────────────────

    def process(self, ctx: FrameContext) -> FrameContext:
        embeddings = ctx.metadata.get("face_embeddings", [])
        matches    = []

        for i, emb in enumerate(embeddings):
            match = self._match_one(np.array(emb, dtype=np.float32))
            matches.append(match)

            # Update detection class_name + confidence with identity
            if i < len(ctx.detections):
                ctx.detections[i].class_name = match["name"]
                ctx.detections[i].confidence = match["similarity"]

        ctx.metadata["face_matches"] = matches
        # Expose self so face_enroll and face_seen_log can call enroll()
        ctx.metadata["_face_vector_db_ref"] = self

        # Emit DB summary event at most once per second
        now = time.time()
        if now - self._last_event_t >= 1.0:
            self._last_event_t = now
            ws_server.send_event(ctx.session_id, {
                "type":       "face_db_status",
                "identity_count": len(self._identities),
                "matches":    matches,
            })

        return ctx

    def teardown(self):
        self._save_db()
