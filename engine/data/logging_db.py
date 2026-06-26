"""
LoggingDB — FIFO face logging database with cooldown-based deduplication.

Storage layout per collection:
  <storage_dir>/<name>/index.npy     — (N, 512) float32, L2-normalized (for cosine dedup)
  <storage_dir>/<name>/meta.json     — [{id, timestamp, ...}] in insertion order
  <storage_dir>/<name>/images/{id}.jpg — individual face JPEG crops

Deduplication logic:
  - On add(), cosine similarity is computed against all stored embeddings.
  - If the most similar entry exceeds sim_threshold AND was added within cooldown_sec,
    the new face is suppressed (returns added=False, reason="cooldown").
  - Otherwise the face is stored and, if count > max_faces, the oldest entry is evicted.

Thread safety: threading.RLock; write-through persistence (no background flush).
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np


class LoggingDB:
    """FIFO face logging database with cooldown-based dedup and size cap."""

    def __init__(self, name: str, storage_dir: str | Path, max_faces: int = 1000) -> None:
        self.name      = name
        self.max_faces = max_faces
        self._dir      = Path(storage_dir) / name
        self._img_dir  = self._dir / "images"
        self._lock     = threading.RLock()
        self._embeddings: np.ndarray | None = None  # (N, 512) float32, L2-normalized
        self._meta: list[dict[str, Any]] = []        # insertion-order list

        self._dir.mkdir(parents=True, exist_ok=True)
        self._img_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def can_add(
        self,
        embedding: np.ndarray,
        cooldown_sec: int = 300,
        sim_threshold: float = 0.6,
    ) -> tuple[bool, str]:
        """Check whether a face should be logged.

        Returns:
            (True, "ok")       — face is new or cooldown expired, proceed to add()
            (False, "cooldown") — a similar face was logged within cooldown_sec
            (False, "empty")   — no embeddings in DB yet (always allow)
        """
        with self._lock:
            if self._embeddings is None or len(self._embeddings) == 0:
                return True, "ok"

            q = _l2_normalize(np.array(embedding, dtype=np.float32))
            scores = self._embeddings @ q          # cosine similarity via dot product
            best   = int(np.argmax(scores))
            best_score = float(scores[best])

            if best_score < sim_threshold:
                return True, "ok"

            # Similar face found — check cooldown
            last_ts = self._meta[best].get("timestamp", 0.0)
            elapsed = time.time() - last_ts
            if elapsed < cooldown_sec:
                return False, "cooldown"
            return True, "ok"

    def add(
        self,
        embedding: np.ndarray,
        image: np.ndarray,
        metadata: dict | None = None,
    ) -> str:
        """Store a face crop with its embedding. Returns the new record id.

        Evicts the oldest entry if count would exceed max_faces.
        """
        import cv2

        record_id = str(uuid.uuid4())
        emb = _l2_normalize(np.array(embedding, dtype=np.float32))

        # Save JPEG
        img_path = self._img_dir / f"{record_id}.jpg"
        cv2.imwrite(str(img_path), image, [cv2.IMWRITE_JPEG_QUALITY, 80])

        entry = {
            "id":        record_id,
            "timestamp": time.time(),
            **(metadata or {}),
        }

        with self._lock:
            if self._embeddings is None:
                self._embeddings = emb.reshape(1, -1)
            else:
                self._embeddings = np.vstack([self._embeddings, emb.reshape(1, -1)])
            self._meta.append(entry)

            # Enforce max_faces FIFO eviction
            while len(self._meta) > self.max_faces:
                old = self._meta.pop(0)
                old_img = self._img_dir / f"{old['id']}.jpg"
                if old_img.exists():
                    old_img.unlink(missing_ok=True)
                self._embeddings = self._embeddings[1:]

            if len(self._embeddings) == 0:
                self._embeddings = None

            self._save_locked()

        return record_id

    def list_faces(self, limit: int = 20, offset: int = 0) -> list[dict]:
        """Return metadata slice (newest first)."""
        with self._lock:
            # Return newest first (reverse order)
            items = list(reversed(self._meta))
            return items[offset: offset + limit]

    def get_image_path(self, record_id: str) -> Path | None:
        """Return path to image JPEG, or None if not found."""
        path = self._img_dir / f"{record_id}.jpg"
        return path if path.exists() else None

    def delete(self, record_id: str) -> bool:
        """Delete a single record. Returns True if deleted."""
        with self._lock:
            idx = next((i for i, m in enumerate(self._meta) if m["id"] == record_id), None)
            if idx is None:
                return False
            self._meta.pop(idx)
            if self._embeddings is not None:
                keep = np.ones(len(self._embeddings) + 1, dtype=bool)
                keep[idx] = False
                # embeddings may be 1 longer if entry was just added; trim safely
                emb_len = len(self._embeddings)
                if idx < emb_len:
                    mask = np.ones(emb_len, dtype=bool)
                    mask[idx] = False
                    kept = self._embeddings[mask]
                    self._embeddings = kept if len(kept) > 0 else None
            img = self._img_dir / f"{record_id}.jpg"
            img.unlink(missing_ok=True)
            self._save_locked()
        return True

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._meta)

    @property
    def dim(self) -> int:
        with self._lock:
            if self._embeddings is not None and self._embeddings.ndim == 2:
                return self._embeddings.shape[1]
            return 512

    @property
    def estimate_size_mb(self) -> float:
        """Rough estimate: ~4.2 KB per face (2KB JPEG + 2KB embedding + 0.2KB meta)."""
        with self._lock:
            n = len(self._meta)
        return round(n * 4.2 / 1024, 3)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_locked(self) -> None:
        """Must be called with self._lock held."""
        if self._embeddings is not None and len(self._embeddings) > 0:
            np.save(str(self._dir / "index.npy"), self._embeddings)
        else:
            np.save(str(self._dir / "index.npy"), np.zeros((0, 512), dtype=np.float32))
        (self._dir / "meta.json").write_text(
            json.dumps(self._meta, indent=2, ensure_ascii=False)
        )

    def _load(self) -> None:
        """Load from disk (called from __init__, no lock needed)."""
        idx_path  = self._dir / "index.npy"
        meta_path = self._dir / "meta.json"
        if idx_path.exists():
            arr = np.load(str(idx_path))
            if arr.ndim == 2 and arr.shape[0] > 0:
                self._embeddings = arr.astype(np.float32)
        if meta_path.exists():
            try:
                self._meta = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                self._meta = []


# ── Helpers ───────────────────────────────────────────────────────────────────

def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-8 else v
