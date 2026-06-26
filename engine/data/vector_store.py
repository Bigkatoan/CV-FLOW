"""
VectorStore — numpy-backed cosine similarity store for named embedding collections.

Each collection is stored as:
  <storage_dir>/<name>/index.npy   — stacked float32 embedding matrix (N × dim)
  <storage_dir>/<name>/meta.json   — list of [{id, label, ...metadata}]

Thread safety:
  - threading.RLock allows concurrent reads via search()
  - Writes (add, delete) hold the lock exclusively
  - _save_locked() must be called with lock held

Write strategy:
  - Write-through: every add() and delete() saves immediately
  - Avoids data loss on crash vs. periodic-flush approach
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SearchResult:
    id:       str
    score:    float
    metadata: dict


class VectorStore:
    """Named collection of L2-normalised float32 embeddings, cosine similarity via dot product.

    Args:
        name:        Collection name (used as subdirectory).
        storage_dir: Parent directory; collection files live at storage_dir/name/.
        dim:         Expected embedding dimension (validated on add).
                     Ignored if loading an existing index — actual dim inferred from index.npy.
    """

    def __init__(self, name: str, storage_dir: str | Path, dim: int = 512) -> None:
        self.name   = name
        self._dir   = Path(storage_dir) / name
        self._dim   = dim
        self._lock  = threading.RLock()
        self._embeddings: np.ndarray | None = None  # shape (N, dim), float32, unit-normed
        self._meta: list[dict[str, Any]]    = []    # [{id, label, ...}]

        self._dir.mkdir(parents=True, exist_ok=True)
        if (self._dir / "index.npy").exists():
            self._load()

    # ── Public interface ──────────────────────────────────────────────────────

    def add(self, id: str, embedding: np.ndarray | list, metadata: dict | None = None) -> None:
        """Add a new embedding.

        Raises:
            ValueError: if embedding dimension doesn't match expected dim.
        """
        emb = np.array(embedding, dtype=np.float32).reshape(-1)
        if emb.shape[0] != self._dim:
            raise ValueError(
                f"Expected embedding dim={self._dim}, got {emb.shape[0]}. "
                "Pass dim=N to VectorStore constructor to override."
            )
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm  # normalise to unit sphere: cosine sim = dot product

        with self._lock:
            if self._embeddings is None:
                self._embeddings = emb.reshape(1, -1)
            else:
                self._embeddings = np.vstack([self._embeddings, emb.reshape(1, -1)])
            entry = {"id": id, "label": id, **(metadata or {})}
            self._meta.append(entry)
            self._save_locked()

    def search(self, query: np.ndarray | list, top_k: int = 5) -> list[SearchResult]:
        """Cosine similarity search. Returns top_k results sorted descending by score."""
        q = np.array(query, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        with self._lock:
            if self._embeddings is None or len(self._embeddings) == 0:
                return []
            # dot product of unit vectors = cosine similarity
            scores = self._embeddings @ q
            k      = min(top_k, len(scores))
            idxs   = np.argsort(scores)[::-1][:k]
            return [
                SearchResult(
                    id       = self._meta[i]["id"],
                    score    = float(scores[i]),
                    metadata = {k: v for k, v in self._meta[i].items() if k not in ("id",)},
                )
                for i in idxs
            ]

    def delete(self, id: str) -> bool:
        """Delete all entries with the given id. Returns True if any were deleted."""
        with self._lock:
            idxs = [i for i, m in enumerate(self._meta) if m["id"] == id]
            if not idxs:
                return False
            keep = np.ones(len(self._meta), dtype=bool)
            for i in idxs:
                keep[i] = False
            if self._embeddings is not None:
                kept = self._embeddings[keep]
                self._embeddings = kept if len(kept) > 0 else None
            self._meta = [m for i, m in enumerate(self._meta) if keep[i]]
            self._save_locked()
        return True

    def clear(self) -> None:
        """Remove all embeddings and metadata."""
        with self._lock:
            self._embeddings = None
            self._meta = []
            self._save_locked()

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
            return self._dim

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _save_locked(self) -> None:
        """Persist embeddings and metadata to disk. Must be called with self._lock held."""
        if self._embeddings is not None and len(self._embeddings) > 0:
            np.save(str(self._dir / "index.npy"), self._embeddings)
        else:
            # Write empty array so the file exists for future loads
            np.save(str(self._dir / "index.npy"), np.zeros((0, self._dim), dtype=np.float32))
        (self._dir / "meta.json").write_text(json.dumps(self._meta, indent=2, ensure_ascii=False))

    def _load(self) -> None:
        """Load from disk. Safe to call without lock (called only from __init__)."""
        idx_path  = self._dir / "index.npy"
        meta_path = self._dir / "meta.json"
        if idx_path.exists():
            arr = np.load(str(idx_path))
            if arr.ndim == 2 and arr.shape[0] > 0:
                self._embeddings = arr.astype(np.float32)
                self._dim        = arr.shape[1]
            else:
                self._embeddings = None
        if meta_path.exists():
            try:
                self._meta = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                self._meta = []
