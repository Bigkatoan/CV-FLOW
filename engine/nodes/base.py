"""BaseNode ABC — all engine nodes must implement this interface."""
from __future__ import annotations
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Optional

from engine.core.frame_context import FrameContext


class BaseNode(ABC):
    node_id:   str = ""
    node_type: str = ""
    config:    dict

    # Profiling state (populated by PipelineRunner after each process() call)
    _timing_window: deque   # rolling window of elapsed_ms values (last 100 frames)
    _frame_count:   int
    _error_count:   int
    _last_fps_t:    float
    _fps:           float

    def setup(self, node_id: str, config: dict, node_type: str = "") -> None:
        self.node_id   = node_id
        self.node_type = node_type
        self.config    = config
        # Profiling init
        self._timing_window = deque(maxlen=100)
        self._frame_count   = 0
        self._error_count   = 0
        self._last_fps_t    = time.monotonic()
        self._fps           = 0.0
        self.initialize()

    def initialize(self) -> None:
        """Override to set up resources (open files, load weights, etc.)."""

    @abstractmethod
    def process(self, ctx: FrameContext) -> Optional[FrameContext]:
        """
        Process one frame.

        Return ctx (modified or unchanged) to pass it downstream.
        Return None to DROP this frame — it will not be forwarded.
        Raise StopIteration to signal that this source is exhausted.
        """

    def teardown(self) -> None:
        """Override to release resources."""

    # ── Profiling helpers (called by runner, not by node code) ────────────────

    def _record_timing(self, elapsed_ms: float) -> None:
        self._timing_window.append(elapsed_ms)
        self._frame_count += 1
        now = time.monotonic()
        dt  = now - self._last_fps_t
        if dt >= 1.0:
            self._fps = self._frame_count / max(dt, 1e-6)
            self._frame_count = 0
            self._last_fps_t  = now

    def get_stats(self) -> dict:
        """Return profiling stats for this node."""
        w = list(self._timing_window)
        if not w:
            return {"avg_ms": 0.0, "p95_ms": 0.0, "fps": self._fps, "errors": self._error_count}
        sorted_w = sorted(w)
        return {
            "avg_ms":  round(sum(w) / len(w), 2),
            "p95_ms":  round(sorted_w[int(len(sorted_w) * 0.95)], 2),
            "fps":     round(self._fps, 2),
            "errors":  self._error_count,
        }
