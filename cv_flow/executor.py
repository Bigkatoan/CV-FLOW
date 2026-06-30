"""
cv_flow.executor — Executor: manages Node lifecycle and spin loop.

Responsibilities:
  1. Initialize all nodes (call node.initialize()).
  2. Spin nodes in declared order (call node.spin_once() in a loop).
  3. Respect hz rate limiting.
  4. Handle StopIteration / SIGINT for graceful shutdown.
  5. Optionally monitor buffer depths and auto-scale elastic nodes.
"""
from __future__ import annotations

import logging
import signal
import threading
import time
import uuid
from typing import Callable, Optional

logger = logging.getLogger("cv_flow.executor")


class Executor:
    """
    Runs a list of Nodes in a spin loop.

    Parameters
    ----------
    nodes     : Ordered list of Node instances.
    hz        : Target iterations per second (0 = as fast as possible).
    elastic   : Enable background buffer-depth monitor for auto-scaling.
    """

    def __init__(
        self,
        nodes: list,
        *,
        hz: float = 0.0,
        elastic: bool = False,
    ) -> None:
        self.nodes   = list(nodes)
        self.hz      = hz
        self.elastic = elastic

        self._session_id = uuid.uuid4().hex[:8]
        self._running    = False
        self._stop_event = threading.Event()

        # Inject session_id into every node before spin
        for node in self.nodes:
            node._session_id = self._session_id

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Call initialize() on all nodes in order."""
        for node in self.nodes:
            node.initialize()

    def shutdown(self) -> None:
        """Call shutdown() on all nodes in REVERSE order."""
        for node in reversed(self.nodes):
            try:
                node.shutdown()
            except Exception as exc:
                logger.warning("Node %s shutdown error: %s", node.name, exc)
        self._running = False

    # ── spin ──────────────────────────────────────────────────────────────────

    def spin(self) -> None:
        """
        Blocking spin loop. Initializes nodes, then calls spin_once() in a
        loop until StopIteration is raised or SIGINT is received.
        """
        self._install_sigint()
        self.initialize()
        self._running = True

        period = (1.0 / self.hz) if self.hz > 0 else 0.0

        if self.elastic:
            self._start_elastic_monitor()

        try:
            while self._running and not self._stop_event.is_set():
                t0 = time.monotonic()
                try:
                    for node in self.nodes:
                        node.spin_once()
                except StopIteration:
                    logger.info("Executor: StopIteration received, shutting down.")
                    break

                if period > 0:
                    elapsed = time.monotonic() - t0
                    remaining = period - elapsed
                    if remaining > 0:
                        time.sleep(remaining)
        finally:
            self.shutdown()

    def spin_background(self) -> threading.Thread:
        """Launch spin() in a background thread. Returns the Thread."""
        t = threading.Thread(target=self.spin, daemon=True, name="cv-flow-executor")
        t.start()
        return t

    def stop(self) -> None:
        """Signal the spin loop to stop."""
        self._stop_event.set()

    # ── SIGINT ────────────────────────────────────────────────────────────────

    def _install_sigint(self) -> None:
        def _handler(sig, frame):
            logger.info("Executor: SIGINT received, stopping.")
            self._stop_event.set()

        try:
            signal.signal(signal.SIGINT, _handler)
        except (OSError, ValueError):
            pass  # Can't install in non-main thread

    # ── elastic monitor ───────────────────────────────────────────────────────

    def _start_elastic_monitor(self) -> threading.Thread:
        """Background thread that monitors buffer depths and triggers scale."""
        t = threading.Thread(
            target=self._elastic_loop, daemon=True, name="cv-flow-elastic"
        )
        t.start()
        return t

    def _elastic_loop(self) -> None:
        """Minimal elastic monitor — subclass or replace for production use."""
        scale_up_threshold   = 6
        scale_down_count_req = 5
        idle_counts: dict[str, int] = {}

        while self._running and not self._stop_event.is_set():
            time.sleep(0.5)
            for node in self.nodes:
                if not getattr(node, "_elastic_capable", False):
                    continue
                depth = getattr(node, "get_buffer_depth", lambda: 0)()
                name  = node.name

                if depth > scale_up_threshold:
                    idle_counts[name] = 0
                    self.scale_up(node)
                elif depth == 0:
                    idle_counts[name] = idle_counts.get(name, 0) + 1
                    if idle_counts[name] >= scale_down_count_req:
                        idle_counts[name] = 0
                        self.scale_down(node)
                else:
                    idle_counts[name] = 0

    def scale_up(self, node) -> None:
        """
        Hook called when a node's buffer is too deep.

        Default: if `node` exposes `add_worker()` (e.g. cv_flow.elastic.ElasticStage,
        which spawns a real multiprocessing.Process worker), call it. Override
        for custom behavior.
        """
        if hasattr(node, "add_worker"):
            node.add_worker()
        else:
            logger.debug("scale_up(%s)", node.name)

    def scale_down(self, node) -> None:
        """
        Hook called when a node's buffer is consistently empty.

        Default: if `node` exposes `remove_worker()`, call it. Override for
        custom behavior.
        """
        if hasattr(node, "remove_worker"):
            node.remove_worker()
        else:
            logger.debug("scale_down(%s)", node.name)
