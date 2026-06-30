"""
cv_flow.dam.merge — MergeBus: Elastic fan-in (N writers → 1 reader).

Reads from all buses and delivers the lowest seq_no among whatever buses
currently have data ready.

Important limitation (found building cv_flow.elastic.ElasticStage): this is
BEST-EFFORT ordering, not a strict guarantee. read() only compares buses
that already have a frame available *at the moment it is called* — it does
not wait for a momentarily-empty bus that might still produce an earlier
seq_no a few milliseconds later. If one worker is briefly slower than
another, a later-seq result from the faster worker can be returned before
an earlier-seq result from the slower one. Strict ordering needs a
reorder buffer on top of this (see `ElasticStage.spin_once()`), not just
MergeBus alone.

Also see cv_flow/dam/bus.py's concurrency note: peek()-then-read() here is
two separate calls with no atomicity between them, so a genuine concurrent
writer process can race this read in rare cases. `ElasticStage` does not
use this class directly for that reason — it does its own per-worker
peek/read under a `multiprocessing.Lock` shared with that worker.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

from cv_flow.dam.bus import PortBus, Detection


class MergeBus:
    """
    Read from N PortBus sources, deliver frames sorted by seq_no.

    Thread-safe for concurrent add_bus / remove_bus operations.
    """

    def __init__(self, buses: list[PortBus]) -> None:
        self._lock  = threading.Lock()
        self._buses = list(buses)

    # ── read ─────────────────────────────────────────────────────────────────

    def read(
        self,
        timeout_ms: int = 30,
    ) -> Optional[tuple[bytes, int, list[Detection], dict]]:
        """
        Poll all buses once, return the frame with the lowest seq_no.

        Blocks up to timeout_ms if no bus has data yet.
        Returns None if all buses are empty after the timeout.
        """
        deadline = time.monotonic() + timeout_ms / 1000.0

        while True:
            with self._lock:
                buses = list(self._buses)

            # Peek (non-destructive) at each bus to get the head seq_no
            candidates: list[tuple[int, PortBus]] = []
            for bus in buses:
                result = bus.peek()
                if result is not None:
                    _, seq, _, _ = result
                    candidates.append((seq, bus))

            if candidates:
                candidates.sort(key=lambda c: c[0])
                _, winning_bus = candidates[0]
                # Only consume from the bus with the lowest seq_no
                result = winning_bus.read(timeout_ms=0)
                if result is not None:
                    return result

            if time.monotonic() >= deadline:
                return None
            time.sleep(0.0005)

    # ── scale ─────────────────────────────────────────────────────────────────

    def add_bus(self, bus: PortBus) -> None:
        with self._lock:
            self._buses.append(bus)

    def remove_bus(self, bus: PortBus) -> None:
        with self._lock:
            try:
                self._buses.remove(bus)
            except ValueError:
                pass

    # ── monitoring ────────────────────────────────────────────────────────────

    @property
    def bus_count(self) -> int:
        with self._lock:
            return len(self._buses)

    def __repr__(self) -> str:
        return f"MergeBus(n={self.bus_count})"
