"""
cv_flow.dam.merge — MergeBus: Elastic fan-in (N writers → 1 reader).

Reads from all buses and delivers frames in seq_no order.
Solves the concurrent-worker ordering problem: even if N elastic workers
return at the same time, MergeBus sorts by seq_no so the downstream
node always sees a monotonically increasing sequence.
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
