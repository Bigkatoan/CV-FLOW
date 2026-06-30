"""
cv_flow.dam.round_robin — RoundRobinBus: Elastic fan-out (1 writer → N readers).

Each write distributes to one worker bus in a round-robin fashion.
Workers can be added/removed at runtime for elastic scaling.
"""
from __future__ import annotations

import threading
import uuid
from typing import Any

from cv_flow.dam.bus import PortBus


class RoundRobinBus:
    """
    1 writer → N PortBus readers, distributed round-robin.

    Thread-safe for concurrent write + scale operations.
    """

    def __init__(
        self,
        name: str,
        n: int,
        slot_bytes: int,
        **bus_kwargs: Any,
    ) -> None:
        self.base_name  = name
        self.slot_bytes = slot_bytes
        self._bus_kwargs = bus_kwargs
        self._lock = threading.Lock()
        self._counter = 0

        self._buses: list[PortBus] = [
            PortBus(
                f"{name}_{i}",
                slot_bytes,
                create=True,
                **bus_kwargs,
            )
            for i in range(n)
        ]

    # ── write ─────────────────────────────────────────────────────────────────

    def write(self, data: bytes, seq: int, **kwargs: Any) -> None:
        """Distribute one frame to the next worker bus (round-robin)."""
        with self._lock:
            if not self._buses:
                return
            idx = self._counter % len(self._buses)
            self._counter += 1
            bus = self._buses[idx]
        bus.write(data, seq, **kwargs)

    # ── scale ─────────────────────────────────────────────────────────────────

    def add_bus(self) -> PortBus:
        """Add a new worker bus and return it. Caller attaches as reader."""
        with self._lock:
            idx  = len(self._buses)
            name = f"{self.base_name}_{idx}_{uuid.uuid4().hex[:4]}"
            bus  = PortBus(name, self.slot_bytes, create=True, **self._bus_kwargs)
            self._buses.append(bus)
        return bus

    def remove_bus(self, idx: int) -> None:
        """Remove bus at index idx and close + unlink it."""
        with self._lock:
            if idx < 0 or idx >= len(self._buses):
                raise IndexError(f"No bus at index {idx}")
            bus = self._buses.pop(idx)
        bus.close(unlink=True)

    # ── monitoring ────────────────────────────────────────────────────────────

    def get_buffer_depth(self) -> float:
        """Average pending-frame depth across all buses."""
        with self._lock:
            buses = list(self._buses)
        if not buses:
            return 0.0
        return sum(b.get_buffer_depth() for b in buses) / len(buses)

    @property
    def buses(self) -> list[PortBus]:
        """Read-only snapshot of current bus list."""
        with self._lock:
            return list(self._buses)

    # ── cleanup ───────────────────────────────────────────────────────────────

    def close(self, unlink: bool = True) -> None:
        with self._lock:
            for bus in self._buses:
                bus.close(unlink=unlink)
            self._buses.clear()

    def __del__(self) -> None:
        self.close(unlink=False)

    def __repr__(self) -> str:
        return (f"RoundRobinBus(name={self.base_name!r}, "
                f"n={len(self._buses)}, slot={self.slot_bytes}B)")
