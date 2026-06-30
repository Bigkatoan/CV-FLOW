"""
Tests for cv_flow.dam.round_robin.RoundRobinBus

T-RR-01: n=3, ghi 9 frames → mỗi bus nhận đúng 3 frames (round-robin)
T-RR-02: add_bus() → frames tiếp theo được phân phối sang bus mới
T-RR-03: remove_bus() → frames không còn vào bus đã xóa
T-RR-04: get_buffer_depth() = average depth của tất cả buses
"""
from __future__ import annotations

import uuid
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from cv_flow.dam.round_robin import RoundRobinBus
from cv_flow.dam.bus import PortBus


def _unique(prefix="rrb"):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# ── T-RR-01 ───────────────────────────────────────────────────────────────────

def test_rr_01_round_robin_distribution():
    """n=3, write 9 frames → each bus gets exactly 3 frames."""
    name = _unique()
    rrb  = RoundRobinBus(name, n=3, slot_bytes=8, queue_depth=8)
    try:
        for i in range(9):
            rrb.write(b"\x00" * 8, seq=i + 1)

        counts = []
        for bus in rrb.buses:
            count = 0
            while bus.read(timeout_ms=10) is not None:
                count += 1
            counts.append(count)

        assert counts == [3, 3, 3], f"Expected [3,3,3], got {counts}"
    finally:
        rrb.close()


# ── T-RR-02 ───────────────────────────────────────────────────────────────────

def test_rr_02_add_bus():
    """add_bus() → subsequent frames distributed to new bus."""
    name = _unique()
    rrb  = RoundRobinBus(name, n=2, slot_bytes=8, queue_depth=8)
    try:
        # Write 4 frames before add (bus 0 and 1 each get 2)
        for i in range(4):
            rrb.write(b"\xAA" * 8, seq=i + 1)

        new_bus = rrb.add_bus()   # now 3 buses
        assert len(rrb.buses) == 3

        # Write 6 more frames — each of the 3 buses should get 2
        for i in range(6):
            rrb.write(b"\xBB" * 8, seq=i + 100)

        # The new bus must have received some frames
        new_count = 0
        while new_bus.read(timeout_ms=10) is not None:
            new_count += 1

        assert new_count > 0, "New bus received no frames after add_bus()"
        assert new_count == 2, f"Expected 2, got {new_count}"
    finally:
        rrb.close()


def test_rr_02b_add_bus_with_explicit_name():
    """add_bus(name=...) creates the bus under that exact deterministic name
    (needed so a separately-spawned worker process can derive and attach to
    the same shared-memory segment by name)."""
    name = _unique()
    rrb  = RoundRobinBus(name, n=1, slot_bytes=8, queue_depth=8)
    try:
        explicit_name = f"{name}__explicit"
        new_bus = rrb.add_bus(name=explicit_name)
        assert new_bus.name == explicit_name

        # A second PortBus opened by the same name (create=False) attaches
        # to the SAME shared memory segment.
        attached = PortBus(explicit_name, 8, create=False)
        try:
            new_bus.write(b"\xEE" * 8, seq=1)
            result = attached.read(timeout_ms=100)
            assert result is not None
        finally:
            attached.close(unlink=False)
    finally:
        rrb.close()


# ── T-RR-03 ───────────────────────────────────────────────────────────────────

def test_rr_03_remove_bus():
    """remove_bus() → removed bus receives no more frames."""
    name = _unique()
    rrb  = RoundRobinBus(name, n=3, slot_bytes=8, queue_depth=8)
    try:
        # Drain any queued frames first
        for bus in rrb.buses:
            while bus.read(timeout_ms=5) is not None:
                pass

        # Remove bus at index 1
        rrb.remove_bus(1)
        assert len(rrb.buses) == 2

        # Write 10 frames — should go only to buses 0 and (new) 1
        for i in range(10):
            rrb.write(b"\xCC" * 8, seq=i + 1)

        total = sum(
            sum(1 for _ in iter(lambda: bus.read(timeout_ms=10), None))
            for bus in rrb.buses
        )
        assert total == 10, f"Expected 10 total frames, got {total}"
    finally:
        rrb.close()


# ── T-RR-04 ───────────────────────────────────────────────────────────────────

def test_rr_04_get_buffer_depth():
    """get_buffer_depth() returns average depth of all buses."""
    name = _unique()
    rrb  = RoundRobinBus(name, n=4, slot_bytes=8, queue_depth=8)
    try:
        # Write 8 frames → each of 4 buses gets 2
        for i in range(8):
            rrb.write(b"\xDD" * 8, seq=i + 1)

        depth = rrb.get_buffer_depth()
        # Each bus has 2 unread frames → average = 2.0
        assert depth == 2.0, f"Expected avg depth 2.0, got {depth}"
    finally:
        rrb.close()
