"""
Tests for cv_flow.dam.merge.MergeBus

T-MERGE-01: 3 buses, write seq=1,2,3 out of order → read() always returns in seq order
T-MERGE-02: fast/slow worker doesn't affect ordering
T-MERGE-03: add_bus() during run → no frame loss
T-MERGE-04: remove_bus() during run → no deadlock
T-MERGE-05: timeout if all buses are empty
"""
from __future__ import annotations

import sys
import time
import threading
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from cv_flow.dam.bus import PortBus
from cv_flow.dam.merge import MergeBus


def _unique(prefix="mrg"):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _make_bus(name, slot_bytes=8, queue_depth=16):
    return PortBus(name, slot_bytes=slot_bytes, queue_depth=queue_depth, create=True)


# ── T-MERGE-01 ────────────────────────────────────────────────────────────────

def test_merge_01_ordered_delivery():
    """Write to 3 buses out of seq order → MergeBus delivers sorted by seq."""
    b0 = _make_bus(_unique())
    b1 = _make_bus(_unique())
    b2 = _make_bus(_unique())
    merge = MergeBus([b0, b1, b2])
    try:
        # Write to buses out of order: b2→seq3, b0→seq1, b1→seq2
        b2.write(b"\x03" * 8, seq=3)
        b0.write(b"\x01" * 8, seq=1)
        b1.write(b"\x02" * 8, seq=2)

        seqs = []
        for _ in range(3):
            r = merge.read(timeout_ms=100)
            assert r is not None
            seqs.append(r[1])

        assert seqs == [1, 2, 3], f"Expected [1,2,3] but got {seqs}"
    finally:
        b0.close(unlink=True); b1.close(unlink=True); b2.close(unlink=True)


# ── T-MERGE-02 ────────────────────────────────────────────────────────────────

def test_merge_02_fast_slow_worker():
    """Fast and slow worker → MergeBus ordering is maintained."""
    b_fast = _make_bus(_unique())
    b_slow = _make_bus(_unique())
    merge  = MergeBus([b_fast, b_slow])
    try:
        # fast worker writes even seqs immediately
        for i in [2, 4, 6]:
            b_fast.write(b"\xFF" * 8, seq=i)

        # slow worker writes odd seqs with a tiny delay
        def slow_write():
            for i in [1, 3, 5]:
                time.sleep(0.005)
                b_slow.write(b"\xEE" * 8, seq=i)

        t = threading.Thread(target=slow_write)
        t.start()
        t.join(timeout=2)

        seqs = []
        deadline = time.monotonic() + 1.0
        while len(seqs) < 6 and time.monotonic() < deadline:
            r = merge.read(timeout_ms=20)
            if r is not None:
                seqs.append(r[1])

        assert sorted(seqs) == seqs, f"Seqs not monotonic: {seqs}"
        assert len(seqs) == 6, f"Expected 6 frames, got {len(seqs)}"
    finally:
        b_fast.close(unlink=True); b_slow.close(unlink=True)


# ── T-MERGE-03 ────────────────────────────────────────────────────────────────

def test_merge_03_add_bus_no_loss():
    """add_bus() during operation → no frames lost."""
    b0    = _make_bus(_unique())
    merge = MergeBus([b0])
    try:
        # Write first batch to b0
        for i in range(3):
            b0.write(b"\xAA" * 8, seq=i + 1)

        # Add new bus and write more frames
        b1 = _make_bus(_unique())
        merge.add_bus(b1)
        assert merge.bus_count == 2

        for i in range(3):
            b1.write(b"\xBB" * 8, seq=i + 10)

        received = []
        deadline = time.monotonic() + 1.0
        while len(received) < 6 and time.monotonic() < deadline:
            r = merge.read(timeout_ms=20)
            if r is not None:
                received.append(r[1])

        assert len(received) == 6, f"Expected 6 frames, got {len(received)}"
    finally:
        b0.close(unlink=True)
        b1.close(unlink=True)


# ── T-MERGE-04 ────────────────────────────────────────────────────────────────

def test_merge_04_remove_bus_no_deadlock():
    """remove_bus() during operation → no deadlock, no crash."""
    b0    = _make_bus(_unique())
    b1    = _make_bus(_unique())
    merge = MergeBus([b0, b1])
    try:
        for i in range(4):
            b0.write(b"\xCC" * 8, seq=i + 1)

        # Remove b1 while reading
        merge.remove_bus(b1)
        assert merge.bus_count == 1

        # Should still be able to read from b0
        received = 0
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            r = merge.read(timeout_ms=20)
            if r is not None:
                received += 1
            else:
                break

        assert received == 4, f"Expected 4 frames from b0, got {received}"
    finally:
        b0.close(unlink=True); b1.close(unlink=True)


# ── T-MERGE-05 ────────────────────────────────────────────────────────────────

def test_merge_05_timeout_empty():
    """read() returns None after timeout if all buses are empty."""
    b0    = _make_bus(_unique())
    merge = MergeBus([b0])
    try:
        t0     = time.monotonic()
        result = merge.read(timeout_ms=50)
        elapsed_ms = (time.monotonic() - t0) * 1000

        assert result is None
        assert elapsed_ms >= 45, f"Timeout too short: {elapsed_ms:.1f}ms"
        assert elapsed_ms < 200, f"Timeout too long: {elapsed_ms:.1f}ms"
    finally:
        b0.close(unlink=True)
