"""
Tests for cv_flow.dam.bus.PortBus

T-BUS-01: write → read round-trip, data không bị corrupt
T-BUS-02: queue_depth=2, ghi 3 frames → drop frame cũ nhất, log WARNING
T-BUS-03: drop_count tăng đúng sau mỗi lần drop
T-BUS-04: read() trả None sau timeout nếu không có data mới
T-BUS-05: seq_no tăng đơn điệu, read() trả đúng seq_no
T-BUS-06: drop_mode=True → overwrite silently, không log
T-BUS-07: stats() trả đúng writes/reads/drops/depth
T-BUS-08: multi-process: process A ghi, process B đọc, data đúng
T-BUS-09: close(unlink=True) giải phóng shared memory
T-BUS-10: nhiều writes liên tiếp không gây memory leak
"""
from __future__ import annotations

import multiprocessing as mp
import sys
import time
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from cv_flow.dam.bus import PortBus, Detection


def _unique(prefix="test"):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# ── T-BUS-01 ──────────────────────────────────────────────────────────────────

def test_bus_01_round_trip():
    """write → read, data không bị corrupt."""
    name = _unique()
    try:
        pub = PortBus(name, slot_bytes=64, queue_depth=4, create=True)
        sub = PortBus(name, slot_bytes=64, create=False)

        payload = b"Hello CV-FLOW!" + b"\x00" * 50
        pub.write(payload[:64], seq=1)
        result = sub.read(timeout_ms=100)

        assert result is not None, "read() returned None"
        data, seq_no, dets, meta = result
        assert data[:14] == b"Hello CV-FLOW!"
        assert seq_no == 1
        assert dets == []
        assert meta == {}
    finally:
        pub.close(unlink=True)
        sub.close()


# ── T-BUS-02 ──────────────────────────────────────────────────────────────────

def test_bus_02_drop_oldest_on_full(caplog):
    """queue_depth=2, ghi 3 frames → drop cũ nhất, log WARNING."""
    import logging
    name = _unique()
    try:
        pub = PortBus(name, slot_bytes=8, queue_depth=2, create=True)
        sub = PortBus(name, slot_bytes=8, create=False)

        with caplog.at_level(logging.WARNING, logger="cv_flow.dam"):
            dropped1 = pub.write(b"\x01" * 8, seq=1)
            dropped2 = pub.write(b"\x02" * 8, seq=2)
            dropped3 = pub.write(b"\x03" * 8, seq=3)  # → drop seq=1

        assert not dropped1
        assert not dropped2
        assert dropped3, "3rd write should signal a drop"
        assert any("queue full" in r.message for r in caplog.records), \
            "Expected WARNING log about queue full"

        # read should get seq=2 (seq=1 was dropped)
        result = sub.read(timeout_ms=100)
        assert result is not None
        _, seq_no, _, _ = result
        assert seq_no == 2, f"Expected seq=2 (oldest unread), got {seq_no}"
    finally:
        pub.close(unlink=True)
        sub.close()


# ── T-BUS-03 ──────────────────────────────────────────────────────────────────

def test_bus_03_drop_count():
    """drop_count tăng đúng."""
    name = _unique()
    try:
        pub = PortBus(name, slot_bytes=4, queue_depth=1, drop_mode=True, create=True)
        pub.write(b"\xAA\xBB\xCC\xDD", seq=1)
        pub.write(b"\xAA\xBB\xCC\xDD", seq=2)  # drop 1
        pub.write(b"\xAA\xBB\xCC\xDD", seq=3)  # drop 2

        stats = pub.stats
        assert stats["drops"] == 2, f"Expected 2 drops, got {stats['drops']}"
    finally:
        pub.close(unlink=True)


# ── T-BUS-04 ──────────────────────────────────────────────────────────────────

def test_bus_04_read_timeout():
    """read() trả None sau timeout nếu không có data."""
    name = _unique()
    try:
        pub = PortBus(name, slot_bytes=8, queue_depth=4, create=True)
        sub = PortBus(name, slot_bytes=8, create=False)

        t0 = time.monotonic()
        result = sub.read(timeout_ms=50)
        elapsed_ms = (time.monotonic() - t0) * 1000

        assert result is None
        assert elapsed_ms >= 45, f"Timeout too short: {elapsed_ms:.1f}ms"
        assert elapsed_ms < 200, f"Timeout too long: {elapsed_ms:.1f}ms"
    finally:
        pub.close(unlink=True)
        sub.close()


# ── T-BUS-05 ──────────────────────────────────────────────────────────────────

def test_bus_05_seq_monotonic():
    """seq_no đúng và đọc theo thứ tự."""
    name = _unique()
    try:
        pub = PortBus(name, slot_bytes=4, queue_depth=8, create=True)
        sub = PortBus(name, slot_bytes=4, create=False)

        for i in range(5):
            pub.write(b"\x00" * 4, seq=i + 1)

        seqs = []
        for _ in range(5):
            r = sub.read(timeout_ms=100)
            assert r is not None
            seqs.append(r[1])

        assert seqs == [1, 2, 3, 4, 5], f"Seq not monotonic: {seqs}"
    finally:
        pub.close(unlink=True)
        sub.close()


# ── T-BUS-06 ──────────────────────────────────────────────────────────────────

def test_bus_06_drop_mode_silent(caplog):
    """drop_mode=True → overwrite silently, không log."""
    import logging
    name = _unique()
    try:
        pub = PortBus(name, slot_bytes=4, queue_depth=1, drop_mode=True, create=True)

        with caplog.at_level(logging.WARNING, logger="cv_flow.dam"):
            pub.write(b"\x01\x02\x03\x04", seq=1)
            pub.write(b"\x05\x06\x07\x08", seq=2)

        assert not any("queue full" in r.message for r in caplog.records), \
            "drop_mode=True should not emit WARNING"
    finally:
        pub.close(unlink=True)


# ── T-BUS-07 ──────────────────────────────────────────────────────────────────

def test_bus_07_stats():
    """stats() trả đúng writes/reads/drops/depth."""
    name = _unique()
    try:
        pub = PortBus(name, slot_bytes=8, queue_depth=4, drop_mode=True, create=True)
        sub = PortBus(name, slot_bytes=8, create=False)

        pub.write(b"\x00" * 8, seq=1)
        pub.write(b"\x00" * 8, seq=2)
        sub.read(timeout_ms=50)

        s = pub.stats
        assert s["writes"] == 2
        assert s["reads"]  == 1
        assert s["drops"]  == 0
        assert s["depth"]  == 1
    finally:
        pub.close(unlink=True)
        sub.close()


# ── T-BUS-08 ──────────────────────────────────────────────────────────────────

def _writer_proc(name, slot_bytes, payload, count):
    pub = PortBus(name, slot_bytes=slot_bytes, create=False)
    for i in range(count):
        pub.write(payload, seq=i + 1)
        time.sleep(0.002)
    pub.close()


def test_bus_08_multiprocess():
    """multi-process: process A ghi, process B đọc, data đúng."""
    name   = _unique()
    n      = 5
    payload = b"\xDE\xAD\xBE\xEF" * 2  # 8 bytes

    pub = PortBus(name, slot_bytes=8, queue_depth=16, create=True)
    sub = PortBus(name, slot_bytes=8, create=False)

    proc = mp.Process(target=_writer_proc, args=(name, 8, payload, n))
    proc.start()

    received = []
    deadline = time.monotonic() + 2.0
    while len(received) < n and time.monotonic() < deadline:
        r = sub.read(timeout_ms=100)
        if r is not None:
            data, seq, _, _ = r
            received.append((seq, data[:8]))

    proc.join(timeout=3)

    assert len(received) == n, f"Expected {n} frames, got {len(received)}"
    for seq, data in received:
        assert data == payload, f"Data mismatch at seq={seq}"

    pub.close(unlink=True)
    sub.close()


# ── T-BUS-09 ──────────────────────────────────────────────────────────────────

def test_bus_09_close_unlink():
    """close(unlink=True) giải phóng shared memory."""
    from multiprocessing.shared_memory import SharedMemory
    name = _unique()

    pub = PortBus(name, slot_bytes=8, queue_depth=2, create=True)
    pub.close(unlink=True)

    with pytest.raises(Exception):
        SharedMemory(name=name, create=False)


# ── T-BUS-10 ──────────────────────────────────────────────────────────────────

def test_bus_10_no_memory_leak():
    """Nhiều writes không gây leak (stats tăng đúng, không crash)."""
    name = _unique()
    try:
        pub = PortBus(name, slot_bytes=64, queue_depth=4, drop_mode=True, create=True)
        sub = PortBus(name, slot_bytes=64, create=False)

        for i in range(100):
            pub.write(b"\xAB" * 64, seq=i)

        s = pub.stats
        assert s["writes"] == 100
        assert s["depth"] <= 4  # ring buffer không vượt queue_depth
    finally:
        pub.close(unlink=True)
        sub.close()


# ── Bonus: detections + metadata round-trip ───────────────────────────────────

def test_bus_detections_metadata():
    """Detections và metadata được ghi/đọc đúng."""
    name = _unique()
    try:
        pub = PortBus(name, slot_bytes=4, queue_depth=4, create=True)
        sub = PortBus(name, slot_bytes=4, create=False)

        det = Detection(x1=10, y1=20, x2=100, y2=200, confidence=0.9,
                        class_id=1, class_name="person", track_id=42,
                        metadata={"score": 0.9})
        pub.write(b"\x00" * 4, seq=7,
                  detections=[det],
                  metadata={"frame_id": 123, "cam": "cam0"})

        result = sub.read(timeout_ms=100)
        assert result is not None
        data, seq, dets, meta = result

        assert seq == 7
        assert len(dets) == 1
        assert dets[0].class_name == "person"
        assert dets[0].track_id == 42
        assert abs(dets[0].confidence - 0.9) < 1e-5
        assert meta["frame_id"] == 123
        assert meta["cam"] == "cam0"
    finally:
        pub.close(unlink=True)
        sub.close()
