"""
Tests for cv_flow.topic.subscriber.Subscriber

T-SUB-01: read() returns None after timeout
T-SUB-02: after Publisher.write(), read() returns correct data
T-SUB-03: multi-field → dict; single-field → ndarray directly
T-SUB-04: output_device=cuda:0 → read() returns torch.Tensor on GPU (skip if no GPU)
T-SUB-05: read_full() returns (data, detections, metadata, seq_no)
T-SUB-06: seq_no in correct order after multiple writes
T-SUB-07: seq gap detected via has_seq_gap()
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from cv_flow.dam.bus import PortBus, Detection
from cv_flow.dam.cuda_bus import _CUDA_AVAILABLE
from cv_flow.topic.types import FieldDef, PortDef
from cv_flow.topic.publisher import Publisher
from cv_flow.topic.subscriber import Subscriber


def _unique(prefix="sub"):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _make_port(*field_specs):
    fields = [FieldDef.build(n, d, s) for n, d, s in field_specs]
    return PortDef(device="cpu", fields=fields)


def _make_pair(port, queue_depth=8):
    name  = _unique()
    total = port.total_bytes or 8
    pub_bus = PortBus(name, slot_bytes=total, queue_depth=queue_depth, create=True)
    sub_bus = PortBus(name, slot_bytes=total, create=False)
    pub = Publisher(pub_bus, port)
    sub = Subscriber(sub_bus, port)
    return pub_bus, sub_bus, pub, sub


# ── T-SUB-01 ──────────────────────────────────────────────────────────────────

def test_sub_01_timeout_returns_none():
    """read() returns None if no data before timeout."""
    port = _make_port(("x", "uint8", (4,)))
    pub_bus, sub_bus, pub, sub = _make_pair(port)
    try:
        result = sub.read(timeout_ms=30)
        assert result is None
    finally:
        pub_bus.close(unlink=True); sub_bus.close()


# ── T-SUB-02 ──────────────────────────────────────────────────────────────────

def test_sub_02_read_after_write():
    """After Publisher.write(), Subscriber.read() returns correct array."""
    port = _make_port(("val", "float32", (3,)))
    pub_bus, sub_bus, pub, sub = _make_pair(port)
    try:
        arr = np.array([1.1, 2.2, 3.3], dtype=np.float32)
        pub.write(arr)
        result = sub.read(timeout_ms=100)
        assert result is not None
        np.testing.assert_array_almost_equal(result, arr)
    finally:
        pub_bus.close(unlink=True); sub_bus.close()


# ── T-SUB-03 ──────────────────────────────────────────────────────────────────

def test_sub_03_single_vs_multi_field():
    """Single field → ndarray; multi field → dict."""
    # Single field
    port1 = _make_port(("v", "uint8", (4,)))
    pub_bus1, sub_bus1, pub1, sub1 = _make_pair(port1)

    # Multi field
    port2 = _make_port(("a", "uint8", (2,)), ("b", "float32", (2,)))
    pub_bus2, sub_bus2, pub2, sub2 = _make_pair(port2)
    try:
        # Single
        pub1.write(np.array([1, 2, 3, 4], dtype=np.uint8))
        r1 = sub1.read(timeout_ms=100)
        assert isinstance(r1, np.ndarray), f"Expected ndarray, got {type(r1)}"

        # Multi
        pub2.write({"a": np.array([9, 8], dtype=np.uint8),
                    "b": np.array([3.14, 2.71], dtype=np.float32)})
        r2 = sub2.read(timeout_ms=100)
        assert isinstance(r2, dict), f"Expected dict, got {type(r2)}"
        assert set(r2.keys()) == {"a", "b"}
    finally:
        pub_bus1.close(unlink=True); sub_bus1.close()
        pub_bus2.close(unlink=True); sub_bus2.close()


# ── T-SUB-04 ──────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="No CUDA GPU available")
def test_sub_04_cuda_output_device():
    """output_device=cuda:0 → read() returns torch.Tensor on GPU."""
    import torch
    port = _make_port(("x", "float32", (4,)))
    name = _unique()
    total = port.total_bytes
    pub_bus = PortBus(name, slot_bytes=total, queue_depth=4, create=True)
    sub_bus = PortBus(name, slot_bytes=total, create=False)
    pub     = Publisher(pub_bus, port)
    sub     = Subscriber(sub_bus, port, output_device="cuda:0")
    try:
        pub.write(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        result = sub.read(timeout_ms=100)
        assert result is not None
        assert isinstance(result, torch.Tensor)
        assert result.device.type == "cuda"
    finally:
        pub_bus.close(unlink=True); sub_bus.close()


# ── T-SUB-05 ──────────────────────────────────────────────────────────────────

def test_sub_05_read_full():
    """read_full() returns (data, detections, metadata, seq_no)."""
    port = _make_port(("x", "uint8", (2,)))
    pub_bus, sub_bus, pub, sub = _make_pair(port)
    try:
        det = Detection(x1=1, y1=2, x2=3, y2=4, confidence=0.7,
                        class_id=0, class_name="person", track_id=5)
        pub.write(np.array([7, 8], dtype=np.uint8),
                  detections=[det],
                  metadata={"info": "test"})

        result = sub.read_full(timeout_ms=100)
        assert result is not None
        data, dets, meta, seq = result
        assert seq == 1
        assert len(dets) == 1
        assert dets[0].class_name == "person"
        assert meta["info"] == "test"
    finally:
        pub_bus.close(unlink=True); sub_bus.close()


# ── T-SUB-06 ──────────────────────────────────────────────────────────────────

def test_sub_06_seq_order():
    """seq_no is in correct order after multiple writes."""
    port = _make_port(("x", "uint8", (1,)))
    pub_bus, sub_bus, pub, sub = _make_pair(port, queue_depth=8)
    try:
        for _ in range(5):
            pub.write(np.array([0], dtype=np.uint8))

        seqs = []
        for _ in range(5):
            r = sub.read_full(timeout_ms=50)
            assert r is not None
            seqs.append(r[3])  # seq_no

        assert seqs == list(range(1, 6)), f"Expected [1..5], got {seqs}"
    finally:
        pub_bus.close(unlink=True); sub_bus.close()


# ── T-SUB-07 ──────────────────────────────────────────────────────────────────

def test_sub_07_seq_gap_detection():
    """has_seq_gap() detects when seq_no is not consecutive."""
    port = _make_port(("x", "uint8", (1,)))
    pub_bus, sub_bus, pub, sub = _make_pair(port)
    try:
        pub.write(np.array([0], dtype=np.uint8), seq=1)
        pub.write(np.array([0], dtype=np.uint8), seq=3)  # gap!

        r1 = sub.read_full(timeout_ms=50)
        assert r1 is not None
        assert not sub.has_seq_gap(1)   # first read — no previous seq

        r2 = sub.read_full(timeout_ms=50)
        assert r2 is not None
        assert sub.has_seq_gap(3)       # gap: last=1, current=3
    finally:
        pub_bus.close(unlink=True); sub_bus.close()
