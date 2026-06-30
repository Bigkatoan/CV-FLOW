"""
Tests for cv_flow.topic.publisher.Publisher

T-PUB-01: write(dict) → correct bytes packed in field order
T-PUB-02: write(np.ndarray) single-field shortcut works
T-PUB-03: write wrong dtype → TypeError
T-PUB-04: write wrong shape → ValueError
T-PUB-05: auto-increment seq_no on each write
T-PUB-06: write(torch.Tensor on cpu) → cpu bytes written correctly
T-PUB-07: detections and metadata written correctly to PortBus
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from cv_flow.dam.bus import PortBus, Detection
from cv_flow.topic.types import FieldDef, PortDef
from cv_flow.topic.publisher import Publisher


def _unique(prefix="pub"):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _make_port(*field_specs):
    """field_specs: [(name, dtype_str, shape), ...]"""
    fields = [FieldDef.build(n, d, s) for n, d, s in field_specs]
    return PortDef(device="cpu", fields=fields)


# ── T-PUB-01 ──────────────────────────────────────────────────────────────────

def test_pub_01_dict_packed_in_order():
    """write(dict) → bytes packed sequentially in field definition order."""
    port = _make_port(
        ("a", "uint8",   (4,)),
        ("b", "float32", (2,)),
    )
    total = port.total_bytes
    bus = PortBus(_unique(), slot_bytes=total, queue_depth=4, create=True)
    pub = Publisher(bus, port)
    try:
        a_val = np.array([1, 2, 3, 4], dtype=np.uint8)
        b_val = np.array([1.5, 2.5], dtype=np.float32)
        pub.write({"a": a_val, "b": b_val})

        result = bus.read(timeout_ms=100)
        assert result is not None
        raw, seq, _, _ = result

        a_back = np.frombuffer(raw[0:4], dtype=np.uint8)
        b_back = np.frombuffer(raw[4:12], dtype=np.float32)
        np.testing.assert_array_equal(a_back, a_val)
        np.testing.assert_array_almost_equal(b_back, b_val)
    finally:
        bus.close(unlink=True)


# ── T-PUB-02 ──────────────────────────────────────────────────────────────────

def test_pub_02_single_field_ndarray():
    """Single-field topic: write(np.ndarray) directly (no dict wrapper)."""
    port  = _make_port(("x", "float32", (3,)))
    bus   = PortBus(_unique(), slot_bytes=12, queue_depth=4, create=True)
    pub   = Publisher(bus, port)
    try:
        val = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        pub.write(val)
        result = bus.read(timeout_ms=100)
        assert result is not None
        raw = result[0]
        back = np.frombuffer(raw[:12], dtype=np.float32)
        np.testing.assert_array_almost_equal(back, val)
    finally:
        bus.close(unlink=True)


# ── T-PUB-03 ──────────────────────────────────────────────────────────────────

def test_pub_03_wrong_type_raises():
    """write(string) for a numeric field → TypeError."""
    port = _make_port(("x", "float32", (4,)))
    bus  = PortBus(_unique(), slot_bytes=16, queue_depth=4, create=True)
    pub  = Publisher(bus, port)
    try:
        with pytest.raises(TypeError):
            pub.write("not an array")
    finally:
        bus.close(unlink=True)


# ── T-PUB-04 ──────────────────────────────────────────────────────────────────

def test_pub_04_wrong_shape_raises():
    """write(wrong-shape array) → ValueError."""
    port = _make_port(("frame", "bgr8", (4, 4)))   # expects shape (4,4,3)
    bus  = PortBus(_unique(), slot_bytes=48, queue_depth=4, create=True)
    pub  = Publisher(bus, port)
    try:
        wrong = np.zeros((4, 4, 1), dtype=np.uint8)   # wrong channel dim
        with pytest.raises(ValueError, match="shape"):
            pub.write(wrong)
    finally:
        bus.close(unlink=True)


# ── T-PUB-05 ──────────────────────────────────────────────────────────────────

def test_pub_05_auto_seq():
    """seq_no auto-increments on each write()."""
    port = _make_port(("x", "uint8", (1,)))
    bus  = PortBus(_unique(), slot_bytes=1, queue_depth=8, create=True)
    pub  = Publisher(bus, port)
    sub  = PortBus(bus.name, slot_bytes=1, create=False)
    try:
        for _ in range(4):
            pub.write(np.array([0], dtype=np.uint8))

        seqs = []
        for _ in range(4):
            r = sub.read(timeout_ms=50)
            assert r is not None
            seqs.append(r[1])

        assert seqs == [1, 2, 3, 4], f"Expected [1,2,3,4], got {seqs}"
    finally:
        bus.close(unlink=True)
        sub.close()


# ── T-PUB-06 ──────────────────────────────────────────────────────────────────

def test_pub_06_torch_tensor():
    """torch.Tensor on CPU → converted to bytes, written correctly."""
    pytest.importorskip("torch")
    import torch

    port = _make_port(("t", "float32", (3,)))
    bus  = PortBus(_unique(), slot_bytes=12, queue_depth=4, create=True)
    pub  = Publisher(bus, port)
    try:
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        pub.write(t)
        result = bus.read(timeout_ms=100)
        assert result is not None
        raw = result[0]
        back = np.frombuffer(raw[:12], dtype=np.float32)
        np.testing.assert_array_almost_equal(back, t.numpy())
    finally:
        bus.close(unlink=True)


# ── T-PUB-07 ──────────────────────────────────────────────────────────────────

def test_pub_07_detections_metadata():
    """Detections and metadata are correctly written to PortBus."""
    port = _make_port(("x", "uint8", (4,)))
    bus  = PortBus(_unique(), slot_bytes=4, queue_depth=4, create=True)
    sub  = PortBus(bus.name, slot_bytes=4, create=False)
    pub  = Publisher(bus, port)
    try:
        det = Detection(x1=0, y1=0, x2=10, y2=10, confidence=0.8,
                        class_id=2, class_name="car", track_id=99)
        pub.write(
            np.array([1, 2, 3, 4], dtype=np.uint8),
            detections=[det],
            metadata={"cam": "front"},
        )
        result = sub.read(timeout_ms=100)
        assert result is not None
        _, seq, dets, meta = result
        assert len(dets) == 1
        assert dets[0].class_name == "car"
        assert dets[0].track_id == 99
        assert meta["cam"] == "front"
    finally:
        bus.close(unlink=True)
        sub.close()
