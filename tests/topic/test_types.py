"""
Tests for cv_flow.topic.types

T-TYPES-01: bgr8 shape=[720,1280] → full_shape=(720,1280,3), dtype=uint8
T-TYPES-02: float32 shape=[1,3,640,640] → full_shape=(1,3,640,640)
T-TYPES-03: uint64 shape=[] → full_shape=(), n_bytes=8
T-TYPES-04: unknown dtype → ValueError with clear message
T-TYPES-05: n_bytes = np.prod(full_shape) × itemsize
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from cv_flow.topic.types import DTYPE_MAP, FieldDef, PortDef, TopicDef


# ── T-TYPES-01 ────────────────────────────────────────────────────────────────

def test_types_01_bgr8_expansion():
    """bgr8 shape=[720,1280] → full_shape=(720,1280,3), dtype=uint8."""
    f = FieldDef.build("frame", "bgr8", (720, 1280))
    assert f.full_shape  == (720, 1280, 3)
    assert f.numpy_dtype == np.dtype(np.uint8)
    assert f.n_bytes     == 720 * 1280 * 3


# ── T-TYPES-02 ────────────────────────────────────────────────────────────────

def test_types_02_float32_no_expansion():
    """float32 shape=[1,3,640,640] → full_shape=(1,3,640,640)."""
    f = FieldDef.build("tensor", "float32", (1, 3, 640, 640))
    assert f.full_shape  == (1, 3, 640, 640)
    assert f.numpy_dtype == np.dtype(np.float32)
    assert f.n_bytes     == 1 * 3 * 640 * 640 * 4


# ── T-TYPES-03 ────────────────────────────────────────────────────────────────

def test_types_03_uint64_scalar():
    """uint64 shape=[] → full_shape=(), n_bytes=8."""
    f = FieldDef.build("seq", "uint64", ())
    assert f.full_shape  == ()
    assert f.n_bytes     == 8
    assert f.numpy_dtype == np.dtype(np.uint64)


# ── T-TYPES-04 ────────────────────────────────────────────────────────────────

def test_types_04_unknown_dtype_raises():
    """Unknown dtype → ValueError with message listing valid dtypes."""
    with pytest.raises(ValueError, match="Unknown dtype"):
        FieldDef.build("x", "imaginary_type", (10,))


# ── T-TYPES-05 ────────────────────────────────────────────────────────────────

def test_types_05_n_bytes_formula():
    """n_bytes = prod(full_shape) × itemsize for all dtypes."""
    for dtype_str, entry in DTYPE_MAP.items():
        base  = (4, 4)
        f     = FieldDef.build("x", dtype_str, base)
        prod  = int(np.prod(f.full_shape)) if f.full_shape else 1
        expected = prod * entry["numpy"].itemsize
        assert f.n_bytes == expected, \
            f"dtype={dtype_str}: expected {expected}, got {f.n_bytes}"


# ── PortDef / TopicDef sanity ─────────────────────────────────────────────────

def test_portdef_total_bytes():
    """PortDef.total_bytes sums all field bytes."""
    f1 = FieldDef.build("frame", "bgr8", (720, 1280))
    f2 = FieldDef.build("seq",   "uint64", ())
    port = PortDef(device="cpu", fields=[f1, f2])
    assert port.total_bytes == f1.n_bytes + f2.n_bytes


def test_portdef_none():
    """PortDef.none_port() → is_none=True, total_bytes=0."""
    port = PortDef.none_port()
    assert port.is_none
    assert port.total_bytes == 0


def test_topicdef_defaults():
    """TopicDef has correct default values."""
    td = TopicDef(
        name="test_topic",
        input_port=PortDef.none_port(),
        output_port=PortDef(device="cpu"),
    )
    assert td.elastic      == False
    assert td.max_replicas == 4
    assert td.queue_depth  == 8
    assert td.drop_mode    == False
