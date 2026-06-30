"""
Tests for cv_flow.nodes.preprocess (letterbox_resize, normalize_chw,
Preprocess Node, GrayscaleConvert Node).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
cv2 = pytest.importorskip("cv2")

from cv_flow.nodes.preprocess import letterbox_resize, normalize_chw, Preprocess, GrayscaleConvert
from cv_flow.topic.topic import Topic, clear_topics
from cv_flow.topic.types import TopicDef, PortDef, FieldDef
from cv_flow.topic.publisher import Publisher
from cv_flow.dam.bus import PortBus


@pytest.fixture(autouse=True)
def clean_registry():
    clear_topics()
    yield
    clear_topics()


# ── letterbox_resize ──────────────────────────────────────────────────────────

def test_letterbox_resize_correct_output_shape():
    """letterbox_resize() always returns exactly (target_h, target_w, 3)."""
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    out = letterbox_resize(frame, 640, 640, keep_aspect=True)
    assert out.shape == (640, 640, 3)


def test_letterbox_resize_stretch_mode():
    """keep_aspect=False stretches to target shape directly."""
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    out = letterbox_resize(frame, 320, 320, keep_aspect=False)
    assert out.shape == (320, 320, 3)


def test_letterbox_resize_pads_with_zeros():
    """keep_aspect=True pads non-square input with black borders."""
    frame = np.full((100, 200, 3), 255, dtype=np.uint8)  # wide white frame
    out = letterbox_resize(frame, 200, 200, keep_aspect=True)
    # top-left corner should be padding (black) since frame is wider than tall
    assert out[0, 0].sum() == 0


# ── normalize_chw ─────────────────────────────────────────────────────────────

def test_normalize_chw_shape():
    """normalize_chw() returns (1, 3, H, W) float32."""
    frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    tensor = normalize_chw(frame, normalize="imagenet")
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32


def test_normalize_chw_01_range():
    """normalize='[0,1]' scales pixel values into [0, 1]."""
    frame = np.full((4, 4, 3), 255, dtype=np.uint8)
    tensor = normalize_chw(frame, normalize="[0,1]")
    assert np.allclose(tensor, 1.0)


def test_normalize_chw_none():
    """normalize='none' leaves raw pixel values untouched (just transposed)."""
    frame = np.full((4, 4, 3), 100, dtype=np.uint8)
    tensor = normalize_chw(frame, normalize="none")
    assert np.allclose(tensor, 100.0)


# ── Preprocess Node ───────────────────────────────────────────────────────────

def test_preprocess_node_pipeline():
    """Preprocess Node reads bgr8 frame, writes float32 CHW tensor."""
    frame_field = FieldDef.build("frame", "bgr8", (32, 32))
    Topic(TopicDef(
        name="pre_in",
        input_port=PortDef.none_port(),
        output_port=PortDef(device="cpu", fields=[frame_field]),
    ))

    tensor_field = FieldDef.build("tensor", "float32", (1, 3, 16, 16))
    Topic(TopicDef(
        name="pre_out",
        input_port=PortDef.none_port(),
        output_port=PortDef(device="cpu", fields=[tensor_field]),
    ))

    session = "pre-session"
    in_bus = PortBus(f"pre_in_{session}", slot_bytes=frame_field.n_bytes,
                      queue_depth=4, create=True)
    in_pub = Publisher(in_bus, PortDef(device="cpu", fields=[frame_field]))
    in_pub.write(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))

    node = Preprocess("pre_in", "pre_out", width=16, height=16, normalize="none")
    node._session_id = session
    node.initialize()
    node.spin_once()

    out_bus = PortBus(f"pre_out_{session}", slot_bytes=tensor_field.n_bytes, create=False)
    result = out_bus.read(timeout_ms=200)
    assert result is not None
    tensor = np.frombuffer(result[0], dtype=np.float32).reshape(1, 3, 16, 16)
    assert tensor.shape == (1, 3, 16, 16)

    in_bus.close(unlink=True)
    out_bus.close(unlink=True)
    for p in node._publishers:
        p._bus.close(unlink=True)


# ── GrayscaleConvert Node ─────────────────────────────────────────────────────

def test_grayscale_convert_node():
    """GrayscaleConvert Node converts bgr8 -> mono8."""
    frame_field = FieldDef.build("frame", "bgr8", (16, 16))
    Topic(TopicDef(
        name="gray_in",
        input_port=PortDef.none_port(),
        output_port=PortDef(device="cpu", fields=[frame_field]),
    ))

    gray_field = FieldDef.build("frame", "mono8", (16, 16))
    Topic(TopicDef(
        name="gray_out",
        input_port=PortDef.none_port(),
        output_port=PortDef(device="cpu", fields=[gray_field]),
    ))

    session = "gray-session"
    in_bus = PortBus(f"gray_in_{session}", slot_bytes=frame_field.n_bytes,
                      queue_depth=4, create=True)
    in_pub = Publisher(in_bus, PortDef(device="cpu", fields=[frame_field]))
    in_pub.write(np.full((16, 16, 3), 200, dtype=np.uint8))

    node = GrayscaleConvert("gray_in", "gray_out")
    node._session_id = session
    node.initialize()
    node.spin_once()

    out_bus = PortBus(f"gray_out_{session}", slot_bytes=gray_field.n_bytes, create=False)
    result = out_bus.read(timeout_ms=200)
    assert result is not None
    gray = np.frombuffer(result[0], dtype=np.uint8).reshape(16, 16)
    assert gray.shape == (16, 16)
    assert gray[0, 0] == 200  # constant input -> constant grayscale output

    in_bus.close(unlink=True)
    out_bus.close(unlink=True)
    for p in node._publishers:
        p._bus.close(unlink=True)
