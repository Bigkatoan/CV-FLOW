"""
Tests for cv_flow.nodes.draw (draw_boxes pure function + DrawBbox Node).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

cv2 = pytest.importorskip("cv2")

from cv_flow.nodes.draw import draw_boxes, DrawBbox
from cv_flow.topic.topic import Topic, clear_topics
from cv_flow.topic.types import TopicDef, PortDef, FieldDef
from cv_flow.topic.publisher import Publisher
from cv_flow.dam.bus import PortBus


@pytest.fixture(autouse=True)
def clean_registry():
    clear_topics()
    yield
    clear_topics()


# ── draw_boxes() pure function ────────────────────────────────────────────────

def test_draw_boxes_modifies_pixels():
    """Drawing a box changes pixel values inside the frame (vs. a blank frame)."""
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
    out = draw_boxes(frame, boxes, class_ids=np.array([0]))
    assert not np.array_equal(out, frame), "draw_boxes() did not modify the frame"
    assert out.shape == frame.shape


def test_draw_boxes_skips_padding_sentinel():
    """Boxes with class_id == -1 (NMS padding) are not drawn."""
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
    out = draw_boxes(frame, boxes, class_ids=np.array([-1]))
    assert np.array_equal(out, frame), "Padding sentinel box should not be drawn"


def test_draw_boxes_does_not_mutate_input():
    """draw_boxes() returns a copy, leaving the input frame untouched."""
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    original = frame.copy()
    boxes = np.array([[5, 5, 20, 20]], dtype=np.float32)
    draw_boxes(frame, boxes, class_ids=np.array([0]))
    assert np.array_equal(frame, original)


# ── DrawBbox Node ─────────────────────────────────────────────────────────────

def test_drawbbox_node_pipeline():
    """DrawBbox Node reads frame + dets topics, publishes annotated frame."""
    frame_field = FieldDef.build("frame", "bgr8", (32, 32))
    Topic(TopicDef(
        name="frame_test",
        input_port=PortDef.none_port(),
        output_port=PortDef(device="cpu", fields=[frame_field]),
    ))

    det_fields = [
        FieldDef.build("boxes",     "float32", (4, 4)),
        FieldDef.build("scores",    "float32", (4,)),
        FieldDef.build("class_ids", "int32",   (4,)),
    ]
    Topic(TopicDef(
        name="dets_test2",
        input_port=PortDef.none_port(),
        output_port=PortDef(device="cpu", fields=det_fields),
    ))

    out_field = FieldDef.build("frame", "bgr8", (32, 32))
    Topic(TopicDef(
        name="annotated_test",
        input_port=PortDef.none_port(),
        output_port=PortDef(device="cpu", fields=[out_field]),
    ))

    session = "draw-session"

    frame_bus = PortBus(f"frame_test_{session}", slot_bytes=frame_field.n_bytes,
                         queue_depth=4, create=True)
    dets_bus  = PortBus(f"dets_test2_{session}",
                         slot_bytes=PortDef(device="cpu", fields=det_fields).total_bytes,
                         queue_depth=4, create=True)

    frame_pub = Publisher(frame_bus, PortDef(device="cpu", fields=[frame_field]))
    dets_pub  = Publisher(dets_bus,  PortDef(device="cpu", fields=det_fields))

    frame_arr = np.zeros((32, 32, 3), dtype=np.uint8)
    frame_pub.write(frame_arr)

    boxes     = np.array([[2, 2, 20, 20], [0,0,0,0], [0,0,0,0], [0,0,0,0]], dtype=np.float32)
    scores    = np.array([0.95, 0, 0, 0], dtype=np.float32)
    class_ids = np.array([1, -1, -1, -1], dtype=np.int32)
    dets_pub.write({"boxes": boxes, "scores": scores, "class_ids": class_ids})

    node = DrawBbox("frame_test", "dets_test2", "annotated_test")
    node._session_id = session
    node.initialize()
    node.spin_once()

    out_bus = PortBus(f"annotated_test_{session}", slot_bytes=out_field.n_bytes,
                       create=False)
    result = out_bus.read(timeout_ms=200)
    assert result is not None
    raw_out = result[0]
    out_frame = np.frombuffer(raw_out, dtype=np.uint8).reshape(32, 32, 3)
    assert not np.array_equal(out_frame, frame_arr), "Annotated frame should differ from input"

    frame_bus.close(unlink=True)
    dets_bus.close(unlink=True)
    out_bus.close(unlink=True)
    for p in node._publishers:
        p._bus.close(unlink=True)
