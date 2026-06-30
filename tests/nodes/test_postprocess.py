"""
Tests for cv_flow.nodes.postprocess (NMS pure-numpy logic + Node wrapper).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from cv_flow.nodes.postprocess import nms, run_nms, NMS
from cv_flow.topic.topic import Topic, clear_topics
from cv_flow.topic.types import TopicDef, PortDef, FieldDef
from cv_flow.executor import Executor


@pytest.fixture(autouse=True)
def clean_registry():
    clear_topics()
    yield
    clear_topics()


# ── raw nms() ─────────────────────────────────────────────────────────────────

def test_nms_removes_overlapping_boxes():
    """Two heavily-overlapping boxes with different scores → only the higher kept."""
    boxes  = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.95], dtype=np.float32)
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert 0 in keep        # highest-scoring of the overlapping pair
    assert 1 not in keep    # suppressed
    assert 2 in keep        # far away, independent box


def test_nms_empty_input():
    """nms() on empty boxes array returns empty result."""
    boxes  = np.zeros((0, 4), dtype=np.float32)
    scores = np.zeros((0,),   dtype=np.float32)
    keep = nms(boxes, scores)
    assert len(keep) == 0


# ── run_nms() ─────────────────────────────────────────────────────────────────

def test_run_nms_yolov8_format():
    """run_nms() on synthetic yolov8-format raw output filters by confidence."""
    n_classes = 80
    n_boxes   = 100
    raw = np.zeros((1, 4 + n_classes, n_boxes), dtype=np.float32)

    # box 0: high confidence, class 5
    raw[0, 0:4, 0] = [100, 100, 20, 20]   # cx,cy,w,h
    raw[0, 4 + 5, 0] = 0.9

    # box 1: low confidence (below threshold)
    raw[0, 0:4, 1] = [200, 200, 20, 20]
    raw[0, 4 + 3, 1] = 0.1

    boxes, scores, class_ids = run_nms(raw, confidence_threshold=0.4, max_detections=10)

    assert scores[0] > 0.4
    assert class_ids[0] == 5
    # box 1 should not appear since its score is below threshold
    assert not np.any((scores > 0) & (class_ids == 3))


def test_run_nms_respects_max_detections():
    """Output arrays are always padded/truncated to max_detections length."""
    n_classes = 80
    raw = np.zeros((1, 4 + n_classes, 5), dtype=np.float32)
    for i in range(5):
        raw[0, 0:4, i] = [10 * i, 10 * i, 5, 5]
        raw[0, 4, i] = 0.9   # class 0, high conf

    boxes, scores, class_ids = run_nms(raw, confidence_threshold=0.3, max_detections=3)
    assert boxes.shape == (3, 4)
    assert scores.shape == (3,)
    assert class_ids.shape == (3,)


# ── NMS Node integration ──────────────────────────────────────────────────────

def test_nms_node_pipeline():
    """NMS Node reads raw tensor from input topic, writes detections to output topic."""
    raw_field = FieldDef.build("raw", "float32", (1, 84, 100))
    Topic(TopicDef(
        name="raw_test",
        input_port=PortDef.none_port(),
        output_port=PortDef(device="cpu", fields=[raw_field]),
    ))

    det_fields = [
        FieldDef.build("boxes",     "float32", (8, 4)),
        FieldDef.build("scores",    "float32", (8,)),
        FieldDef.build("class_ids", "int32",   (8,)),
    ]
    Topic(TopicDef(
        name="dets_test",
        input_port=PortDef.none_port(),
        output_port=PortDef(device="cpu", fields=det_fields),
    ))

    from cv_flow.dam.bus import PortBus
    from cv_flow.topic.publisher import Publisher

    session = "test-session"
    raw_bus = PortBus(f"raw_test_{session}", slot_bytes=raw_field.n_bytes,
                       queue_depth=4, create=True)
    pub = Publisher(raw_bus, PortDef(device="cpu", fields=[raw_field]))

    raw_arr = np.zeros((1, 84, 100), dtype=np.float32)
    raw_arr[0, 0:4, 0] = [50, 50, 10, 10]
    raw_arr[0, 4, 0] = 0.99
    pub.write(raw_arr)

    node = NMS("raw_test", "dets_test", max_detections=8)
    node._session_id = session
    node.initialize()
    node.spin_once()

    out_bus = PortBus(f"dets_test_{session}",
                       slot_bytes=PortDef(device="cpu", fields=det_fields).total_bytes,
                       create=False)
    result = out_bus.read(timeout_ms=200)
    assert result is not None
    raw_out, seq, _, _ = result
    scores = np.frombuffer(raw_out[8*4*4 : 8*4*4 + 8*4], dtype=np.float32)
    assert scores[0] > 0.9

    raw_bus.close(unlink=True)
    out_bus.close(unlink=True)
    for p in node._publishers:
        p._bus.close(unlink=True)
