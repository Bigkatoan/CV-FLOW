"""
Tests for cv_flow.nodes.tracking (ByteTrackLite pure logic + ObjectTracker Node).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from cv_flow.nodes.tracking import ByteTrackLite, ObjectTracker
from cv_flow.topic.topic import Topic, clear_topics
from cv_flow.topic.types import TopicDef, PortDef, FieldDef
from cv_flow.topic.publisher import Publisher
from cv_flow.dam.bus import PortBus


@pytest.fixture(autouse=True)
def clean_registry():
    clear_topics()
    yield
    clear_topics()


# ── ByteTrackLite pure logic ──────────────────────────────────────────────────

def test_tracker_assigns_persistent_id():
    """Same object across frames (overlapping box) keeps the same track_id."""
    tracker = ByteTrackLite(max_age=10, min_hits=2, iou_threshold=0.3)

    box = np.array([10, 10, 50, 50], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    cls = np.array([0], dtype=np.int32)

    # Frame 1: not confirmed yet (hits=1 < min_hits=2)
    boxes1, _, _, ids1 = tracker.update(box.reshape(1, 4), scores, cls)
    assert len(ids1) == 0  # not confirmed yet

    # Frame 2: same box position → matched, now confirmed
    boxes2, _, _, ids2 = tracker.update(box.reshape(1, 4), scores, cls)
    assert len(ids2) == 1
    track_id_frame2 = ids2[0]

    # Frame 3: still same position → same track_id persists
    boxes3, _, _, ids3 = tracker.update(box.reshape(1, 4), scores, cls)
    assert len(ids3) == 1
    assert ids3[0] == track_id_frame2


def test_tracker_drops_stale_tracks():
    """Track not seen for > max_age frames is dropped."""
    tracker = ByteTrackLite(max_age=2, min_hits=1, iou_threshold=0.3)

    box = np.array([[10, 10, 50, 50]], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    cls = np.array([0], dtype=np.int32)

    tracker.update(box, scores, cls)
    assert len(tracker._tracks) == 1

    empty = np.zeros((0, 4), dtype=np.float32)
    empty_s = np.zeros((0,), dtype=np.float32)
    empty_c = np.zeros((0,), dtype=np.int32)

    for _ in range(5):  # well beyond max_age=2
        tracker.update(empty, empty_s, empty_c)

    assert len(tracker._tracks) == 0


def test_tracker_ignores_padding_sentinel():
    """class_id == -1 (NMS padding) entries are ignored, not tracked."""
    tracker = ByteTrackLite(max_age=10, min_hits=1, iou_threshold=0.3)
    boxes = np.array([[0, 0, 0, 0], [10, 10, 30, 30]], dtype=np.float32)
    scores = np.array([0, 0.9], dtype=np.float32)
    cls = np.array([-1, 2], dtype=np.int32)

    _, _, out_cls, out_ids = tracker.update(boxes, scores, cls)
    assert len(out_ids) == 1
    assert out_cls[0] == 2


# ── ObjectTracker Node ────────────────────────────────────────────────────────

def test_object_tracker_node_pipeline():
    """ObjectTracker Node reads dets, writes tracked output with track_ids."""
    det_fields = [
        FieldDef.build("boxes",     "float32", (2, 4)),
        FieldDef.build("scores",    "float32", (2,)),
        FieldDef.build("class_ids", "int32",   (2,)),
    ]
    Topic(TopicDef(
        name="dets_in_test",
        input_port=PortDef.none_port(),
        output_port=PortDef(device="cpu", fields=det_fields),
    ))

    tracked_fields = [
        FieldDef.build("boxes",     "float32", (2, 4)),
        FieldDef.build("scores",    "float32", (2,)),
        FieldDef.build("class_ids", "int32",   (2,)),
        FieldDef.build("track_ids", "int32",   (2,)),
    ]
    Topic(TopicDef(
        name="tracked_test",
        input_port=PortDef.none_port(),
        output_port=PortDef(device="cpu", fields=tracked_fields),
    ))

    session = "track-session"
    in_bus = PortBus(f"dets_in_test_{session}",
                      slot_bytes=PortDef(device="cpu", fields=det_fields).total_bytes,
                      queue_depth=4, create=True)
    in_pub = Publisher(in_bus, PortDef(device="cpu", fields=det_fields))

    boxes  = np.array([[10, 10, 50, 50], [0, 0, 0, 0]], dtype=np.float32)
    scores = np.array([0.9, 0], dtype=np.float32)
    cls    = np.array([1, -1], dtype=np.int32)

    node = ObjectTracker("dets_in_test", "tracked_test", min_hits=1, max_tracks=2)
    node._session_id = session
    node.initialize()

    in_pub.write({"boxes": boxes, "scores": scores, "class_ids": cls})
    node.spin_once()

    out_bus = PortBus(f"tracked_test_{session}",
                       slot_bytes=PortDef(device="cpu", fields=tracked_fields).total_bytes,
                       create=False)
    result = out_bus.read(timeout_ms=200)
    assert result is not None
    raw_out = result[0]

    n = 2
    boxes_bytes  = n * 4 * 4
    scores_bytes = n * 4
    cls_bytes    = n * 4
    track_ids = np.frombuffer(
        raw_out[boxes_bytes + scores_bytes + cls_bytes : boxes_bytes + scores_bytes + cls_bytes + n * 4],
        dtype=np.int32,
    )
    assert track_ids[0] != 0

    in_bus.close(unlink=True)
    out_bus.close(unlink=True)
    for p in node._publishers:
        p._bus.close(unlink=True)
