"""
Tests for cv_flow.nodes.tee.Tee — fan a single topic out to N topics.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from cv_flow.nodes.tee import Tee
from cv_flow.topic.topic import Topic, clear_topics
from cv_flow.topic.types import TopicDef, PortDef, FieldDef
from cv_flow.topic.publisher import Publisher
from cv_flow.dam.bus import PortBus


@pytest.fixture(autouse=True)
def clean_registry():
    clear_topics()
    yield
    clear_topics()


def test_tee_republishes_to_all_output_topics():
    """A single input frame is independently readable from every output topic."""
    frame_field = FieldDef.build("frame", "bgr8", (8, 8))
    Topic(TopicDef(
        name="tee_in",
        input_port=PortDef.none_port(),
        output_port=PortDef(device="cpu", fields=[frame_field]),
    ))
    Topic(TopicDef(
        name="tee_out_a",
        input_port=PortDef.none_port(),
        output_port=PortDef(device="cpu", fields=[frame_field]),
    ))
    Topic(TopicDef(
        name="tee_out_b",
        input_port=PortDef.none_port(),
        output_port=PortDef(device="cpu", fields=[frame_field]),
    ))

    session = "tee-session"
    in_bus = PortBus(f"tee_in_{session}", slot_bytes=frame_field.n_bytes,
                      queue_depth=4, create=True)
    in_pub = Publisher(in_bus, PortDef(device="cpu", fields=[frame_field]))
    frame = np.full((8, 8, 3), 77, dtype=np.uint8)
    in_pub.write(frame)

    node = Tee("tee_in", ["tee_out_a", "tee_out_b"])
    node._session_id = session
    node.initialize()
    node.spin_once()

    bus_a = PortBus(f"tee_out_a_{session}", slot_bytes=frame_field.n_bytes, create=False)
    bus_b = PortBus(f"tee_out_b_{session}", slot_bytes=frame_field.n_bytes, create=False)
    result_a = bus_a.read(timeout_ms=200)
    result_b = bus_b.read(timeout_ms=200)
    assert result_a is not None
    assert result_b is not None
    out_a = np.frombuffer(result_a[0], dtype=np.uint8).reshape(8, 8, 3)
    out_b = np.frombuffer(result_b[0], dtype=np.uint8).reshape(8, 8, 3)
    assert np.all(out_a == 77)
    assert np.all(out_b == 77)

    in_bus.close(unlink=True)
    bus_a.close(unlink=True)
    bus_b.close(unlink=True)


def test_tee_solves_competing_subscribers_problem():
    """The actual bug Tee exists to fix: two nodes directly subscribing to the
    SAME topic compete for one FIFO and each only see some frames, not all.
    Routing through Tee first gives each downstream node its own queue."""
    frame_field = FieldDef.build("frame", "bgr8", (4, 4))
    Topic(TopicDef(
        name="src",
        input_port=PortDef.none_port(),
        output_port=PortDef(device="cpu", fields=[frame_field]),
    ))
    Topic(TopicDef(
        name="branch_a",
        input_port=PortDef.none_port(),
        output_port=PortDef(device="cpu", fields=[frame_field]),
    ))
    Topic(TopicDef(
        name="branch_b",
        input_port=PortDef.none_port(),
        output_port=PortDef(device="cpu", fields=[frame_field]),
    ))

    session = "tee-fanout-session"
    src_bus = PortBus(f"src_{session}", slot_bytes=frame_field.n_bytes,
                       queue_depth=4, create=True)
    src_pub = Publisher(src_bus, PortDef(device="cpu", fields=[frame_field]))

    tee = Tee("src", ["branch_a", "branch_b"])
    tee._session_id = session
    tee.initialize()

    for i in range(3):
        src_pub.write(np.full((4, 4, 3), i, dtype=np.uint8))
        tee.spin_once()

    bus_a = PortBus(f"branch_a_{session}", slot_bytes=frame_field.n_bytes, create=False)
    bus_b = PortBus(f"branch_b_{session}", slot_bytes=frame_field.n_bytes, create=False)

    # Every one of the 3 frames must be independently readable from BOTH branches.
    for expected in range(3):
        ra = bus_a.read(timeout_ms=200)
        rb = bus_b.read(timeout_ms=200)
        assert ra is not None and rb is not None
        assert np.frombuffer(ra[0], dtype=np.uint8)[0] == expected
        assert np.frombuffer(rb[0], dtype=np.uint8)[0] == expected

    src_bus.close(unlink=True)
    bus_a.close(unlink=True)
    bus_b.close(unlink=True)
