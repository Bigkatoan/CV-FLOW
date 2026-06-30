"""
Tests for cv_flow.node.Node

T-NODE-01: advertise(topic_obj) → Publisher
T-NODE-02: advertise("topic_name") → Publisher (registry lookup)
T-NODE-03: subscribe(topic_obj) → Subscriber
T-NODE-04: subscribe("topic_name") → Subscriber
T-NODE-05: _session_id is injected by Executor
T-NODE-06: initialize() is called exactly once before spin
T-NODE-07: shutdown() is called when Executor stops
T-NODE-08: StopIteration from spin_once() stops Executor
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from cv_flow.node import Node
from cv_flow.executor import Executor
from cv_flow.topic.topic import Topic, clear_topics
from cv_flow.topic.types import TopicDef, PortDef, FieldDef


@pytest.fixture(autouse=True)
def clean_registry():
    clear_topics()
    yield
    clear_topics()


def _make_topic(name: str = "test_topic") -> Topic:
    f   = FieldDef.build("x", "float32", (4,))
    td  = TopicDef(
        name=name,
        input_port=PortDef.none_port(),
        output_port=PortDef(device="cpu", fields=[f]),
    )
    return Topic(td)


class _CountNode(Node):
    """Node that counts initialize/shutdown calls and stops after N spins."""
    def __init__(self, max_spins=2):
        super().__init__()
        self.init_count     = 0
        self.shutdown_count = 0
        self.spin_count     = 0
        self._max_spins     = max_spins

    def initialize(self): self.init_count += 1
    def shutdown(self):   self.shutdown_count += 1
    def spin_once(self):
        self.spin_count += 1
        if self.spin_count >= self._max_spins:
            raise StopIteration


# ── T-NODE-01 ─────────────────────────────────────────────────────────────────

def test_node_01_advertise_topic_obj():
    """advertise(topic_obj) returns a Publisher."""
    from cv_flow.topic.publisher import Publisher
    t    = _make_topic()
    node = Node()
    node._session_id = "sess1"
    pub  = node.advertise(t)
    assert isinstance(pub, Publisher)
    # cleanup
    for p in node._publishers:
        p._bus.close(unlink=True)


# ── T-NODE-02 ─────────────────────────────────────────────────────────────────

def test_node_02_advertise_topic_name():
    """advertise('name') looks up from registry and returns Publisher."""
    from cv_flow.topic.publisher import Publisher
    _make_topic("named_topic")
    node = Node()
    node._session_id = "sess2"
    pub  = node.advertise("named_topic")
    assert isinstance(pub, Publisher)
    for p in node._publishers:
        p._bus.close(unlink=True)


# ── T-NODE-03 ─────────────────────────────────────────────────────────────────

def test_node_03_subscribe_topic_obj():
    """subscribe(topic_obj) returns a Subscriber."""
    from cv_flow.topic.subscriber import Subscriber
    from cv_flow.topic.publisher import Publisher

    t    = _make_topic("sub_test3")
    node_pub = Node(); node_pub._session_id = "s3"
    pub  = node_pub.advertise(t)

    node_sub = Node(); node_sub._session_id = "s3"
    sub  = node_sub.subscribe(t)
    assert isinstance(sub, Subscriber)

    for p in node_pub._publishers:
        p._bus.close(unlink=True)


# ── T-NODE-04 ─────────────────────────────────────────────────────────────────

def test_node_04_subscribe_topic_name():
    """subscribe('name') looks up from registry."""
    from cv_flow.topic.subscriber import Subscriber

    _make_topic("sub_test4")
    node_pub = Node(); node_pub._session_id = "s4"
    node_pub.advertise("sub_test4")

    node_sub = Node(); node_sub._session_id = "s4"
    sub = node_sub.subscribe("sub_test4")
    assert isinstance(sub, Subscriber)

    for p in node_pub._publishers:
        p._bus.close(unlink=True)


# ── T-NODE-05 ─────────────────────────────────────────────────────────────────

def test_node_05_session_id_injected():
    """Executor injects _session_id into nodes before spin."""
    node = _CountNode(max_spins=1)
    assert node._session_id is None

    exec_ = Executor([node])
    exec_.spin()

    assert node._session_id == exec_._session_id


# ── T-NODE-06 ─────────────────────────────────────────────────────────────────

def test_node_06_initialize_called_once():
    """initialize() is called exactly once before spin."""
    node = _CountNode(max_spins=3)
    Executor([node]).spin()
    assert node.init_count == 1


# ── T-NODE-07 ─────────────────────────────────────────────────────────────────

def test_node_07_shutdown_called():
    """shutdown() is called after spin loop ends."""
    node = _CountNode(max_spins=1)
    Executor([node]).spin()
    assert node.shutdown_count == 1


# ── T-NODE-08 ─────────────────────────────────────────────────────────────────

def test_node_08_stopiteration_stops_executor():
    """StopIteration from spin_once() stops spin and calls shutdown on all nodes."""
    n1 = _CountNode(max_spins=1)

    class _PassiveNode(Node):
        def __init__(self): super().__init__(); self.shutdown_count = 0
        def spin_once(self): pass
        def shutdown(self): self.shutdown_count += 1

    n2 = _PassiveNode()
    Executor([n1, n2]).spin()

    assert n1.spin_count >= 1
    assert n2.shutdown_count == 1
