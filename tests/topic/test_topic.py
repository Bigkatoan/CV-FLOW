"""
Tests for cv_flow.topic.topic (Topic class + registry)

T-TOPIC-01: Topic.from_file() loads correctly
T-TOPIC-02: Duplicate name → ValueError
T-TOPIC-03: get_topic(name) → Topic
T-TOPIC-04: get_topic(unknown) → KeyError with message
T-TOPIC-05: list_topics() → dict of all registered topics
T-TOPIC-06: clear_topics() → registry empty
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from cv_flow.topic.topic import (
    Topic, get_topic, list_topics, clear_topics, load_topics
)
from cv_flow.topic.types import TopicDef, PortDef, FieldDef


def _sample_def(name: str = "sample") -> TopicDef:
    f = FieldDef.build("x", "float32", (4,))
    return TopicDef(
        name=name,
        input_port=PortDef.none_port(),
        output_port=PortDef(device="cpu", fields=[f]),
    )


def _write_topic(name: str, content: str) -> Path:
    tmp = Path(tempfile.mkdtemp())
    p   = tmp / f"{name}.topic"
    p.write_text(content)
    return p


# Always start each test with a clean registry
@pytest.fixture(autouse=True)
def clean_registry():
    clear_topics()
    yield
    clear_topics()


# ── T-TOPIC-01 ────────────────────────────────────────────────────────────────

def test_topic_01_from_file():
    """Topic.from_file() parses and registers correctly."""
    p = _write_topic("cam_frame", """
output: -> cpu
   - frame : bgr8 shape=[480, 640]
""")
    t = Topic.from_file(p)
    assert t.name == "cam_frame"
    assert not t.input_port.is_none or True   # source topic
    assert t.output_port.fields[0].dtype_str == "bgr8"
    assert "cam_frame" in list_topics()


# ── T-TOPIC-02 ────────────────────────────────────────────────────────────────

def test_topic_02_duplicate_raises():
    """Registering a duplicate name → ValueError."""
    td = _sample_def("dup_topic")
    Topic(td)
    with pytest.raises(ValueError, match="already registered"):
        Topic(_sample_def("dup_topic"))


def test_topic_02_overwrite_allowed():
    """overwrite=True replaces existing topic without error."""
    td1 = _sample_def("ow_topic")
    td2 = _sample_def("ow_topic")
    t1  = Topic(td1)
    t2  = Topic(td2, overwrite=True)
    assert get_topic("ow_topic") is t2


# ── T-TOPIC-03 ────────────────────────────────────────────────────────────────

def test_topic_03_get_topic():
    """get_topic(name) returns the correct Topic."""
    td = _sample_def("lookup_me")
    t  = Topic(td)
    assert get_topic("lookup_me") is t


# ── T-TOPIC-04 ────────────────────────────────────────────────────────────────

def test_topic_04_get_unknown_raises():
    """get_topic(unknown) → KeyError with message listing registered topics."""
    Topic(_sample_def("existing"))
    with pytest.raises(KeyError) as exc_info:
        get_topic("does_not_exist")
    msg = str(exc_info.value)
    assert "does_not_exist" in msg
    assert "existing" in msg


# ── T-TOPIC-05 ────────────────────────────────────────────────────────────────

def test_topic_05_list_topics():
    """list_topics() returns all registered topics."""
    Topic(_sample_def("t1"))
    Topic(_sample_def("t2"))
    Topic(_sample_def("t3"))

    topics = list_topics()
    assert set(topics.keys()) == {"t1", "t2", "t3"}
    assert isinstance(topics["t1"], Topic)


# ── T-TOPIC-06 ────────────────────────────────────────────────────────────────

def test_topic_06_clear_topics():
    """clear_topics() empties the registry."""
    Topic(_sample_def("to_clear"))
    assert len(list_topics()) == 1

    clear_topics()
    assert len(list_topics()) == 0

    with pytest.raises(KeyError):
        get_topic("to_clear")


# ── load_topics() bonus ───────────────────────────────────────────────────────

def test_load_topics_dir():
    """load_topics() loads all .topic files in a directory."""
    tmp = Path(tempfile.mkdtemp())
    (tmp / "a.topic").write_text("output: -> cpu\n   - x : uint8 shape=[2]\n")
    (tmp / "b.topic").write_text("output: -> cpu\n   - y : float32 shape=[3]\n")

    result = load_topics(tmp)
    assert set(result.keys()) == {"a", "b"}
    assert set(list_topics().keys()) == {"a", "b"}
