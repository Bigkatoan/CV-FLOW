"""
Tests for cv_flow.topic.parser

T-PARSE-01: source topic (only output:) → input_port.is_none=True
T-PARSE-02: sink topic (only input:) → output_port.is_none=True
T-PARSE-03: transform topic → both ports valid
T-PARSE-04: elastic: true + max_replicas: 4 → TopicDef.elastic=True, max_replicas=4
T-PARSE-05: queue_depth: 16 → TopicDef.queue_depth=16
T-PARSE-06: drop_mode: true → TopicDef.drop_mode=True
T-PARSE-07: filename "camera_frame.topic" → name="camera_frame"
T-PARSE-08: comment lines are ignored
T-PARSE-09: invalid dtype → ParseError with line number
T-PARSE-10: load_topics_dir() loads all *.topic in a directory
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from cv_flow.topic.parser import parse_topic_file, load_topics_dir, ParseError


def _write_topic(content: str, name: str = "test_topic") -> Path:
    """Write content to a temp .topic file, return its path."""
    tmp = Path(tempfile.mkdtemp())
    p = tmp / f"{name}.topic"
    p.write_text(content)
    return p


# ── T-PARSE-01 ────────────────────────────────────────────────────────────────

def test_parse_01_source_topic():
    """Source topic: only output: section → input_port.is_none=True."""
    p = _write_topic("""
output: -> cpu
   - frame : bgr8 shape=[720, 1280]
   - seq   : uint64
""")
    td = parse_topic_file(p)
    assert td.input_port.is_none
    assert not td.output_port.is_none
    assert len(td.output_port.fields) == 2
    assert td.output_port.fields[0].name == "frame"
    assert td.output_port.fields[0].full_shape == (720, 1280, 3)
    assert td.output_port.fields[1].full_shape == ()


# ── T-PARSE-02 ────────────────────────────────────────────────────────────────

def test_parse_02_sink_topic():
    """Sink topic: only input: section → output_port.is_none=True."""
    p = _write_topic("""
input: -> cpu
   - frame : bgr8 shape=[720, 1280]
""")
    td = parse_topic_file(p)
    assert not td.input_port.is_none
    assert td.output_port.is_none
    assert len(td.input_port.fields) == 1


# ── T-PARSE-03 ────────────────────────────────────────────────────────────────

def test_parse_03_transform_topic():
    """Transform topic: both input and output present."""
    p = _write_topic("""
input: -> cpu
   - frame : bgr8 shape=[720, 1280]

output: -> cpu
   - tensor : float32 shape=[1, 3, 640, 640]
""")
    td = parse_topic_file(p)
    assert not td.input_port.is_none
    assert not td.output_port.is_none
    assert td.input_port.fields[0].dtype_str  == "bgr8"
    assert td.output_port.fields[0].dtype_str == "float32"


# ── T-PARSE-04 ────────────────────────────────────────────────────────────────

def test_parse_04_elastic_max_replicas():
    """elastic: true + max_replicas: 8 → correct values on TopicDef."""
    p = _write_topic("""
elastic: true
max_replicas: 8

output: -> cpu
   - data : float32 shape=[256]
""")
    td = parse_topic_file(p)
    assert td.elastic
    assert td.max_replicas == 8


# ── T-PARSE-05 ────────────────────────────────────────────────────────────────

def test_parse_05_queue_depth():
    """queue_depth: 16 → TopicDef.queue_depth=16."""
    p = _write_topic("""
queue_depth: 16

output: -> cpu
   - x : float32 shape=[10]
""")
    td = parse_topic_file(p)
    assert td.queue_depth == 16


# ── T-PARSE-06 ────────────────────────────────────────────────────────────────

def test_parse_06_drop_mode():
    """drop_mode: true → TopicDef.drop_mode=True."""
    p = _write_topic("""
drop_mode: true

output: -> cpu
   - x : float32 shape=[10]
""")
    td = parse_topic_file(p)
    assert td.drop_mode


# ── T-PARSE-07 ────────────────────────────────────────────────────────────────

def test_parse_07_name_from_filename():
    """Topic name derived from file stem."""
    p = _write_topic("""
output: -> cpu
   - x : uint8 shape=[10]
""", name="camera_frame")
    td = parse_topic_file(p)
    assert td.name == "camera_frame"


# ── T-PARSE-08 ────────────────────────────────────────────────────────────────

def test_parse_08_comments_ignored():
    """Comment lines (# ...) are silently ignored."""
    p = _write_topic("""
# This is a comment
output: -> cpu
   # another comment
   - x : uint8 shape=[4]
# trailing comment
""")
    td = parse_topic_file(p)
    assert len(td.output_port.fields) == 1


# ── T-PARSE-09 ────────────────────────────────────────────────────────────────

def test_parse_09_invalid_dtype_raises():
    """Invalid dtype → ParseError with line number information."""
    p = _write_topic("""
output: -> cpu
   - x : nonexistent_dtype shape=[4]
""")
    with pytest.raises(ParseError) as exc_info:
        parse_topic_file(p)
    assert exc_info.value.line_no is not None
    assert "nonexistent_dtype" in str(exc_info.value) or "Unknown dtype" in str(exc_info.value)


# ── T-PARSE-10 ────────────────────────────────────────────────────────────────

def test_parse_10_load_topics_dir():
    """load_topics_dir() loads all *.topic files in a directory."""
    import tempfile
    tmp = Path(tempfile.mkdtemp())

    (tmp / "camera_frame.topic").write_text("""
output: -> cpu
   - frame : bgr8 shape=[480, 640]
""")
    (tmp / "detections.topic").write_text("""
input: -> cpu
   - boxes : float32 shape=[100, 4]
output: -> cpu
   - tracked : float32 shape=[100, 5]
""")
    # Non-.topic file should be ignored
    (tmp / "README.txt").write_text("not a topic")

    topics = load_topics_dir(tmp)
    assert set(topics.keys()) == {"camera_frame", "detections"}
    assert topics["camera_frame"].input_port.is_none
    assert topics["detections"].output_port.fields[0].name == "tracked"
