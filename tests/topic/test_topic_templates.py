"""
Tests for cv_flow/topic_templates/*.topic — sample topic library.

Ensures every shipped template parses without error and has sane defaults.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from cv_flow.topic.parser import load_topics_dir

TEMPLATES_DIR = Path(__file__).parent.parent.parent / "cv_flow" / "topic_templates"

EXPECTED_TOPICS = {
    "camera_frame", "depth_frame", "yolo_input", "yolo_raw",
    "detections", "tracked", "annotated_frame", "stream_jpeg",
    "audio_pcm", "embedding",
}


def test_all_templates_parse():
    """Every *.topic file in topic_templates/ parses without raising."""
    topics = load_topics_dir(TEMPLATES_DIR)
    assert set(topics.keys()) == EXPECTED_TOPICS, (
        f"Expected {EXPECTED_TOPICS}, got {set(topics.keys())}"
    )


def test_camera_frame_is_source():
    """camera_frame.topic is a source topic (no input port)."""
    topics = load_topics_dir(TEMPLATES_DIR)
    assert topics["camera_frame"].input_port.is_none
    assert not topics["camera_frame"].output_port.is_none


def test_yolo_raw_is_elastic():
    """yolo_raw.topic declares elastic=True with max_replicas=4."""
    topics = load_topics_dir(TEMPLATES_DIR)
    td = topics["yolo_raw"]
    assert td.elastic
    assert td.max_replicas == 4


def test_stream_jpeg_drop_mode():
    """stream_jpeg.topic declares drop_mode=True for a fast sink."""
    topics = load_topics_dir(TEMPLATES_DIR)
    assert topics["stream_jpeg"].drop_mode


def test_all_templates_have_nonzero_total_bytes():
    """Every declared port (that isn't none) has total_bytes > 0."""
    topics = load_topics_dir(TEMPLATES_DIR)
    for name, td in topics.items():
        if not td.input_port.is_none:
            assert td.input_port.total_bytes > 0, f"{name}: input_port has 0 bytes"
        if not td.output_port.is_none:
            assert td.output_port.total_bytes > 0, f"{name}: output_port has 0 bytes"
