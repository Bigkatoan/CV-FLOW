"""
Tests for backend.app.api.guide — POST /api/pipeline/generate-guide
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.guide_generator import generate_deployment_guide
from backend.app.schemas import PipelineSpec

client = TestClient(app)


def _sample_spec_dict() -> dict:
    return {
        "name": "YOLO Detection Pipeline",
        "description": "Real-time object detection from a USB camera",
        "topics": [
            {
                "name": "camera_frame",
                "device": "cpu",
                "fields": [{"name": "frame", "dtype": "bgr8", "shape": [720, 1280]}],
            },
            {
                "name": "preprocessed",
                "input_device": "cpu",
                "output_device": "cpu",
                "fields_in":  [{"name": "frame", "dtype": "bgr8", "shape": [720, 1280]}],
                "fields_out": [{"name": "tensor", "dtype": "float32", "shape": [1, 3, 640, 640]}],
            },
        ],
        "nodes": [
            {
                "id": "cam_0",
                "type": "CameraSource",
                "config": {"device_index": 0, "width": 1280, "height": 720},
                "connections_out": [{"slot": "frame_out", "topic": "camera_frame"}],
            },
            {
                "id": "pre_0",
                "type": "Preprocess",
                "config": {"width": 640, "height": 640},
                "connections_in":  [{"slot": "frame_in", "topic": "camera_frame"}],
                "connections_out": [{"slot": "tensor_out", "topic": "preprocessed"}],
            },
        ],
    }


# ── pure function ─────────────────────────────────────────────────────────────

def test_generate_guide_contains_topic_files():
    """Generated markdown includes a fenced .topic file block per declared topic."""
    spec = PipelineSpec(**_sample_spec_dict())
    md = generate_deployment_guide(spec)

    assert "topics/camera_frame.topic" in md
    assert "topics/preprocessed.topic" in md
    assert "frame : bgr8" in md
    assert "tensor : float32" in md


def test_generate_guide_contains_node_chain():
    """Generated markdown shows the node execution chain in order."""
    spec = PipelineSpec(**_sample_spec_dict())
    md = generate_deployment_guide(spec)
    assert "CameraSource -> Preprocess" in md


def test_generate_guide_contains_launch_script():
    """Generated markdown includes a launch.py code block referencing cv_flow."""
    spec = PipelineSpec(**_sample_spec_dict())
    md = generate_deployment_guide(spec)
    assert "import cv_flow" in md
    assert "cv_flow.load_topics" in md
    assert "cv-flow run my_pipeline/launch.py" in md


def test_generate_guide_elastic_topic_rendered():
    """elastic: true and max_replicas appear in the rendered topic file block."""
    spec_dict = _sample_spec_dict()
    spec_dict["topics"].append({
        "name": "yolo_raw",
        "input_device": "cpu",
        "output_device": "cpu",
        "fields_in":  [{"name": "tensor", "dtype": "float32", "shape": [1, 3, 640, 640]}],
        "fields_out": [{"name": "raw", "dtype": "float32", "shape": [1, 84, 8400]}],
        "elastic": True,
        "max_replicas": 6,
    })
    spec = PipelineSpec(**spec_dict)
    md = generate_deployment_guide(spec)
    assert "elastic: true" in md
    assert "max_replicas: 6" in md


# ── API endpoint ──────────────────────────────────────────────────────────────

def test_generate_guide_endpoint():
    """POST /api/pipeline/generate-guide returns markdown for a valid spec."""
    resp = client.post("/api/pipeline/generate-guide", json=_sample_spec_dict())
    assert resp.status_code == 200
    body = resp.json()
    assert "markdown" in body
    assert "# Pipeline: YOLO Detection Pipeline" in body["markdown"]
