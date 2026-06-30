"""
Tests for backend.app.api.pipeline — CRUD + validate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.pipeline_store import store

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_store():
    store.clear()
    yield
    store.clear()


def _sample_spec() -> dict:
    return {
        "name": "Test Pipeline",
        "description": "A minimal valid pipeline",
        "topics": [
            {
                "name": "camera_frame",
                "device": "cpu",
                "fields": [{"name": "frame", "dtype": "bgr8", "shape": [720, 1280]}],
            }
        ],
        "nodes": [
            {
                "id": "cam_0",
                "type": "CameraSource",
                "config": {"device_index": 0},
                "connections_out": [{"slot": "frame_out", "topic": "camera_frame"}],
            }
        ],
    }


# ── create / get ──────────────────────────────────────────────────────────────

def test_create_and_get_pipeline():
    """POST /api/pipeline creates a record; GET /api/pipeline/{id} retrieves it."""
    resp = client.post("/api/pipeline", json=_sample_spec())
    assert resp.status_code == 201
    body = resp.json()
    assert "id" in body
    pipeline_id = body["id"]

    resp2 = client.get(f"/api/pipeline/{pipeline_id}")
    assert resp2.status_code == 200
    assert resp2.json()["name"] == "Test Pipeline"


def test_get_unknown_pipeline_404():
    resp = client.get("/api/pipeline/does-not-exist")
    assert resp.status_code == 404


# ── update ────────────────────────────────────────────────────────────────────

def test_update_pipeline():
    """PUT /api/pipeline/{id} updates an existing record."""
    created = client.post("/api/pipeline", json=_sample_spec()).json()
    pipeline_id = created["id"]

    updated_spec = _sample_spec()
    updated_spec["name"] = "Renamed Pipeline"
    resp = client.put(f"/api/pipeline/{pipeline_id}", json=updated_spec)
    assert resp.status_code == 200
    assert resp.json()["name"] == "Renamed Pipeline"


def test_update_unknown_pipeline_404():
    resp = client.put("/api/pipeline/does-not-exist", json=_sample_spec())
    assert resp.status_code == 404


# ── delete ────────────────────────────────────────────────────────────────────

def test_delete_pipeline():
    """DELETE /api/pipeline/{id} removes the record (204, then 404 on re-fetch)."""
    created = client.post("/api/pipeline", json=_sample_spec()).json()
    pipeline_id = created["id"]

    resp = client.delete(f"/api/pipeline/{pipeline_id}")
    assert resp.status_code == 204

    resp2 = client.get(f"/api/pipeline/{pipeline_id}")
    assert resp2.status_code == 404


def test_delete_unknown_pipeline_404():
    resp = client.delete("/api/pipeline/does-not-exist")
    assert resp.status_code == 404


# ── list ──────────────────────────────────────────────────────────────────────

def test_list_pipelines():
    client.post("/api/pipeline", json=_sample_spec())
    client.post("/api/pipeline", json=_sample_spec())

    resp = client.get("/api/pipeline")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


# ── validate ──────────────────────────────────────────────────────────────────

def test_validate_valid_pipeline():
    resp = client.post("/api/pipeline/validate", json=_sample_spec())
    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is True


def test_validate_unknown_node_type():
    spec = _sample_spec()
    spec["nodes"][0]["type"] = "NotARealNodeType"
    resp = client.post("/api/pipeline/validate", json=spec)
    body = resp.json()
    assert body["valid"] is False
    assert any("Unknown node type" in issue["message"] for issue in body["issues"])


def test_validate_undeclared_topic_reference():
    spec = _sample_spec()
    spec["nodes"][0]["connections_out"][0]["topic"] = "nonexistent_topic"
    resp = client.post("/api/pipeline/validate", json=spec)
    body = resp.json()
    assert body["valid"] is False
    assert any("undeclared topic" in issue["message"] for issue in body["issues"])


def test_validate_missing_required_parameter():
    spec = _sample_spec()
    spec["nodes"][0]["type"] = "RtspSource"  # requires 'url' parameter
    spec["nodes"][0]["config"] = {}
    resp = client.post("/api/pipeline/validate", json=spec)
    body = resp.json()
    assert body["valid"] is False
    assert any("Missing required parameter" in issue["message"] for issue in body["issues"])


def test_validate_warns_on_unconnected_input():
    """A node with an unconnected required input slot produces a warning, not an error."""
    spec = _sample_spec()
    spec["nodes"][0]["type"] = "Preprocess"
    spec["nodes"][0]["config"] = {}
    spec["nodes"][0]["connections_in"] = []  # frame_in not wired
    spec["nodes"][0]["connections_out"] = []
    resp = client.post("/api/pipeline/validate", json=spec)
    body = resp.json()
    warnings = [i for i in body["issues"] if i["level"] == "warning"]
    assert any("not connected" in w["message"] for w in warnings)
