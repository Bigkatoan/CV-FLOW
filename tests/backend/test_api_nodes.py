"""
Tests for backend.app.api.nodes — GET /api/nodes, GET /api/nodes/{type}

T-CAT-04: GET /api/nodes -> returns the full catalog list
T-CAT-05: GET /api/nodes/{type} -> returns the correct node metadata
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from fastapi.testclient import TestClient

from backend.app.main import app
from cv_flow.nodes._catalog import NODE_CATALOG

client = TestClient(app)


# ── T-CAT-04 ──────────────────────────────────────────────────────────────────

def test_get_nodes_returns_full_catalog():
    """GET /api/nodes returns every node type defined in NODE_CATALOG."""
    resp = client.get("/api/nodes")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == set(NODE_CATALOG.keys())
    assert body["CameraSource"]["category"] == "input"


# ── T-CAT-05 ──────────────────────────────────────────────────────────────────

def test_get_node_by_type():
    """GET /api/nodes/{type} returns the correct single node's metadata."""
    resp = client.get("/api/nodes/YoloInference")
    assert resp.status_code == 200
    body = resp.json()
    assert body["category"] == "inference"
    assert body["elastic_capable"] is True


def test_get_node_unknown_type_404():
    """GET /api/nodes/{unknown} returns 404."""
    resp = client.get("/api/nodes/NotARealNode")
    assert resp.status_code == 404


# ── topics templates ──────────────────────────────────────────────────────────

def test_get_topics_templates():
    """GET /api/topics/templates returns all shipped sample topics."""
    resp = client.get("/api/topics/templates")
    assert resp.status_code == 200
    body = resp.json()
    names = {t["name"] for t in body}
    assert "camera_frame" in names
    assert "yolo_raw" in names

    yolo_raw = next(t for t in body if t["name"] == "yolo_raw")
    assert yolo_raw["elastic"] is True
    assert yolo_raw["max_replicas"] == 4


def test_health_check():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
