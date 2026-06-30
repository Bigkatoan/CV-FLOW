"""
Tests for backend.app.pipeline_store.PipelineStore — SQLite persistence.

T-STORE-01..05: CRUD behavior (mirrors the old in-memory dict tests).
T-STORE-06: the actual point of this module — a record written by one
            PipelineStore instance is still readable by a *second*,
            independent PipelineStore instance pointed at the same db_path,
            proving data survives a process restart (unlike the old
            in-memory dict store).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.app.pipeline_store import PipelineStore
from backend.app.schemas import PipelineSpec


def _sample_spec(name: str = "Test Pipeline") -> PipelineSpec:
    return PipelineSpec(
        name=name,
        description="A minimal valid pipeline",
        topics=[{
            "name": "camera_frame",
            "device": "cpu",
            "fields": [{"name": "frame", "dtype": "bgr8", "shape": [720, 1280]}],
        }],
        nodes=[{
            "id": "cam_0",
            "type": "CameraSource",
            "config": {"device_index": 0},
            "connections_out": [{"slot": "frame_out", "topic": "camera_frame"}],
        }],
    )


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "pipelines.db"


def test_create_and_get(db_path):
    store = PipelineStore(db_path=db_path)
    record = store.create(_sample_spec())
    assert record.id
    fetched = store.get(record.id)
    assert fetched is not None
    assert fetched.name == "Test Pipeline"


def test_get_unknown_returns_none(db_path):
    store = PipelineStore(db_path=db_path)
    assert store.get("does-not-exist") is None


def test_update_existing(db_path):
    store = PipelineStore(db_path=db_path)
    record = store.create(_sample_spec())
    updated = store.update(record.id, _sample_spec(name="Renamed"))
    assert updated is not None
    assert updated.name == "Renamed"
    assert store.get(record.id).name == "Renamed"


def test_update_unknown_returns_none(db_path):
    store = PipelineStore(db_path=db_path)
    assert store.update("does-not-exist", _sample_spec()) is None


def test_delete_existing_and_unknown(db_path):
    store = PipelineStore(db_path=db_path)
    record = store.create(_sample_spec())
    assert store.delete(record.id) is True
    assert store.get(record.id) is None
    assert store.delete(record.id) is False  # already gone


def test_list_all(db_path):
    store = PipelineStore(db_path=db_path)
    store.create(_sample_spec("A"))
    store.create(_sample_spec("B"))
    names = sorted(r.name for r in store.list_all())
    assert names == ["A", "B"]


def test_clear(db_path):
    store = PipelineStore(db_path=db_path)
    store.create(_sample_spec())
    store.create(_sample_spec())
    store.clear()
    assert store.list_all() == []


# ── T-STORE-06: the actual point of this module ────────────────────────────────

def test_data_survives_a_simulated_process_restart(db_path):
    """A record written by one PipelineStore instance is readable by a brand
    new instance pointed at the same db_path — i.e. NOT lost on restart,
    unlike the old in-memory dict implementation."""
    store_before_restart = PipelineStore(db_path=db_path)
    record = store_before_restart.create(_sample_spec(name="Survives Restart"))
    del store_before_restart  # simulate the process exiting

    store_after_restart = PipelineStore(db_path=db_path)
    fetched = store_after_restart.get(record.id)
    assert fetched is not None
    assert fetched.name == "Survives Restart"
    assert len(store_after_restart.list_all()) == 1
