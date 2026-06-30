"""
Session-wide pytest configuration.

Points the backend's PipelineStore at a temp SQLite file for the whole test
session (instead of the real ~/.cv_flow/pipelines.db) — must be set before
backend.app.pipeline_store is first imported, hence pytest_configure() and
not a fixture.
"""
from __future__ import annotations

import os
import tempfile


def pytest_configure(config) -> None:
    if "CV_FLOW_PIPELINE_DB_PATH" not in os.environ:
        tmp_dir = tempfile.mkdtemp(prefix="cv_flow_test_db_")
        os.environ["CV_FLOW_PIPELINE_DB_PATH"] = os.path.join(tmp_dir, "test_pipelines.db")
