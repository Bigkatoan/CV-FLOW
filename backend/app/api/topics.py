"""
backend.app.api.topics — GET /api/topics/templates
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from cv_flow.topic.parser import load_topics_dir

router = APIRouter(prefix="/api/topics", tags=["topics"])

_TEMPLATES_DIR = Path(__file__).resolve().parents[3] / "cv_flow" / "topic_templates"


@router.get("/templates")
def list_templates() -> list[dict]:
    """Return all sample .topic templates shipped with cv_flow."""
    topics = load_topics_dir(_TEMPLATES_DIR)
    result = []
    for name, td in sorted(topics.items()):
        result.append({
            "name": name,
            "elastic": td.elastic,
            "max_replicas": td.max_replicas,
            "queue_depth": td.queue_depth,
            "drop_mode": td.drop_mode,
            "input_fields": (
                [] if td.input_port.is_none else
                [{"name": f.name, "dtype": f.dtype_str, "shape": list(f.base_shape)}
                 for f in td.input_port.fields]
            ),
            "output_fields": (
                [] if td.output_port.is_none else
                [{"name": f.name, "dtype": f.dtype_str, "shape": list(f.base_shape)}
                 for f in td.output_port.fields]
            ),
        })
    return result
