"""
backend.app.api.nodes — GET /api/nodes, GET /api/nodes/{type}
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from cv_flow.nodes._catalog import NODE_CATALOG

router = APIRouter(prefix="/api/nodes", tags=["nodes"])


@router.get("")
def list_nodes() -> dict:
    """Return the full node catalog: {node_type: metadata}."""
    return NODE_CATALOG


@router.get("/{node_type}")
def get_node(node_type: str) -> dict:
    """Return metadata for a single node type."""
    meta = NODE_CATALOG.get(node_type)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Unknown node type '{node_type}'")
    return meta
