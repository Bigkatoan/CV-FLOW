"""
Tests for cv_flow.nodes._catalog.NODE_CATALOG

T-CAT-01: all nodes have required fields (category, description, inputs, outputs, parameters)
T-CAT-02: each parameter has name, type, and either default or required=True
T-CAT-03: elastic_capable=True nodes have at least 1 input
T-CAT-04: GET /api/nodes → returns correct catalog list (API-level, tested separately)
T-CAT-05: GET /api/nodes/{type} → returns correct node metadata (API-level, tested separately)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from cv_flow.nodes._catalog import NODE_CATALOG, validate_catalog


# ── T-CAT-01 ──────────────────────────────────────────────────────────────────

def test_cat_01_required_fields():
    """Every node in the catalog has category, description, inputs, outputs, parameters."""
    required = {"category", "description", "inputs", "outputs", "parameters"}
    for node_type, meta in NODE_CATALOG.items():
        missing = required - meta.keys()
        assert not missing, f"{node_type} missing fields: {missing}"


# ── T-CAT-02 ──────────────────────────────────────────────────────────────────

def test_cat_02_parameter_schema():
    """Every parameter has name + type, and either 'default' or required=True."""
    for node_type, meta in NODE_CATALOG.items():
        for param in meta["parameters"]:
            assert "name" in param, f"{node_type}: parameter missing 'name'"
            assert "type" in param, f"{node_type}.{param.get('name')}: missing 'type'"
            has_default  = "default" in param
            has_required = param.get("required", False)
            assert has_default or has_required, (
                f"{node_type}.{param['name']}: must have 'default' or 'required': True"
            )


# ── T-CAT-03 ──────────────────────────────────────────────────────────────────

def test_cat_03_elastic_capable_has_inputs():
    """Nodes with elastic_capable=True must have at least 1 input."""
    for node_type, meta in NODE_CATALOG.items():
        if meta.get("elastic_capable"):
            assert len(meta["inputs"]) >= 1, (
                f"{node_type}: elastic_capable=True but has no inputs"
            )


# ── validate_catalog() helper ─────────────────────────────────────────────────

def test_validate_catalog_no_errors():
    """validate_catalog() returns no errors for the built-in catalog."""
    errors = validate_catalog()
    assert errors == [], f"Catalog validation errors: {errors}"


def test_catalog_nonempty():
    """Catalog has at least one node per major category."""
    categories = {meta["category"] for meta in NODE_CATALOG.values()}
    assert "input"         in categories
    assert "processing"    in categories
    assert "inference"     in categories
    assert "postprocess"   in categories
    assert "tracking"      in categories
    assert "visualization" in categories
    assert "output"        in categories
