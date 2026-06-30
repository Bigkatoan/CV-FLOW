"""
backend.app.validator — Pipeline Spec validation.

Checks performed:
  1. Every node's `type` exists in NODE_CATALOG.
  2. Every connection references a topic declared in the spec's `topics` list.
  3. Every parameter marked `required: True` in the catalog is present in
     the node's `config`.
  4. Every catalog input slot for a node is wired via connections_in
     (warning, not error — a node may be used partially during editing).
"""
from __future__ import annotations

from backend.app.schemas import PipelineSpec, ValidationIssue, ValidationResult
from cv_flow.nodes._catalog import NODE_CATALOG


def validate_pipeline(spec: PipelineSpec) -> ValidationResult:
    issues: list[ValidationIssue] = []
    topic_names = {t.name for t in spec.topics}

    for node in spec.nodes:
        meta = NODE_CATALOG.get(node.type)
        if meta is None:
            issues.append(ValidationIssue(
                level="error", node_id=node.id,
                message=f"Unknown node type '{node.type}'",
            ))
            continue

        for conn in node.connections_in + node.connections_out:
            if conn.topic not in topic_names:
                issues.append(ValidationIssue(
                    level="error", node_id=node.id,
                    message=f"Connection references undeclared topic '{conn.topic}'",
                ))

        for param in meta["parameters"]:
            if param.get("required") and param["name"] not in node.config:
                issues.append(ValidationIssue(
                    level="error", node_id=node.id,
                    message=f"Missing required parameter '{param['name']}'",
                ))

        wired_in_slots = {c.slot for c in node.connections_in}
        for input_slot in meta["inputs"]:
            if input_slot["slot"] not in wired_in_slots:
                issues.append(ValidationIssue(
                    level="warning", node_id=node.id,
                    message=f"Input slot '{input_slot['slot']}' is not connected",
                ))

    has_errors = any(i.level == "error" for i in issues)
    return ValidationResult(valid=not has_errors, issues=issues)
