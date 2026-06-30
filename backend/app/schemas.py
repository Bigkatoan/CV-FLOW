"""
backend.app.schemas — Pydantic models for the Pipeline Spec JSON.

Mirrors the structure described in the visual editor design:
a pipeline is a list of topics (DAM channels) plus a list of nodes
(catalog node instances) wired together via named connections.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class TopicFieldSpec(BaseModel):
    name:  str
    dtype: str
    shape: list[int] = Field(default_factory=list)


class TopicSpec(BaseModel):
    name: str
    # Single-port (source/sink) topics use `device` + `fields`.
    device: Optional[str] = None
    fields: list[TopicFieldSpec] = Field(default_factory=list)
    # Transform topics use distinct in/out ports instead.
    input_device:  Optional[str] = None
    output_device: Optional[str] = None
    fields_in:  list[TopicFieldSpec] = Field(default_factory=list)
    fields_out: list[TopicFieldSpec] = Field(default_factory=list)
    elastic:      bool = False
    max_replicas: int  = 4
    queue_depth:  int  = 8
    drop_mode:    bool = False


class NodeConnection(BaseModel):
    slot:  str
    topic: str


class NodePosition(BaseModel):
    x: float = 0.0
    y: float = 0.0


class NodeSpec(BaseModel):
    id:   str
    type: str
    position: NodePosition = Field(default_factory=NodePosition)
    config: dict[str, Any] = Field(default_factory=dict)
    connections_in:  list[NodeConnection] = Field(default_factory=list)
    connections_out: list[NodeConnection] = Field(default_factory=list)


class PipelineSpec(BaseModel):
    version:     str = "1.0"
    name:        str
    description: str = ""
    topics: list[TopicSpec] = Field(default_factory=list)
    nodes:  list[NodeSpec]  = Field(default_factory=list)


class PipelineRecord(PipelineSpec):
    id: str


class ValidationIssue(BaseModel):
    level:   str  # "error" | "warning"
    message: str
    node_id: Optional[str] = None


class ValidationResult(BaseModel):
    valid:  bool
    issues: list[ValidationIssue] = Field(default_factory=list)
