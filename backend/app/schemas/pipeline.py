from datetime import datetime
from pydantic import BaseModel, Field
import uuid


class PipelineCreate(BaseModel):
    version: str = "1.0"
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str | None = None
    nodes: list[dict]
    edges: list[dict]


class PipelineUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    nodes: list[dict] | None = None
    edges: list[dict] | None = None


class PipelineResponse(BaseModel):
    id: str
    version: str
    name: str
    description: str | None
    nodes: list[dict]
    edges: list[dict]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class PipelineListItem(BaseModel):
    id: str
    name: str
    description: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ValidateResponse(BaseModel):
    valid: bool
    errors: list[str] = []
