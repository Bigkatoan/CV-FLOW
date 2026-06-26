from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel


class ModelResponse(BaseModel):
    id: str
    name: str
    version: str
    task: str
    config: dict[str, Any]
    uploaded_at: datetime
    
    slug: Optional[str] = None
    tag: Optional[str] = None
    is_latest: Optional[bool] = None
    parent_id: Optional[str] = None
    changelog: Optional[str] = None
    description: Optional[str] = None
    ports_json: Optional[str] = None
    last_used_at: Optional[datetime] = None
    size_bytes: Optional[int] = None
    author: Optional[str] = None
    license: Optional[str] = None
    extra_meta: Optional[str] = None

    model_config = {"from_attributes": True}


class ModelListItem(BaseModel):
    id: str
    name: str
    version: str
    task: str
    uploaded_at: datetime
    
    slug: Optional[str] = None
    tag: Optional[str] = None
    size_bytes: Optional[int] = None
    is_latest: Optional[bool] = None

    model_config = {"from_attributes": True}

