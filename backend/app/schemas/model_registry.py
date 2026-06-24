from datetime import datetime
from typing import Any
from pydantic import BaseModel


class ModelResponse(BaseModel):
    id: str
    name: str
    version: str
    task: str
    config: dict[str, Any]
    uploaded_at: datetime

    model_config = {"from_attributes": True}


class ModelListItem(BaseModel):
    id: str
    name: str
    version: str
    task: str
    uploaded_at: datetime

    model_config = {"from_attributes": True}
