"""
backend.app.api.guide — POST /api/pipeline/generate-guide
"""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from backend.app.guide_generator import generate_deployment_guide
from backend.app.schemas import PipelineSpec

router = APIRouter(prefix="/api/pipeline", tags=["guide"])


class GuideResponse(BaseModel):
    markdown: str


@router.post("/generate-guide")
def generate_guide(spec: PipelineSpec) -> GuideResponse:
    return GuideResponse(markdown=generate_deployment_guide(spec))
