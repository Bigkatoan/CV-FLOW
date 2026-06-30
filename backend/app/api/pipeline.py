"""
backend.app.api.pipeline — Pipeline Spec CRUD + validation.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.pipeline_store import store
from backend.app.schemas import PipelineRecord, PipelineSpec, ValidationResult
from backend.app.validator import validate_pipeline

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


@router.post("/validate")
def validate(spec: PipelineSpec) -> ValidationResult:
    return validate_pipeline(spec)


@router.post("", status_code=201)
def create_pipeline(spec: PipelineSpec) -> PipelineRecord:
    return store.create(spec)


@router.get("/{pipeline_id}")
def get_pipeline(pipeline_id: str) -> PipelineRecord:
    record = store.get(pipeline_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_id}' not found")
    return record


@router.put("/{pipeline_id}")
def update_pipeline(pipeline_id: str, spec: PipelineSpec) -> PipelineRecord:
    record = store.update(pipeline_id, spec)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_id}' not found")
    return record


@router.delete("/{pipeline_id}", status_code=204, response_model=None)
def delete_pipeline(pipeline_id: str) -> None:
    if not store.delete(pipeline_id):
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_id}' not found")


@router.get("")
def list_pipelines() -> list[PipelineRecord]:
    return store.list_all()
