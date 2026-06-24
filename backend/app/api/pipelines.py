import json
import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.pipeline import Pipeline
from app.schemas.pipeline import (
    PipelineCreate, PipelineResponse, PipelineListItem, ValidateResponse
)

router = APIRouter(prefix="/pipelines", tags=["pipelines"])


def _row_to_response(row: Pipeline) -> PipelineResponse:
    data = json.loads(row.config_json)
    return PipelineResponse(
        id=row.id,
        version=data.get("version", "1.0"),
        name=row.name,
        description=row.description,
        nodes=data.get("nodes", []),
        edges=data.get("edges", []),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


@router.get("", response_model=list[PipelineListItem])
async def list_pipelines(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Pipeline).order_by(Pipeline.updated_at.desc()))
    return [
        PipelineListItem(
            id=r.id, name=r.name, description=r.description,
            created_at=r.created_at, updated_at=r.updated_at,
        )
        for r in result.scalars()
    ]


@router.post("", response_model=PipelineResponse, status_code=201)
async def create_pipeline(body: PipelineCreate, db: AsyncSession = Depends(get_db)):
    pid = body.id or str(uuid.uuid4())
    config_json = body.model_dump_json(exclude={"id"})
    now = datetime.now(timezone.utc)
    row = Pipeline(id=pid, name=body.name, description=body.description,
                   config_json=config_json, created_at=now, updated_at=now)
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return _row_to_response(row)


@router.get("/{pipeline_id}", response_model=PipelineResponse)
async def get_pipeline(pipeline_id: str, db: AsyncSession = Depends(get_db)):
    row = await db.get(Pipeline, pipeline_id)
    if not row:
        raise HTTPException(404, "Pipeline not found")
    return _row_to_response(row)


@router.put("/{pipeline_id}", response_model=PipelineResponse)
async def update_pipeline(pipeline_id: str, body: PipelineCreate, db: AsyncSession = Depends(get_db)):
    row = await db.get(Pipeline, pipeline_id)
    if not row:
        raise HTTPException(404, "Pipeline not found")
    row.name = body.name
    row.description = body.description
    row.config_json = body.model_dump_json()
    row.updated_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(row)
    return _row_to_response(row)


@router.delete("/{pipeline_id}", status_code=204)
async def delete_pipeline(pipeline_id: str, db: AsyncSession = Depends(get_db)):
    row = await db.get(Pipeline, pipeline_id)
    if not row:
        raise HTTPException(404, "Pipeline not found")
    await db.delete(row)
    await db.commit()


@router.post("/validate", response_model=ValidateResponse)
async def validate_pipeline(body: PipelineCreate):
    errors: list[str] = []
    node_ids = {n.get("id") for n in body.nodes}
    for edge in body.edges:
        if edge.get("source") not in node_ids:
            errors.append(f"Edge {edge.get('id')}: source node {edge.get('source')} not found")
        if edge.get("target") not in node_ids:
            errors.append(f"Edge {edge.get('id')}: target node {edge.get('target')} not found")
    return ValidateResponse(valid=len(errors) == 0, errors=errors)
