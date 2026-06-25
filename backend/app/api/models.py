import asyncio
import json
import uuid
import shutil
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.model_registry import ModelEntry
from app.schemas.model_registry import ModelResponse, ModelListItem
from app.config import settings

router = APIRouter(prefix="/models", tags=["models"])

# In-memory registry of session_id → model_ids for hot-reload signaling
# (execution_service populates this)
_session_model_map: dict[str, set[str]] = {}


def _row_to_response(row: ModelEntry) -> ModelResponse:
    return ModelResponse(
        id=row.id,
        name=row.name,
        version=row.version,
        task=row.task,
        config=json.loads(row.config_json),
        uploaded_at=row.uploaded_at,
    )


@router.get("", response_model=list[ModelListItem])
async def list_models(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ModelEntry).order_by(ModelEntry.uploaded_at.desc()))
    return [
        ModelListItem(id=r.id, name=r.name, version=r.version, task=r.task, uploaded_at=r.uploaded_at)
        for r in result.scalars()
    ]


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(model_id: str, db: AsyncSession = Depends(get_db)):
    row = await db.get(ModelEntry, model_id)
    if not row:
        raise HTTPException(404, "Model not found")
    return _row_to_response(row)


@router.post("/upload", response_model=ModelResponse, status_code=201)
async def upload_model(
    model_file: UploadFile = File(..., description=".onnx model file"),
    config_file: UploadFile = File(..., description="config.json metadata file"),
    db: AsyncSession = Depends(get_db),
):
    # Parse and validate config
    try:
        config_data = json.loads(await config_file.read())
    except json.JSONDecodeError:
        raise HTTPException(422, "config.json is not valid JSON")

    required = {"name", "version", "task", "format", "input_shape", "output_shapes"}
    missing = required - config_data.keys()
    if missing:
        raise HTTPException(422, f"config.json missing required fields: {missing}")

    if config_data.get("format") != "onnx":
        raise HTTPException(422, "Only ONNX format is supported")

    model_id = str(uuid.uuid4())
    config_data["model_id"] = model_id

    # Store files
    model_dir = settings.models_dir / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = model_dir / "model.onnx"
    with open(onnx_path, "wb") as f:
        shutil.copyfileobj(model_file.file, f)

    config_path = model_dir / "config.json"
    config_path.write_text(json.dumps(config_data, indent=2))

    # Save to DB
    row = ModelEntry(
        id=model_id,
        name=config_data["name"],
        version=config_data["version"],
        task=config_data["task"],
        file_path=str(onnx_path),
        config_json=json.dumps(config_data),
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return _row_to_response(row)


@router.delete("/{model_id}", status_code=204)
async def delete_model(model_id: str, db: AsyncSession = Depends(get_db)):
    row = await db.get(ModelEntry, model_id)
    if not row:
        raise HTTPException(404, "Model not found")
    model_dir = settings.models_dir / model_id
    if model_dir.exists():
        shutil.rmtree(model_dir)
    await db.delete(row)
    await db.commit()


@router.post("/{model_id}/reload", status_code=202)
async def reload_model(model_id: str, db: AsyncSession = Depends(get_db)):
    """Signal running pipelines using this model to hot-reload."""
    row = await db.get(ModelEntry, model_id)
    if not row:
        raise HTTPException(404, "Model not found")

    import signal, os
    from app.services.execution_service import get_running_sessions

    reloaded = 0
    for session_id, proc in get_running_sessions().items():
        if proc.poll() is None:  # still running
            try:
                os.kill(proc.pid, signal.SIGUSR1)
                reloaded += 1
            except (ProcessLookupError, OSError):
                pass

    return {"reloaded_sessions": reloaded}


@router.get("/{model_id}/download")
async def download_model(model_id: str, db: AsyncSession = Depends(get_db)):
    row = await db.get(ModelEntry, model_id)
    if not row:
        raise HTTPException(404, "Model not found")
    return FileResponse(row.file_path, filename=f"{row.name}_v{row.version}.onnx")


# ── YOLO default model download ───────────────────────────────────────────────

@router.get("/defaults/list")
async def list_default_models():
    """List the full curated model catalog with category, task, and badge info."""
    from engine.model_hub.yolo_downloader import MODEL_CATALOG
    return [
        {
            "key":      k,
            "name":     v["name"],
            "desc":     v["desc"],
            "category": v["category"],
            "task":     v["task"],
            "size_mb":  v["size_mb"],
            "badge":    v.get("badge"),
        }
        for k, v in MODEL_CATALOG.items()
    ]


@router.post("/defaults/download/{model_key}", response_model=ModelResponse, status_code=201)
async def download_default_model(model_key: str, db: AsyncSession = Depends(get_db)):
    """Download a YOLO default model, export to ONNX, and register it."""
    from engine.model_hub.yolo_downloader import download_yolo_model

    try:
        loop = asyncio.get_event_loop()
        config = await loop.run_in_executor(
            None,
            lambda: download_yolo_model(model_key, settings.models_dir),
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(400, str(exc))

    model_id = config["model_id"]
    row = ModelEntry(
        id=model_id,
        name=config["name"],
        version=config["version"],
        task=config["task"],
        file_path=str(settings.models_dir / model_id / "model.onnx"),
        config_json=json.dumps(config),
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return _row_to_response(row)


# ── Face model download ───────────────────────────────────────────────────────

@router.get("/face/list")
async def list_face_models():
    """List the Face ID model catalog (SCRFD, ArcFace, MobileFaceNet)."""
    from engine.model_hub.face_downloader import FACE_MODEL_CATALOG
    return [
        {
            "key":      k,
            "name":     v["name"],
            "desc":     v["desc"],
            "category": v["category"],
            "task":     v["task"],
            "size_mb":  v["size_mb"],
            "badge":    v.get("badge"),
        }
        for k, v in FACE_MODEL_CATALOG.items()
    ]


@router.post("/face/download/{model_key}", response_model=ModelResponse, status_code=201)
async def download_face_model_endpoint(model_key: str, db: AsyncSession = Depends(get_db)):
    """Download a face model via InsightFace and register it in the library."""
    from engine.model_hub.face_downloader import download_face_model

    try:
        loop = asyncio.get_event_loop()
        config = await loop.run_in_executor(
            None,
            lambda: download_face_model(model_key, settings.models_dir),
        )
    except (ImportError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc))

    # If already registered (find_existing returned existing), don't re-insert
    existing = await db.get(ModelEntry, config["model_id"])
    if existing:
        return _row_to_response(existing)

    model_id = config["model_id"]
    face_row = ModelEntry(
        id=model_id,
        name=config["name"],
        version=config.get("version", "1.0"),
        task=config["task"],
        file_path=str(settings.models_dir / model_id / "model.onnx"),
        config_json=json.dumps(config),
    )
    db.add(face_row)
    await db.commit()
    await db.refresh(face_row)
    return _row_to_response(face_row)
