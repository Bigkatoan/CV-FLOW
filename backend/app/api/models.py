import asyncio
import json
import uuid
import shutil
import tempfile
from pathlib import Path
from typing import Optional
from pydantic import BaseModel

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from sqlalchemy import select, or_
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
    # Safely load config
    try:
        config = json.loads(row.config_json) if row.config_json else {}
    except Exception:
        config = {}

    return ModelResponse(
        id=row.id,
        name=row.name,
        version=row.version,
        task=row.task,
        config=config,
        uploaded_at=row.uploaded_at,
        slug=row.slug,
        tag=row.tag,
        is_latest=row.is_latest,
        parent_id=row.parent_id,
        changelog=row.changelog,
        description=row.description,
        ports_json=row.ports_json,
        last_used_at=row.last_used_at,
        size_bytes=row.size_bytes,
        author=row.author,
        license=row.license,
        extra_meta=row.extra_meta,
    )


def _row_to_list_item(row: ModelEntry) -> ModelListItem:
    return ModelListItem(
        id=row.id,
        name=row.name,
        version=row.version,
        task=row.task,
        uploaded_at=row.uploaded_at,
        slug=row.slug,
        tag=row.tag,
        size_bytes=row.size_bytes,
        is_latest=row.is_latest,
    )


@router.get("", response_model=list[ModelListItem])
async def list_models(
    task: Optional[str] = None,
    q: Optional[str] = None,
    tag: Optional[str] = None,
    sort: str = "uploaded_at",  # "name" | "uploaded_at" | "last_used_at" | "size"
    include_deprecated: bool = False,
    db: AsyncSession = Depends(get_db),
):
    stmt = select(ModelEntry)
    if task:
        stmt = stmt.where(ModelEntry.task == task)
    if q:
        stmt = stmt.where(
            or_(
                ModelEntry.name.ilike(f"%{q}%"),
                ModelEntry.description.ilike(f"%{q}%")
            )
        )
    if tag:
        stmt = stmt.where(ModelEntry.tag == tag)
    if not include_deprecated:
        stmt = stmt.where(ModelEntry.tag != "deprecated")

    # sort
    sort_col = {
        "name": ModelEntry.name,
        "uploaded_at": ModelEntry.uploaded_at,
        "last_used_at": ModelEntry.last_used_at,
        "size": ModelEntry.size_bytes,
    }.get(sort, ModelEntry.uploaded_at)

    stmt = stmt.order_by(sort_col.desc())

    result = await db.execute(stmt)
    return [_row_to_list_item(r) for r in result.scalars()]


@router.get("/catalog")
async def list_catalog_models_early():
    """Alias — resolved before /{model_id} to avoid route shadowing. See list_catalog_models."""
    return await list_catalog_models()


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(model_id: str, db: AsyncSession = Depends(get_db)):
    row = await db.get(ModelEntry, model_id)
    if not row:
        raise HTTPException(404, "Model not found")
    return _row_to_response(row)


@router.get("/versions/{slug}", response_model=list[ModelListItem])
async def list_model_versions(slug: str, db: AsyncSession = Depends(get_db)):
    # slug pattern: {name-lower}-{version}
    # To find family, we match the prefix without the version
    # It's better to find any model with this slug, get its name, and query by name
    # Alternatively query by slug directly if they share the exact base name
    # Let's query by name
    row = await db.execute(select(ModelEntry).where(ModelEntry.slug == slug).limit(1))
    model = row.scalar_one_or_none()
    if not model:
        raise HTTPException(404, "Model not found")
    
    stmt = select(ModelEntry).where(ModelEntry.name == model.name).order_by(ModelEntry.uploaded_at.desc())
    result = await db.execute(stmt)
    return [_row_to_list_item(r) for r in result.scalars()]


@router.post("/inspect")
async def inspect_model(
    model_file: UploadFile = File(..., description=".onnx model file"),
):
    """Introspect an ONNX file and return auto-detected port definitions (without saving)."""
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        shutil.copyfileobj(model_file.file, tmp)
        tmp_path = tmp.name
    try:
        from engine.model_hub.onnx_inspector import inspect_onnx
        return inspect_onnx(tmp_path)
    except Exception as e:
        raise HTTPException(400, f"ONNX inspection failed: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.post("/upload", response_model=ModelResponse, status_code=201)
async def upload_model(
    model_file: UploadFile = File(...),
    config_file: UploadFile = File(...),
    ports_json: str = Form(None),  # JSON string, optional (user can skip)
    db: AsyncSession = Depends(get_db),
):
    try:
        config_data = json.loads(await config_file.read())
    except json.JSONDecodeError:
        raise HTTPException(422, "config.json is not valid JSON")

    # Validate required fields
    required = {"name", "version", "task", "format", "input_shape", "output_shapes"}
    missing = required - config_data.keys()
    if missing:
        raise HTTPException(422, f"config.json missing: {missing}")
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

    # Auto-introspect nếu user không cung cấp ports_json
    if not ports_json:
        try:
            from engine.model_hub.onnx_inspector import inspect_onnx
            ports = inspect_onnx(str(onnx_path))
            ports_json_str = json.dumps(ports)
        except Exception:
            ports_json_str = None
    else:
        ports_json_str = ports_json

    size_bytes = onnx_path.stat().st_size if onnx_path.exists() else None
    slug = f"{config_data['name'].lower().replace(' ', '-')}-{config_data['version']}"

    row = ModelEntry(
        id=model_id,
        name=config_data["name"],
        version=config_data["version"],
        task=config_data["task"],
        file_path=str(onnx_path),
        config_json=json.dumps(config_data),
        ports_json=ports_json_str,
        size_bytes=size_bytes,
        slug=slug,
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


@router.put("/{model_id}/ports", response_model=ModelResponse)
async def update_model_ports(model_id: str, ports: dict, db: AsyncSession = Depends(get_db)):
    row = await db.get(ModelEntry, model_id)
    if not row:
        raise HTTPException(404, "Model not found")
    row.ports_json = json.dumps(ports)
    await db.commit()
    await db.refresh(row)
    return _row_to_response(row)


class TagUpdate(BaseModel):
    tag: str

@router.put("/{model_id}/tag", response_model=ModelResponse)
async def update_model_tag(model_id: str, tag_update: TagUpdate, db: AsyncSession = Depends(get_db)):
    tag = tag_update.tag
    if tag not in ("stable", "experimental", "deprecated"):
        raise HTTPException(400, "Tag must be stable, experimental, or deprecated")
    row = await db.get(ModelEntry, model_id)
    if not row:
        raise HTTPException(404, "Model not found")
    row.tag = tag
    await db.commit()
    await db.refresh(row)
    return _row_to_response(row)


@router.post("/{model_id}/fork", response_model=ModelResponse, status_code=201)
async def fork_model(model_id: str, new_version: str = Form(...), db: AsyncSession = Depends(get_db)):
    parent_row = await db.get(ModelEntry, model_id)
    if not parent_row:
        raise HTTPException(404, "Parent model not found")
    
    new_id = str(uuid.uuid4())
    # Copy files
    old_dir = settings.models_dir / model_id
    new_dir = settings.models_dir / new_id
    shutil.copytree(str(old_dir), str(new_dir))
    
    # Update config.json
    config_path = new_dir / "config.json"
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text())
            cfg["version"] = new_version
            cfg["model_id"] = new_id
            config_path.write_text(json.dumps(cfg, indent=2))
        except Exception:
            pass

    slug = f"{parent_row.name.lower().replace(' ', '-')}-{new_version}"
    
    new_row = ModelEntry(
        id=new_id,
        name=parent_row.name,
        version=new_version,
        task=parent_row.task,
        file_path=str(new_dir / "model.onnx"),
        config_json=json.dumps(cfg) if 'cfg' in locals() else parent_row.config_json,
        ports_json=parent_row.ports_json,
        size_bytes=parent_row.size_bytes,
        slug=slug,
        parent_id=model_id,
        tag="experimental",
        is_latest=True,
    )
    
    # Update parent is_latest
    parent_row.is_latest = False
    
    db.add(new_row)
    await db.commit()
    await db.refresh(new_row)
    return _row_to_response(new_row)


# ── Catalog (Unified YOLO + Face) ─────────────────────────────────────────────

async def list_catalog_models():
    """List the full curated model catalog with category, task, and badge info."""
    catalog = []
    
    try:
        from engine.model_hub.yolo_downloader import MODEL_CATALOG
        for k, v in MODEL_CATALOG.items():
            catalog.append({
                "key":      k,
                "name":     v["name"],
                "desc":     v["desc"],
                "category": v["category"],
                "task":     v["task"],
                "size_mb":  v["size_mb"],
                "badge":    v.get("badge"),
                "source":   "yolo",
            })
    except ImportError:
        pass

    try:
        from engine.model_hub.face_downloader import FACE_MODEL_CATALOG
        for k, v in FACE_MODEL_CATALOG.items():
            catalog.append({
                "key":      k,
                "name":     v["name"],
                "desc":     v["desc"],
                "category": v["category"],
                "task":     v["task"],
                "size_mb":  v["size_mb"],
                "badge":    v.get("badge"),
                "source":   "face",
            })
    except ImportError:
        pass
        
    return catalog


@router.post("/catalog/{model_key}/download", response_model=ModelResponse, status_code=201)
async def download_catalog_model(model_key: str, db: AsyncSession = Depends(get_db)):
    """Download a model from catalog, export to ONNX, introspect ports, and register it."""
    
    # Try YOLO
    try:
        from engine.model_hub.yolo_downloader import MODEL_CATALOG, download_yolo_model
        if model_key in MODEL_CATALOG:
            loop = asyncio.get_event_loop()
            config = await loop.run_in_executor(
                None,
                lambda: download_yolo_model(model_key, settings.models_dir),
            )
            return await _register_downloaded_model(config, db)
    except Exception as e:
        pass # Not a YOLO model or failed
        
    # Try Face
    try:
        from engine.model_hub.face_downloader import FACE_MODEL_CATALOG, download_face_model
        if model_key in FACE_MODEL_CATALOG:
            loop = asyncio.get_event_loop()
            config = await loop.run_in_executor(
                None,
                lambda: download_face_model(model_key, settings.models_dir),
            )
            # If already registered
            existing = await db.get(ModelEntry, config["model_id"])
            if existing:
                return _row_to_response(existing)
                
            return await _register_downloaded_model(config, db)
    except Exception as e:
        raise HTTPException(400, f"Download failed: {e}")
        
    raise HTTPException(404, "Model key not found in catalog")


async def _register_downloaded_model(config: dict, db: AsyncSession) -> ModelResponse:
    model_id = config["model_id"]
    onnx_path = settings.models_dir / model_id / "model.onnx"
    
    ports_json_str = None
    try:
        from engine.model_hub.onnx_inspector import inspect_onnx
        ports = inspect_onnx(str(onnx_path))
        ports_json_str = json.dumps(ports)
    except Exception:
        pass

    size_bytes = onnx_path.stat().st_size if onnx_path.exists() else None
    slug = f"{config['name'].lower().replace(' ', '-')}-{config.get('version', '1.0')}"
    
    row = ModelEntry(
        id=model_id,
        name=config["name"],
        version=config.get("version", "1.0"),
        task=config["task"],
        file_path=str(onnx_path),
        config_json=json.dumps(config),
        ports_json=ports_json_str,
        size_bytes=size_bytes,
        slug=slug,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return _row_to_response(row)
