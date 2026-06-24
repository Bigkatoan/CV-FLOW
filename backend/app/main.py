from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import settings
from app.database import create_tables
from app.api.router import api_router

_STATIC_DIR = Path(__file__).parent.parent.parent / "frontend" / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.compiled_dir.mkdir(parents=True, exist_ok=True)
    settings.pipelines_tmp_dir.mkdir(parents=True, exist_ok=True)
    await create_tables()
    yield


app = FastAPI(
    title="CV-FLOW API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def serve_ui():
    index = _STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "CV-FLOW API is running. Place index.html in frontend/static/"}


if _STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
