from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import create_tables
from app.api.router import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create storage directories
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.compiled_dir.mkdir(parents=True, exist_ok=True)
    settings.pipelines_tmp_dir.mkdir(parents=True, exist_ok=True)
    # Create DB tables
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
