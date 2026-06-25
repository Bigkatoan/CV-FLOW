from fastapi import APIRouter
from app.api import pipelines, models, execution, compiler, system, facedb, python_lsp

api_router = APIRouter(prefix="/api")
api_router.include_router(pipelines.router)
api_router.include_router(models.router)
api_router.include_router(execution.router)
api_router.include_router(compiler.router)
api_router.include_router(system.router)
api_router.include_router(facedb.router)
api_router.include_router(python_lsp.router)
