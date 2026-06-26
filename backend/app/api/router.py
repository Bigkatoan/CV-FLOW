from fastapi import APIRouter
from app.api import pipelines, models, execution, compiler, system, facedb, python_lsp, datahub, logging_db

api_router = APIRouter(prefix="/api")
api_router.include_router(pipelines.router)
api_router.include_router(models.router)
api_router.include_router(execution.router)
api_router.include_router(compiler.router)
api_router.include_router(system.router)
api_router.include_router(facedb.router)
api_router.include_router(python_lsp.router)
api_router.include_router(datahub.router)
api_router.include_router(logging_db.router)
