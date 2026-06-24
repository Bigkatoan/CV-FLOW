from fastapi import APIRouter
from app.api import pipelines, models, execution, compiler

api_router = APIRouter(prefix="/api")
api_router.include_router(pipelines.router)
api_router.include_router(models.router)
api_router.include_router(execution.router)
api_router.include_router(compiler.router)
