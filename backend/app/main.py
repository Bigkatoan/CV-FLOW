"""
backend.app.main — FastAPI app for the CV-FLOW visual pipeline editor.
"""
from __future__ import annotations

from fastapi import FastAPI

from backend.app.api import guide, nodes, pipeline, topics

app = FastAPI(title="CV-FLOW Visual Editor API", version="0.3.0")

app.include_router(nodes.router)
app.include_router(topics.router)
app.include_router(pipeline.router)
app.include_router(guide.router)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}
