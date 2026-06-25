"""Embedding node — extracts face embeddings from cropped images.

Reads ctx.metadata["crop_images"] (output of crop_bbox).
Active gate: ctx.metadata["face_has_detection"] — if explicitly False, idles.
Output: ctx.metadata["face_embeddings"] list of normalised numpy arrays.
"""
from __future__ import annotations
import logging

import numpy as np

from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext
from engine.core.node_registry import register

logger = logging.getLogger(__name__)

_MODEL_NAMES = {
    "mobilefacenet": "w600k_mbf",
    "arcface_r50":   "w600k_r50",
    "arcface_r100":  "w600k_r100",
}


@register("embedding")
class EmbeddingNode(BaseNode):
    """Extract face embeddings from N cropped images and output N embedding vectors."""

    def initialize(self):
        self._model_key = self.config.get("model_key", "mobilefacenet")
        self._normalize = bool(self.config.get("normalize", True))
        self._rec = None
        self._load_model()

    def _load_model(self):
        name = _MODEL_NAMES.get(self._model_key, "w600k_mbf")
        try:
            from insightface import model_zoo
            logger.info("[Embedding] Loading %s …", name)
            self._rec = model_zoo.get_model(name)
            self._rec.prepare(ctx_id=-1)
            logger.info("[Embedding] %s ready", name)
        except ImportError:
            logger.error("[Embedding] insightface not installed. pip install insightface")
        except Exception as e:
            logger.error("[Embedding] Failed to load: %s", e, exc_info=True)

    def process(self, ctx: FrameContext) -> FrameContext:
        ctx.metadata["face_embeddings"] = []

        # Active gate — idles when face_has_detection is explicitly False
        if ctx.metadata.get("face_has_detection") is False:
            logger.debug("[Embedding] active=False — idle")
            return ctx

        crops = ctx.metadata.get("crop_images", [])
        if not crops:
            return ctx

        if self._rec is None:
            logger.warning("[Embedding] No model loaded — skipping.")
            return ctx

        embeddings = []
        for crop in crops:
            try:
                emb = self._rec.get_feat(crop)
                if self._normalize:
                    emb = emb / (np.linalg.norm(emb) + 1e-8)
                embeddings.append(emb.flatten())
            except Exception as e:
                logger.warning("[Embedding] get_feat failed: %s", e)
                embeddings.append(np.zeros(512, dtype=np.float32))

        ctx.metadata["face_embeddings"] = embeddings
        logger.debug("[Embedding] %d embedding(s) extracted", len(embeddings))
        return ctx
