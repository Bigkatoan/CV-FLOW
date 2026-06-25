"""Face Embedding node — extracts 512-dim L2-normalised embeddings from aligned face crops.

Default model: MobileFaceNet (auto-downloaded via insightface, ~4 MB, CPU real-time).
Supports ArcFace-R50 for higher accuracy (GPU recommended).
Also accepts a custom ONNX via model_id.
"""
from __future__ import annotations
import logging
import os
from pathlib import Path

import cv2
import numpy as np

from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext
from engine.core.node_registry import register

logger = logging.getLogger(__name__)

_INSIGHTFACE_NAMES = {
    "mobilefacenet": "w600k_mbf",
    "arcface_r50":   "w600k_r50",
    "arcface_r100":  "w600k_r100",
}


@register("face_embed")
class FaceEmbedNode(BaseNode):
    """Extract face embeddings from aligned 112×112 crops using ArcFace/MobileFaceNet."""

    DEFAULT_MODEL_KEY = "mobilefacenet"

    def initialize(self):
        self._normalize  = bool(self.config.get("normalize", True))
        self._model_key  = self.config.get("model_key", self.DEFAULT_MODEL_KEY)
        model_id         = self.config.get("model_id", "")

        self._rec = None
        self._session = None

        if model_id:
            self._load_custom_onnx(model_id)
        else:
            self._load_insightface()

    # ── Loaders ───────────────────────────────────────────────────────────────

    def _load_insightface(self):
        insightface_name = _INSIGHTFACE_NAMES.get(self._model_key, "w600k_mbf")
        try:
            from insightface.model_zoo import model_zoo
            logger.info("[FaceEmbed] Loading insightface model %s …", insightface_name)
            self._rec = model_zoo.get_model(insightface_name)
            self._rec.prepare(ctx_id=-1)
            logger.info("[FaceEmbed] %s ready", insightface_name)
        except ImportError:
            logger.error("[FaceEmbed] insightface not installed. pip install insightface")
        except Exception as e:
            logger.error("[FaceEmbed] Failed to load: %s", e, exc_info=True)

    def _load_custom_onnx(self, model_id: str):
        import onnxruntime as ort
        models_dir = Path(os.environ.get("CVFLOW_MODELS_DIR", "storage/models"))
        model_path = models_dir / model_id / "model.onnx"
        if not model_path.exists():
            logger.error("[FaceEmbed] model.onnx not found at %s", model_path)
            return
        providers = self._ort_providers()
        self._session = ort.InferenceSession(str(model_path), providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        logger.info("[FaceEmbed] custom ONNX loaded from %s", model_path)

    def _ort_providers(self):
        device = self.config.get("device", "cpu")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" \
               else ["CPUExecutionProvider"]

    # ── Process ───────────────────────────────────────────────────────────────

    def process(self, ctx: FrameContext) -> FrameContext:
        aligned = ctx.metadata.get("aligned_faces", [])
        if not aligned:
            ctx.metadata["face_embeddings"] = []
            return ctx

        if self._rec is not None:
            embeddings = self._embed_insightface(aligned)
        elif self._session is not None:
            embeddings = self._embed_onnx(aligned)
        else:
            logger.warning("[FaceEmbed] No model loaded — skipping.")
            ctx.metadata["face_embeddings"] = []
            return ctx

        ctx.metadata["face_embeddings"] = embeddings
        logger.debug("[FaceEmbed] %d embedding(s) extracted", len(embeddings))
        return ctx

    def _embed_insightface(self, crops: list) -> list[np.ndarray]:
        embeddings = []
        for crop in crops:
            try:
                emb = self._rec.get_feat(crop)
                if self._normalize:
                    emb = emb / (np.linalg.norm(emb) + 1e-8)
                embeddings.append(emb.flatten())
            except Exception as e:
                logger.warning("[FaceEmbed] get_feat failed: %s", e)
                embeddings.append(np.zeros(512, dtype=np.float32))
        return embeddings

    def _embed_onnx(self, crops: list) -> list[np.ndarray]:
        """Generic ArcFace ONNX: input [N, 3, 112, 112] normalised to [-1, 1]."""
        embeddings = []
        for crop in crops:
            try:
                img = cv2.resize(crop, (112, 112)).astype(np.float32)
                img = (img / 127.5) - 1.0                    # [-1, 1]
                inp = img.transpose(2, 0, 1)[np.newaxis]     # [1,3,112,112]
                out = self._session.run(None, {self._input_name: inp})[0][0]
                if self._normalize:
                    out = out / (np.linalg.norm(out) + 1e-8)
                embeddings.append(out.flatten())
            except Exception as e:
                logger.warning("[FaceEmbed] ONNX infer failed: %s", e)
                embeddings.append(np.zeros(512, dtype=np.float32))
        return embeddings
