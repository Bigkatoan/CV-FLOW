"""Face Alignment node — affine-warps each detected face to 112×112 canonical pose.

Uses insightface's norm_crop utility when available, falls back to a pure OpenCV
implementation. No model required — CPU-only, <1 ms per face.
"""
from __future__ import annotations
import logging
import numpy as np
import cv2

from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext
from engine.core.node_registry import register

logger = logging.getLogger(__name__)

# ArcFace standard 5-point destination template at 112×112
_ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def _norm_crop_manual(img: np.ndarray, landmark: list, size: int = 112) -> np.ndarray:
    """Pure-OpenCV affine crop to ArcFace standard template."""
    src = np.array(landmark, dtype=np.float32)
    # Scale dst to target size
    scale = size / 112.0
    dst = _ARCFACE_DST * scale
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if M is None:
        # fallback: simple crop from bbox
        x1, y1 = int(src[:, 0].min()), int(src[:, 1].min())
        x2, y2 = int(src[:, 0].max()), int(src[:, 1].max())
        crop = img[max(0, y1):y2, max(0, x1):x2]
        return cv2.resize(crop, (size, size)) if crop.size > 0 else np.zeros((size, size, 3), np.uint8)
    return cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR)


@register("face_align")
class FaceAlignNode(BaseNode):
    """Crop and affine-warp each face to a canonical 112×112 pose for embedding."""

    def initialize(self):
        self._size   = int(self.config.get("target_size", 112))
        self._margin = float(self.config.get("margin", 0.0))

        # Try insightface's faster implementation
        try:
            from insightface.utils.face_align import norm_crop
            self._norm_crop = norm_crop
            logger.info("[FaceAlign] Using insightface norm_crop")
        except ImportError:
            self._norm_crop = None
            logger.info("[FaceAlign] Using manual OpenCV norm_crop")

    def process(self, ctx: FrameContext) -> FrameContext:
        if ctx.frame is None:
            return ctx

        frame    = ctx.ensure_cpu()
        dets     = ctx.detections
        lm_list  = ctx.metadata.get("face_landmarks", [])

        aligned = []
        for i, det in enumerate(dets):
            lm = lm_list[i] if i < len(lm_list) else None

            if lm is not None and len(lm) == 5:
                kps = np.array(lm, dtype=np.float32)
                if self._norm_crop is not None:
                    try:
                        crop = self._norm_crop(frame, kps, image_size=self._size)
                        aligned.append(crop)
                        continue
                    except Exception:
                        pass
                aligned.append(_norm_crop_manual(frame, lm, self._size))
            else:
                # No landmarks — simple bbox crop
                h, w = frame.shape[:2]
                m = self._margin
                x1 = max(0, int(det.x1 - (det.x2 - det.x1) * m))
                y1 = max(0, int(det.y1 - (det.y2 - det.y1) * m))
                x2 = min(w, int(det.x2 + (det.x2 - det.x1) * m))
                y2 = min(h, int(det.y2 + (det.y2 - det.y1) * m))
                crop = frame[y1:y2, x1:x2]
                aligned.append(cv2.resize(crop, (self._size, self._size))
                                if crop.size > 0
                                else np.zeros((self._size, self._size, 3), np.uint8))

        ctx.metadata["aligned_faces"] = aligned
        logger.debug("[FaceAlign] aligned %d face(s)", len(aligned))
        return ctx
