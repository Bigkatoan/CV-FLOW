"""Crop BBox node — crops regions from a frame using bounding box coordinates.

Reads bboxes from ctx.metadata["face_bboxes"] (face_detect) or ctx.detections.
Outputs:
  crop_images  — list of N numpy arrays, each image_size × image_size
  crop_count   — int, number of crops produced
  crop_ious    — N×N list-of-lists, pairwise IoU between each pair of bboxes
"""
from __future__ import annotations
import logging

import cv2
import numpy as np

from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext
from engine.core.node_registry import register

logger = logging.getLogger(__name__)


def _pairwise_iou(bboxes: list[tuple]) -> list[list[float]]:
    """Compute N×N pairwise IoU matrix for bboxes = [(x1,y1,x2,y2), ...]."""
    n = len(bboxes)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1.0
        for j in range(i + 1, n):
            ax1, ay1, ax2, ay2 = bboxes[i]
            bx1, by1, bx2, by2 = bboxes[j]
            ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            area_a = (ax2 - ax1) * (ay2 - ay1)
            area_b = (bx2 - bx1) * (by2 - by1)
            union = area_a + area_b - inter
            iou = round(inter / union, 4) if union > 0 else 0.0
            matrix[i][j] = matrix[j][i] = iou
    return matrix


@register("crop_bbox")
class CropBBoxNode(BaseNode):
    """Crop bounding box regions from a frame, resize, and report pairwise IoU."""

    def initialize(self):
        self._size    = int(self.config.get("image_size", 112))
        self._padding = float(self.config.get("padding", 0.0))

    def process(self, ctx: FrameContext) -> FrameContext:
        ctx.metadata["crop_images"] = []
        ctx.metadata["crop_count"]  = 0
        ctx.metadata["crop_ious"]   = []

        # Active gate — idles when face_has_detection is explicitly False
        if ctx.metadata.get("face_has_detection") is False:
            logger.debug("[CropBBox] active=False — idle this frame")
            return ctx

        if ctx.frame is None:
            return ctx

        frame = ctx.ensure_cpu()
        h, w = frame.shape[:2]

        # Prefer face_bboxes [[x1,y1,x2,y2,conf?], ...] from face_detect,
        # fall back to generic ctx.detections (Detection objects)
        raw = ctx.metadata.get("face_bboxes")
        if raw:
            bboxes = [(float(b[0]), float(b[1]), float(b[2]), float(b[3])) for b in raw]
        elif ctx.detections:
            bboxes = [(d.x1, d.y1, d.x2, d.y2) for d in ctx.detections]
        else:
            return ctx

        # Apply padding then clip, keeping track of valid bboxes only
        valid_bboxes: list[tuple] = []
        crops: list[np.ndarray] = []

        for x1, y1, x2, y2 in bboxes:
            if self._padding > 0:
                pw = (x2 - x1) * self._padding
                ph = (y2 - y1) * self._padding
                x1 -= pw; y1 -= ph; x2 += pw; y2 += ph

            xi1 = max(0, int(x1)); yi1 = max(0, int(y1))
            xi2 = min(w, int(x2)); yi2 = min(h, int(y2))

            if xi2 <= xi1 or yi2 <= yi1:
                continue

            crop = frame[yi1:yi2, xi1:xi2]
            if self._size > 0:
                crop = cv2.resize(crop, (self._size, self._size), interpolation=cv2.INTER_LINEAR)
            crops.append(crop)
            valid_bboxes.append((float(xi1), float(yi1), float(xi2), float(yi2)))

        ctx.metadata["crop_images"] = crops
        ctx.metadata["crop_count"]  = len(crops)
        ctx.metadata["crop_ious"]   = _pairwise_iou(valid_bboxes)

        logger.debug("[CropBBox] %d crop(s), IoU matrix %dx%d", len(crops), len(crops), len(crops))
        return ctx
