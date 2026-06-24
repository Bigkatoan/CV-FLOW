import cv2
import numpy as np
from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext
from engine.core.node_registry import register

_OP_MAP = {
    "erode":    cv2.MORPH_ERODE,
    "dilate":   cv2.MORPH_DILATE,
    "open":     cv2.MORPH_OPEN,
    "close":    cv2.MORPH_CLOSE,
    "gradient": cv2.MORPH_GRADIENT,
    "tophat":   cv2.MORPH_TOPHAT,
    "blackhat": cv2.MORPH_BLACKHAT,
}


@register("morph")
class MorphNode(BaseNode):
    def process(self, ctx: FrameContext) -> FrameContext:
        if ctx.frame is None:
            return ctx
        op_code = _OP_MAP.get(self.config.get("operation", "erode"), cv2.MORPH_ERODE)
        k = int(self.config.get("kernel_size", 3))
        iters = int(self.config.get("iterations", 1))
        kernel = np.ones((k, k), np.uint8)
        ctx.frame = cv2.morphologyEx(ctx.frame, op_code, kernel, iterations=iters)
        return ctx
