import cv2
from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext
from engine.core.node_registry import register


@register("threshold")
class ThresholdNode(BaseNode):
    def process(self, ctx: FrameContext) -> FrameContext:
        if ctx.frame is None:
            return ctx
        thresh_type = self.config.get("type", "binary")
        threshold = int(self.config.get("threshold", 127))
        max_val = int(self.config.get("max_val", 255))
        frame = ctx.frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if thresh_type == "binary":
            _, result = cv2.threshold(gray, threshold, max_val, cv2.THRESH_BINARY)
        elif thresh_type == "binary_inv":
            _, result = cv2.threshold(gray, threshold, max_val, cv2.THRESH_BINARY_INV)
        elif thresh_type == "otsu":
            _, result = cv2.threshold(gray, 0, max_val, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif thresh_type == "adaptive":
            result = cv2.adaptiveThreshold(
                gray, max_val, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
        else:
            result = gray
        ctx.frame = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        return ctx
