import cv2
from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext
from engine.core.node_registry import register


@register("blur")
class BlurNode(BaseNode):
    def process(self, ctx: FrameContext) -> FrameContext:
        if ctx.frame is None:
            return ctx
        blur_type = self.config.get("type", "gaussian")
        k = int(self.config.get("kernel_size", 5))
        if k % 2 == 0:
            k += 1
        if blur_type == "gaussian":
            sigma = float(self.config.get("sigma", 0))
            ctx.frame = cv2.GaussianBlur(ctx.frame, (k, k), sigma)
        elif blur_type == "box":
            ctx.frame = cv2.blur(ctx.frame, (k, k))
        elif blur_type == "median":
            ctx.frame = cv2.medianBlur(ctx.frame, k)
        return ctx
