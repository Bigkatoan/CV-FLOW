import cv2
from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext
from engine.core.node_registry import register

_INTERP_MAP = {
    "nearest": cv2.INTER_NEAREST,
    "linear":  cv2.INTER_LINEAR,
    "cubic":   cv2.INTER_CUBIC,
    "area":    cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}


@register("resize")
class ResizeNode(BaseNode):
    def process(self, ctx: FrameContext) -> FrameContext:
        if ctx.frame is None:
            return ctx
        w = int(self.config.get("width", 640))
        h = int(self.config.get("height", 480))
        if w <= 0 or h <= 0:
            return ctx
        interp = _INTERP_MAP.get(self.config.get("interpolation", "area"), cv2.INTER_AREA)
        ctx.frame = cv2.resize(ctx.frame, (w, h), interpolation=interp)
        return ctx
