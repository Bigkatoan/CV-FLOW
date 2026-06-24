import cv2
from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext
from engine.core.node_registry import register

_CONV_MAP = {
    "bgr2gray": cv2.COLOR_BGR2GRAY,
    "bgr2hsv":  cv2.COLOR_BGR2HSV,
    "bgr2rgb":  cv2.COLOR_BGR2RGB,
    "bgr2lab":  cv2.COLOR_BGR2Lab,
    "bgr2yuv":  cv2.COLOR_BGR2YUV,
    "gray2bgr": cv2.COLOR_GRAY2BGR,
    "hsv2bgr":  cv2.COLOR_HSV2BGR,
}


@register("color_convert")
class ColorConvertNode(BaseNode):
    def process(self, ctx: FrameContext) -> FrameContext:
        if ctx.frame is None:
            return ctx
        code = _CONV_MAP.get(self.config.get("conversion", "bgr2gray"))
        if code is None:
            return ctx
        result = cv2.cvtColor(ctx.frame, code)
        if result.ndim == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        ctx.frame = result
        return ctx
