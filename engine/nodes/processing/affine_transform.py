import cv2
from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext
from engine.core.node_registry import register


@register("affine_transform")
class AffineTransformNode(BaseNode):
    def process(self, ctx: FrameContext) -> FrameContext:
        if ctx.frame is None:
            return ctx
        h, w = ctx.frame.shape[:2]
        tx    = float(self.config.get("tx", 0))
        ty    = float(self.config.get("ty", 0))
        angle = float(self.config.get("angle", 0))
        scale = float(self.config.get("scale", 1.0))
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        ctx.frame = cv2.warpAffine(ctx.frame, M, (w, h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
        return ctx
