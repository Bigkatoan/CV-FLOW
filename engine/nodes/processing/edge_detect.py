import cv2
import numpy as np
from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext
from engine.core.node_registry import register


@register("edge_detect")
class EdgeDetectNode(BaseNode):
    def process(self, ctx: FrameContext) -> FrameContext:
        if ctx.frame is None:
            return ctx
        algo = self.config.get("algorithm", "canny")
        frame = ctx.frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if algo == "canny":
            t1 = float(self.config.get("threshold1", 50))
            t2 = float(self.config.get("threshold2", 150))
            edges = cv2.Canny(gray, t1, t2)
        elif algo == "sobel":
            sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges = cv2.convertScaleAbs(np.sqrt(sx**2 + sy**2))
        elif algo == "laplacian":
            edges = cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F))
        else:
            edges = gray
        ctx.frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return ctx
