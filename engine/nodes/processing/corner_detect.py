import cv2
import numpy as np
from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext
from engine.core.node_registry import register


@register("corner_detect")
class CornerDetectNode(BaseNode):
    def process(self, ctx: FrameContext) -> FrameContext:
        if ctx.frame is None:
            return ctx
        algo = self.config.get("algorithm", "harris")
        frame = ctx.frame.copy()
        gray_uint8 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        if algo == "harris":
            block = int(self.config.get("block_size", 2))
            k = float(self.config.get("k", 0.04))
            quality = float(self.config.get("quality", 0.01))
            dst = cv2.cornerHarris(gray_uint8.astype(np.float32), block, 3, k)
            frame[dst > dst.max() * quality] = [0, 0, 255]
        elif algo == "fast":
            detector = cv2.FastFeatureDetector_create(
                threshold=int(self.config.get("quality", 10) * 1000),
                nonmaxSuppression=True,
            )
            kps = detector.detect(gray_uint8, None)
            max_c = int(self.config.get("max_corners", 100))
            frame = cv2.drawKeypoints(frame, kps[:max_c], None, color=(0, 0, 255))
        elif algo == "shitomasi":
            max_c = int(self.config.get("max_corners", 100))
            quality = float(self.config.get("quality", 0.01))
            min_d = float(self.config.get("min_dist", 10.0))
            corners = cv2.goodFeaturesToTrack(gray_uint8, max_c, quality, min_d)
            if corners is not None:
                for c in corners.astype(int):
                    cv2.circle(frame, tuple(c[0]), 3, (0, 0, 255), -1)

        ctx.frame = frame
        return ctx
