import cv2
import numpy as np
from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext
from engine.core.node_registry import register


@register("draw_line")
class DrawLineNode(BaseNode):
    def initialize(self):
        self._line_id = self.config.get("line_id", "line_1")
        self._color = tuple(self.config.get("color", [0, 0, 255]))
        self._direction = self.config.get("direction", "both")
        # [[x0_pct, y0_pct], [x1_pct, y1_pct]]
        line = self.config.get("line", [[10, 50], [90, 50]])
        self._p0_pct = np.array(line[0], dtype=np.float32)
        self._p1_pct = np.array(line[1], dtype=np.float32)

    def _to_px(self, pt_pct: np.ndarray, fw: int, fh: int) -> tuple[int, int]:
        return int(pt_pct[0] * fw / 100), int(pt_pct[1] * fh / 100)

    @staticmethod
    def _cross_product_sign(p0, p1, pt) -> float:
        """Sign of (p1-p0) × (pt-p0). Positive = left side, negative = right side."""
        return (p1[0] - p0[0]) * (pt[1] - p0[1]) - (p1[1] - p0[1]) * (pt[0] - p0[0])

    def process(self, ctx: FrameContext) -> FrameContext:
        fh, fw = ctx.frame.shape[:2]
        p0 = self._to_px(self._p0_pct, fw, fh)
        p1 = self._to_px(self._p1_pct, fw, fh)

        cv2.line(ctx.frame, p0, p1, self._color, 2)
        # Draw arrow at midpoint to show direction
        mid = ((p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2)
        cv2.circle(ctx.frame, mid, 5, self._color, -1)

        # Store line geometry in metadata for CounterNode
        ctx.metadata[f"line_{self._line_id}"] = {
            "p0": p0, "p1": p1, "direction": self._direction
        }
        return ctx
