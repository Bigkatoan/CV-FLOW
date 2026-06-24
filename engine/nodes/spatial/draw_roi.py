import cv2
import numpy as np
from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext
from engine.core.node_registry import register


@register("draw_roi")
class DrawROINode(BaseNode):
    def initialize(self):
        self._zone_id = self.config.get("zone_id", "zone_1")
        self._color = tuple(self.config.get("color", [0, 255, 0]))
        self._draw = self.config.get("draw_on_frame", True)
        self._filter = self.config.get("filter_outside", True)
        # Polygon stored as [[x_pct, y_pct], ...] (percentages 0-100)
        self._polygon_pct = np.array(self.config.get("polygon", []), dtype=np.float32)

    def _get_polygon_px(self, frame_w: int, frame_h: int) -> np.ndarray:
        pts = self._polygon_pct.copy()
        pts[:, 0] *= frame_w / 100.0
        pts[:, 1] *= frame_h / 100.0
        return pts.astype(np.int32)

    def process(self, ctx: FrameContext) -> FrameContext:
        if len(self._polygon_pct) < 3:
            return ctx

        fh, fw = ctx.frame.shape[:2]
        poly = self._get_polygon_px(fw, fh)

        if self._draw:
            overlay = ctx.frame.copy()
            cv2.fillPoly(overlay, [poly], (*self._color, 50))
            cv2.addWeighted(overlay, 0.3, ctx.frame, 0.7, 0, ctx.frame)
            cv2.polylines(ctx.frame, [poly], isClosed=True, color=self._color, thickness=2)

        if self._filter and ctx.detections:
            kept = []
            for det in ctx.detections:
                cx, cy = det.center
                inside = cv2.pointPolygonTest(poly, (cx, cy), measureDist=False) >= 0
                if inside:
                    kept.append(det)
            ctx.detections = kept
            ctx.metadata[f"zone_{self._zone_id}_count"] = len(kept)

        return ctx
