"""Draw bounding-box detections onto the frame."""
import cv2
import numpy as np
from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext
from engine.core.node_registry import register

_PALETTE = [
    (56, 182, 255), (255, 100, 56),  (56, 255, 100), (255, 220, 56),
    (200, 56, 255),  (56, 230, 230), (255, 56, 150),  (130, 255, 56),
    (255, 160, 0),   (0, 160, 255),
]


@register("draw_bbox")
class DrawBboxNode(BaseNode):
    def initialize(self):
        self._thickness  = int(self.config.get("thickness", 2))
        self._show_label = bool(self.config.get("show_label", True))
        self._show_conf  = bool(self.config.get("show_confidence", True))
        self._show_track = bool(self.config.get("show_track_id", True))
        self._font_scale = float(self.config.get("font_scale", 0.45))

    def process(self, ctx: FrameContext) -> FrameContext:
        if ctx.frame is None:
            return ctx

        frame = ctx.ensure_cpu().copy()
        if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        for det in ctx.detections:
            color = _PALETTE[det.class_id % len(_PALETTE)]
            x1, y1 = int(det.x1), int(det.y1)
            x2, y2 = int(det.x2), int(det.y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self._thickness)

            label_parts = []
            if self._show_track and det.track_id >= 0:
                label_parts.append(f"#{det.track_id}")
            if self._show_label:
                label_parts.append(det.class_name)
            if self._show_conf:
                label_parts.append(f"{det.confidence:.2f}")

            if label_parts:
                label = " ".join(label_parts)
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, self._font_scale, 1
                )
                cv2.rectangle(
                    frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, cv2.FILLED
                )
                cv2.putText(
                    frame, label, (x1 + 3, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, self._font_scale,
                    (255, 255, 255), 1, cv2.LINE_AA,
                )

        ctx.frame = frame
        return ctx
