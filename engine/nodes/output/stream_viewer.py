import base64
import numpy as np
import cv2
from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext
from engine.core.node_registry import register
from engine.streaming import ws_server

# Palette for bounding-box colors (class_id % len cycles)
_COLORS = [
    (56, 182, 255), (255, 100, 56), (56, 255, 100), (255, 220, 56),
    (200, 56, 255), (56, 230, 230), (255, 56, 150), (130, 255, 56),
    (255, 160, 0),  (0, 160, 255),
]


def draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    out = frame.copy()
    if out.ndim == 2 or (out.ndim == 3 and out.shape[2] == 1):
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    for det in detections:
        color = _COLORS[det.class_id % len(_COLORS)]
        x1, y1 = int(det.x1), int(det.y1)
        x2, y2 = int(det.x2), int(det.y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        if det.track_id >= 0:
            label = f"#{det.track_id} {label}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        # Dark background for label readability
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, cv2.FILLED)
        cv2.putText(out, label, (x1 + 3, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


@register("stream_viewer")
class StreamViewerNode(BaseNode):
    def initialize(self):
        self._quality = self.config.get("jpeg_quality", 80)
        self._encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._quality]
        # Default False — use draw_bbox node for visualization to avoid double-drawing
        self._draw_dets = bool(self.config.get("draw_detections", False))

    def process(self, ctx: FrameContext) -> FrameContext:
        frame = ctx.ensure_cpu()
        if self._draw_dets and ctx.detections:
            frame = draw_detections(frame, ctx.detections)

        ok, buf = cv2.imencode(".jpg", frame, self._encode_params)
        if not ok:
            return ctx

        jpeg_bytes = buf.tobytes()
        ws_server.send_frame(ctx.session_id, jpeg_bytes)

        # Lightweight frame event for status bar
        ws_server.send_event(ctx.session_id, {
            "type": "frame",
            "session_id": ctx.session_id,
            "frame_number": ctx.frame_number,
            "timestamp": ctx.timestamp,
            "detection_count": len(ctx.detections),
        })

        # Emit one counter_update event per counter node found in metadata
        for key, val in ctx.metadata.items():
            if key.startswith("counter_"):
                node_id = key[len("counter_"):]
                ws_server.send_event(ctx.session_id, {
                    "type": "counter_update",
                    "counter_id": node_id,
                    "value": val,
                })

        return ctx
