import base64
import cv2
from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext
from engine.core.node_registry import register
from engine.streaming import ws_server


@register("stream_viewer")
class StreamViewerNode(BaseNode):
    def initialize(self):
        self._quality = self.config.get("jpeg_quality", 80)
        self._encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._quality]

    def process(self, ctx: FrameContext) -> FrameContext:
        ok, buf = cv2.imencode(".jpg", ctx.frame, self._encode_params)
        if not ok:
            return ctx

        jpeg_bytes = buf.tobytes()

        # Send to WebSocket stream channel
        ws_server.send_frame(ctx.session_id, jpeg_bytes)

        # Also send as base64 JSON event for frontend fallback
        b64 = base64.b64encode(jpeg_bytes).decode()
        ws_server.send_event(ctx.session_id, {
            "type": "frame",
            "session_id": ctx.session_id,
            "frame_number": ctx.frame_number,
            "timestamp": ctx.timestamp,
            "data": b64,
        })
        return ctx
