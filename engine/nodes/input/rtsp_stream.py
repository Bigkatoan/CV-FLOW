import time
import cv2
from engine.nodes.input.camera import CameraNode
from engine.core.frame_context import FrameContext
from engine.core.node_registry import register


@register("rtsp_stream")
class RTSPStreamNode(CameraNode):
    def initialize(self):
        self.config["source_type"] = "rtsp"
        self._src = self.config.get("url", "")
        super().initialize()

    def process(self, ctx: FrameContext) -> FrameContext:
        if self._frame_delay > 0:
            now = time.monotonic()
            wait = self._frame_delay - (now - self._last_frame_time)
            if wait > 0:
                time.sleep(wait)
            self._last_frame_time = time.monotonic()

        ok, frame = self._cap.read()
        if not ok:
            delay = self.config.get("reconnect_delay_s", 3.0)
            time.sleep(delay)
            self._cap.release()
            self._cap = cv2.VideoCapture(self._src)
            ok, frame = self._cap.read()
        if not ok:
            raise StopIteration("RTSP stream lost")
        ctx.frame = frame
        ctx.timestamp = time.time()
        return ctx
