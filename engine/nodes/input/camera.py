import time
import cv2
from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext
from engine.core.node_registry import register


@register("camera")
class CameraNode(BaseNode):
    _cap: cv2.VideoCapture

    def initialize(self):
        source_type = self.config.get("source_type", "usb")
        if source_type == "rtsp":
            src = self.config.get("url", "")
        else:
            src = int(self.config.get("device_index", 0))
        self._cap = cv2.VideoCapture(src)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {src!r}")
        fps = self.config.get("fps_limit", 0)
        self._frame_delay = 1.0 / fps if fps > 0 else 0.0
        self._last_frame_time = 0.0

    def process(self, ctx: FrameContext) -> FrameContext:
        # FPS limiting
        if self._frame_delay > 0:
            now = time.monotonic()
            wait = self._frame_delay - (now - self._last_frame_time)
            if wait > 0:
                time.sleep(wait)
            self._last_frame_time = time.monotonic()

        ok, frame = self._cap.read()
        if not ok:
            # Reconnect once before giving up
            self._cap.release()
            delay = self.config.get("reconnect_delay_s", 3.0)
            time.sleep(delay)
            self._cap.open(self._cap.get(cv2.CAP_PROP_POS_AVI_RATIO))
            ok, frame = self._cap.read()
        if not ok:
            raise StopIteration("Camera source lost")
        ctx.frame = frame
        ctx.timestamp = time.time()
        return ctx

    def teardown(self):
        if hasattr(self, "_cap"):
            self._cap.release()
