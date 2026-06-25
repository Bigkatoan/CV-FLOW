import sys
import time
import logging
import cv2
from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext
from engine.core.node_registry import register

logger = logging.getLogger(__name__)


@register("camera")
class CameraNode(BaseNode):
    _cap: cv2.VideoCapture
    _src: int | str  # original source, stored for reconnect

    def initialize(self):
        source_type = self.config.get("source_type", "usb")
        if source_type in ("rtsp", "http"):
            self._src = self.config.get("url", "")
        else:
            self._src = int(self.config.get("device_index", 0))
        self._open_cap()

    def _open_cap(self):
        # Primary attempt
        self._cap = cv2.VideoCapture(self._src)

        # On Windows, the default MSMF backend can silently fail for some cameras.
        # Fall back to DirectShow which is more broadly compatible.
        if not self._cap.isOpened() and sys.platform == "win32" and isinstance(self._src, int):
            logger.warning("Camera %s not opened with default backend — retrying with CAP_DSHOW", self._src)
            self._cap.release()
            self._cap = cv2.VideoCapture(self._src, cv2.CAP_DSHOW)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera source: {self._src!r}. "
                f"Check that the camera is plugged in and not used by another app."
            )

        fps = self.config.get("fps_limit", 0)
        self._frame_delay = 1.0 / fps if fps > 0 else 0.0
        self._last_frame_time = 0.0

        # Discard first 3 frames — many USB cameras output black/green
        # frames until the sensor initialises (especially on Windows).
        for _ in range(3):
            self._cap.read()
        logger.info("Camera opened: src=%r  backend=%s", self._src,
                    self._cap.getBackendName() if hasattr(self._cap, "getBackendName") else "?")

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
            self._cap.release()
            delay = self.config.get("reconnect_delay_s", 3.0)
            time.sleep(delay)
            try:
                self._open_cap()
            except RuntimeError as exc:
                raise StopIteration(f"Camera source lost — reconnect failed: {exc}") from exc
            ok, frame = self._cap.read()
        if not ok:
            raise StopIteration("Camera source lost")
        ctx.frame = frame
        ctx.timestamp = time.time()
        return ctx

    def teardown(self):
        if hasattr(self, "_cap"):
            self._cap.release()
