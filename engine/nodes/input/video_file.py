import time
import cv2
from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext
from engine.core.node_registry import register


@register("video_file")
class VideoFileNode(BaseNode):
    def initialize(self):
        path = self.config.get("file_path", "")
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {path!r}")
        self._loop = self.config.get("loop", False)
        fps_limit = self.config.get("fps_limit", 0)
        native_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30
        fps = fps_limit if fps_limit > 0 else native_fps
        self._frame_delay = 1.0 / fps
        self._last_frame_time = 0.0

    def process(self, ctx: FrameContext) -> FrameContext:
        now = time.monotonic()
        wait = self._frame_delay - (now - self._last_frame_time)
        if wait > 0:
            time.sleep(wait)
        self._last_frame_time = time.monotonic()

        ok, frame = self._cap.read()
        if not ok:
            if self._loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
            if not ok:
                raise StopIteration("Video file exhausted")
        ctx.frame = frame
        ctx.timestamp = time.time()
        return ctx

    def teardown(self):
        if hasattr(self, "_cap"):
            self._cap.release()
