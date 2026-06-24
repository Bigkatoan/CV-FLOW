import cv2
from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext
from engine.core.node_registry import register


@register("video_writer")
class VideoWriterNode(BaseNode):
    _writer: cv2.VideoWriter | None = None

    def initialize(self):
        self._output_path = self.config.get("output_path", "./output.mp4")
        self._codec = self.config.get("codec", "mp4v")
        self._fps = float(self.config.get("fps", 30))
        self._writer = None   # Lazily opened on first frame (need frame dimensions)

    def process(self, ctx: FrameContext) -> FrameContext:
        if self._writer is None:
            h, w = ctx.frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*self._codec)
            self._writer = cv2.VideoWriter(self._output_path, fourcc, self._fps, (w, h))
        self._writer.write(ctx.frame)
        return ctx

    def teardown(self):
        if self._writer:
            self._writer.release()
            self._writer = None
