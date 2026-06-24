import glob
import time
import cv2
from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext
from engine.core.node_registry import register


@register("image_directory")
class ImageDirectoryNode(BaseNode):
    def initialize(self):
        directory = self.config.get("directory_path", "")
        pattern   = self.config.get("pattern", "*.jpg")
        self._paths = sorted(glob.glob(f"{directory}/{pattern}"))
        if not self._paths:
            raise RuntimeError(f"No images found in {directory!r} matching {pattern!r}")
        self._idx = 0
        self._delay = self.config.get("delay_ms", 100) / 1000.0

    def process(self, ctx: FrameContext) -> FrameContext:
        if self._idx >= len(self._paths):
            raise StopIteration("Image directory exhausted")
        frame = cv2.imread(self._paths[self._idx])
        if frame is None:
            raise RuntimeError(f"Failed to read image: {self._paths[self._idx]}")
        ctx.frame = frame
        ctx.timestamp = time.time()
        self._idx += 1
        if self._delay > 0:
            time.sleep(self._delay)
        return ctx
