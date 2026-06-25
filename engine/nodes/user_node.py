"""
Base class for all user-defined pipeline nodes.

Users subclass Node and override either loop() or iteration():

  class MyCamera(Node):
      def setup(self, config):
          import cv2
          self.cap = cv2.VideoCapture(config.get("device", 0))

      def loop(self, ctx):
          ret, frame = self.cap.read()
          if not ret:
              raise StopIteration
          ctx.frame = frame
          return ctx

      def teardown(self):
          self.cap.release()


  class MyFilter(Node):
      def setup(self, config):
          self.threshold = config.get("threshold", 0.5)

      def iteration(self, ctx, active):
          if active:
              ctx.metadata["filtered"] = True
          return ctx

The engine calls setup() once at start, then loop()/iteration() every frame,
then teardown() when the pipeline stops.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.core.frame_context import FrameContext


class Node:
    """
    Base class for user-defined pipeline nodes.

    Subclass this and override:
      - setup(config)      — called once at startup
      - loop(ctx)          — called every frame, no conditions
      - iteration(ctx, active)  — called every frame, runs body only when active=True
      - teardown()         — called once at shutdown
    """

    # Optionally declare port labels for the canvas (cosmetic only; engine ignores)
    inputs:  list[dict] = []   # e.g. [{"id": "in", "label": "Frame"}]
    outputs: list[dict] = []   # e.g. [{"id": "out", "label": "Result"}]

    def setup(self, config: dict) -> None:
        """
        Called once when the pipeline starts.
        Import libraries, open files, load models, allocate state here.
        config is the dict saved in the node's properties panel.
        """

    def loop(self, ctx: "FrameContext") -> "FrameContext":
        """
        Called every frame. Always executes; no condition.
        Must return ctx (or a new FrameContext).
        Raise StopIteration to signal end-of-source.
        """
        return ctx

    def iteration(self, ctx: "FrameContext", active: bool) -> "FrameContext":
        """
        Called every frame. active is read from ctx.metadata[active_key].
        Only do meaningful work when active=True; return ctx either way.
        Raise StopIteration to signal end-of-source.
        """
        return ctx

    def teardown(self) -> None:
        """
        Called once when the pipeline stops (even after errors).
        Release cameras, close files, free GPU memory, etc.
        """
