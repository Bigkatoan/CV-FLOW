import logging
import numpy as np
import cv2
from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext
from engine.core.node_registry import register

logger = logging.getLogger(__name__)


@register("python_function")
class PythonFunctionNode(BaseNode):
    def initialize(self):
        code = self.config.get("code", "def process(frame, detections, params):\n    return frame, detections\n")
        self._namespace: dict = {"np": np, "cv2": cv2}
        try:
            exec(compile(code, "<python_function_node>", "exec"), self._namespace)
        except SyntaxError as e:
            logger.error("Syntax error in PythonFunctionNode %s: %s", self.node_id, e)
            self._namespace["process"] = lambda f, d, p: (f, d)

    def process(self, ctx: FrameContext) -> FrameContext:
        fn = self._namespace.get("process")
        if fn is None:
            return ctx
        try:
            result = fn(ctx.frame, ctx.detections, ctx.metadata.get("params", {}))
            if isinstance(result, tuple) and len(result) == 2:
                ctx.frame, ctx.detections = result
        except Exception as e:
            logger.error("Runtime error in PythonFunctionNode %s frame %d: %s",
                         self.node_id, ctx.frame_number, e)
        return ctx
