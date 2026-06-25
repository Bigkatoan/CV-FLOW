import logging
import numpy as np
import cv2
from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext, Detection
from engine.core.node_registry import register

logger = logging.getLogger(__name__)


@register("python_function")
class PythonFunctionNode(BaseNode):
    def initialize(self):
        code = self.config.get(
            "code",
            "def process(frame, detections, params):\n    return frame, detections\n"
        )
        self._namespace: dict = {"np": np, "cv2": cv2, "Detection": Detection}
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
            self._apply_result(ctx, result)
        except Exception as e:
            logger.error(
                "Runtime error in PythonFunctionNode %s frame %d: %s",
                self.node_id, ctx.frame_number, e,
            )
        return ctx

    def _apply_result(self, ctx: FrameContext, result) -> None:
        """Dispatch flexible return types from user-defined process() functions.

        Supported return signatures:
          - np.ndarray                       → update frame only
          - (frame, detections)              → update frame + detections
          - (frame, detections, metadata)    → update all three
          - (frame, metadata_dict)           → update frame + merge metadata
          - {"frame":…, "detections":…, …}  → dict form (any keys optional)
          - None                             → no change (pass-through)
        """
        if result is None:
            return

        if isinstance(result, np.ndarray):
            ctx.frame = result
            return

        if isinstance(result, dict):
            if "frame" in result and result["frame"] is not None:
                ctx.frame = result["frame"]
            if "detections" in result and result["detections"] is not None:
                ctx.detections = result["detections"]
            if "metadata" in result and isinstance(result["metadata"], dict):
                ctx.metadata.update(result["metadata"])
            return

        if isinstance(result, (tuple, list)):
            if len(result) >= 1 and result[0] is not None:
                if isinstance(result[0], np.ndarray):
                    ctx.frame = result[0]

            if len(result) >= 2 and result[1] is not None:
                r1 = result[1]
                if isinstance(r1, list):
                    ctx.detections = r1
                elif isinstance(r1, dict):
                    ctx.metadata.update(r1)

            if len(result) >= 3 and result[2] is not None:
                r2 = result[2]
                if isinstance(r2, dict):
                    ctx.metadata.update(r2)
            return
