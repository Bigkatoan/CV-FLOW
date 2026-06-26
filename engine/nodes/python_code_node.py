"""
PythonCodeNode — executes user-written Python functions as a pipeline node.

Function-based interface (no class required):
  def setup():               – called once at startup; config dict available as global
  def loop(p1, p2):          – called every frame; params are input ports
  def iteration(p1, active): – called every frame; active is a control signal (not a port)
  def teardown():            – called once at shutdown

Parameter → input port routing:
  "frame"      → ctx.frame  (numpy BGR ndarray)
  "metadata"   → ctx.metadata  (the shared dict)
  "detections" → ctx.detections
  anything else → ctx.metadata.get(name)

Return → output routing:
  ndarray → ctx.frame
  tuple   → first ndarray element → ctx.frame; rest ignored

Built-in helpers injected into user namespace:
  show_image(img, label="") – encode image and queue for Properties panel display
  show_text(text)           – queue text for Properties panel display
  config                    – the node's config dict (read-only recommended)
"""
from __future__ import annotations
import base64
import inspect
import logging
import time
import traceback
from collections import deque
from typing import Any

import cv2
import numpy as np

from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext
from engine.streaming import ws_server

logger = logging.getLogger(__name__)

_DEFAULT_LOOP_CODE = '''\
import cv2

def setup():
    # Runs once — import libs, load models, open resources.
    # "config" dict is available as a module-level global.
    pass

def loop(frame):
    # Called every frame. Parameters become input ports.
    #   "frame"     → ctx.frame (numpy BGR array)
    #   other names → ctx.metadata.get(name)
    # Return value (ndarray) → ctx.frame. StopIteration stops pipeline.
    if isinstance(frame, np.ndarray):
        show_image(frame)
        show_text(f"shape: {frame.shape}")
    else:
        show_text(f"frame: {type(frame).__name__}")
    return frame

def teardown():
    # Release resources (cameras, files, models).
    pass
'''

_DEFAULT_ITERATION_CODE = '''\
def setup():
    pass

def iteration(frame, active):
    # Called every frame. Body runs only when active=True.
    # active_key in config sets which ctx.metadata key controls this.
    # "active" is a control signal — not an input port.
    if active:
        show_text("Active!")
    return frame

def teardown():
    pass
'''


def _route_input(name: str, ctx: FrameContext) -> Any:
    """Map a function parameter name to the corresponding ctx field."""
    if name == "frame":
        return ctx.frame
    if name == "metadata":
        return ctx.metadata
    if name == "detections":
        return ctx.detections
    return ctx.metadata.get(name)


def _apply_output(result: Any, ctx: FrameContext) -> FrameContext:
    """Write the function's return value back into ctx."""
    if result is None or isinstance(result, bool):
        return ctx
    if isinstance(result, np.ndarray):
        ctx.frame = result
        return ctx
    if isinstance(result, tuple):
        for val in result:
            if isinstance(val, np.ndarray):
                ctx.frame = val
                break
    return ctx


class PythonCodeNode(BaseNode):
    """
    Wraps user-defined top-level Python functions (setup / loop or iteration / teardown).
    Injects show_image(), show_text(), and config into the execution namespace.
    """

    def setup(self, node_id: str, config: dict[str, Any], node_type: str) -> None:
        self.node_id   = node_id
        self.config    = config
        self.node_type = node_type
        # Profiling attributes required by BaseNode / PipelineRunner
        self._timing_window = deque(maxlen=100)
        self._frame_count   = 0
        self._error_count   = 0
        self._last_fps_t    = time.monotonic()
        self._fps           = 0.0
        self._current_ctx: FrameContext | None = None

        mode       = config.get("mode", "loop")
        active_key = config.get("active_key", "active")
        code       = config.get("code", "").strip()

        if not code:
            code = _DEFAULT_LOOP_CODE if mode == "loop" else _DEFAULT_ITERATION_CODE

        # viz items accumulated per frame; pipeline_runner reads via get_viz()
        self._latest_viz: list[dict] = []

        # Build helpers that write to _latest_viz via closure
        def _show_image(img, label: str = "") -> None:
            if img is None:
                return
            try:
                arr = img
                if isinstance(arr, np.ndarray):
                    h, w = arr.shape[:2]
                    if w > 320:
                        arr = cv2.resize(arr, (320, int(h * 320 / w)))
                    ok, buf = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ok:
                        b64 = base64.b64encode(buf.tobytes()).decode()
                        self._latest_viz.append({"type": "viz_image", "b64": b64, "label": label})
            except Exception as exc:
                logger.debug("show_image error in node %s: %s", node_id, exc)

        def _show_text(text) -> None:
            self._latest_viz.append({"type": "viz_text", "data": str(text)})

        # UI-block helpers — return current config value (or default) when called in setup()
        def _slider(name: str, min_val=0, max_val=100, default=50, **_):
            return config.get(name, default)

        def _checkbox(name: str, default: bool = False, **_):
            return config.get(name, default)

        def _text_input(name: str, default: str = "", **_):
            return config.get(name, default)

        def _button(name: str, **_):
            pass  # no-op at runtime; only the label matters for the UI

        def _send_frame(img, quality: int = 80) -> None:
            """Broadcast a frame JPEG to the frontend stream viewer."""
            if self._current_ctx is None:
                return
            try:
                if isinstance(img, np.ndarray):
                    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
                    if ok:
                        ws_server.send_frame(self._current_ctx.session_id, buf.tobytes())
                elif isinstance(img, (bytes, bytearray)):
                    ws_server.send_frame(self._current_ctx.session_id, bytes(img))
            except Exception as exc:
                logger.debug("send_frame error in node %s: %s", node_id, exc)

        def _send_event(event: dict) -> None:
            """Push a custom event dict to the frontend WebSocket."""
            if self._current_ctx is None:
                return
            try:
                ws_server.send_event(self._current_ctx.session_id, event)
            except Exception as exc:
                logger.debug("send_event error in node %s: %s", node_id, exc)

        # Execute user code in an isolated namespace
        ns: dict[str, Any] = {
            "show_image":   _show_image,
            "show_text":    _show_text,
            "send_frame":   _send_frame,
            "send_event":   _send_event,
            "config":       config,
            "slider":       _slider,
            "checkbox":     _checkbox,
            "text_input":   _text_input,
            "button":       _button,
            "FrameContext": FrameContext,
            "np":           np,
        }
        try:
            exec(compile(code, f"<node:{node_id}>", "exec"), ns)
        except Exception:
            raise ValueError(
                f"Node {node_id}: syntax/import error in code:\n{traceback.format_exc()}"
            )

        # Locate setup / loop / iteration / teardown at top level
        self._fn_setup     = ns.get("setup")
        self._fn_loop      = ns.get("loop")
        self._fn_iteration = ns.get("iteration")
        self._fn_teardown  = ns.get("teardown")

        # When both loop and iteration exist, respect config mode
        if self._fn_loop and self._fn_iteration:
            if mode == "iteration":
                self._fn_loop = None
            else:
                self._fn_iteration = None

        if not self._fn_loop and not self._fn_iteration:
            raise ValueError(
                f"Node {node_id}: code must define def loop(...) or def iteration(..., active)."
            )

        self._mode       = "iteration" if self._fn_iteration else "loop"
        self._active_key = active_key

        # Inspect parameter names for input routing
        fn = self._fn_iteration or self._fn_loop
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        # "active" in iteration is a control signal, not an input port
        if self._mode == "iteration" and "active" in params:
            params = [p for p in params if p != "active"]
        self._input_params = params

        if self._fn_setup:
            try:
                self._fn_setup()
            except Exception:
                raise ValueError(
                    f"Node {node_id}: error in setup():\n{traceback.format_exc()}"
                )

        logger.info(
            "PythonCodeNode %r ready — mode=%s, params=%s",
            node_id, self._mode, self._input_params,
        )

    def get_viz(self) -> list[dict]:
        """Return viz items accumulated during the most recent process() call."""
        return self._latest_viz

    def process(self, ctx: FrameContext) -> FrameContext:
        self._current_ctx = ctx   # expose to send_frame / send_event helpers
        self._latest_viz = []  # clear for this frame
        args = [_route_input(p, ctx) for p in self._input_params]
        try:
            if self._mode == "iteration":
                raw    = ctx.metadata.get(self._active_key, True)
                active = bool(raw) if not isinstance(raw, bool) else raw
                result = self._fn_iteration(*args, active)
            else:
                result = self._fn_loop(*args)
            return _apply_output(result, ctx)
        except StopIteration:
            raise
        except Exception:
            logger.error(
                "Node %s runtime error:\n%s",
                self.node_id, traceback.format_exc(),
            )
            return ctx

    def teardown(self) -> None:
        if self._fn_teardown:
            try:
                self._fn_teardown()
            except Exception:
                logger.warning(
                    "Node %s teardown error:\n%s",
                    self.node_id, traceback.format_exc(),
                )
