"""
CV-FLOW Node SDK — ergonomic API for writing new nodes.

Two styles are supported:

1. Class-based (full control):
   @cv_node(label="My Node", group="processing")
   class MyNode(BaseNode):
       def initialize(self): ...
       def process(self, ctx): ...
       def teardown(self): ...

2. Function-based (minimal boilerplate):
   @cv_node_fn(label="Grayscale", inputs=["frame"], outputs=["frame"])
   def grayscale(frame):
       return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[..., None].repeat(3, -1)

Both styles:
  - Return None from process() / the function to DROP the current frame
    (it will not be forwarded to downstream nodes).
  - Are fully backward-compatible with BaseNode and the sequential runner.
  - Can be tested in isolation with NodeTestHarness (no camera/model needed).
  - Automatically export metadata so the frontend can render a config panel.
"""
from __future__ import annotations

import contextlib
import inspect
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Type

import numpy as np

from engine.core.frame_context import Detection, FrameContext
from engine.nodes.base import BaseNode

# ── Port types ────────────────────────────────────────────────────────────────

class PortType(Enum):
    """Semantic type annotation for a node port."""
    FRAME      = "frame"       # np.ndarray BGR uint8 (H×W×3)
    TENSOR     = "tensor"      # np.ndarray float32 CHW preprocessed
    DETECTIONS = "detections"  # List[Detection]
    METADATA   = "metadata"    # dict (pass-through)
    EVENT      = "event"       # JSON-serializable dict
    EMBEDDINGS = "embeddings"  # List[np.ndarray float32 512-dim]
    COUNT      = "count"       # int


# ── Node metadata registry ────────────────────────────────────────────────────

@dataclass
class NodeMeta:
    node_type:    str
    label:        str
    group:        str
    icon:         str
    inputs:       list[str]
    outputs:      list[str]
    port_types:   dict[str, PortType] = field(default_factory=dict)
    default_config: dict = field(default_factory=dict)
    node_class:   Optional[Type] = None

_REGISTRY: dict[str, NodeMeta] = {}


def get_registry() -> dict[str, NodeMeta]:
    """Return all SDK-registered node types."""
    return dict(_REGISTRY)


# ── @cv_node decorator ────────────────────────────────────────────────────────

def cv_node(
    label: str = "",
    group: str = "custom",
    icon: str  = "box",
    inputs:  list[str] = ("frame",),
    outputs: list[str] = ("frame",),
    port_types: dict[str, PortType] | None = None,
    default_config: dict | None = None,
) -> Callable[[Type], Type]:
    """
    Class decorator that registers a BaseNode subclass with SDK metadata.

    The decorated class remains a fully functional BaseNode subclass — it
    works in both sequential and multi-process mode.

    Example
    -------
    @cv_node(label="Edge Detect", group="processing")
    class EdgeDetectNode(BaseNode):
        def process(self, ctx):
            import cv2
            ctx.frame = cv2.Canny(ctx.frame, 100, 200)
            return ctx
    """
    def decorator(cls: Type) -> Type:
        # Infer node_type from class name (CamelCase → snake_case)
        import re
        snake = re.sub(r"(?<!^)(?=[A-Z])", "_", cls.__name__).lower()
        node_type = snake.replace("_node", "")

        meta = NodeMeta(
            node_type=node_type,
            label=label or cls.__name__,
            group=group,
            icon=icon,
            inputs=list(inputs),
            outputs=list(outputs),
            port_types=port_types or {},
            default_config=default_config or {},
            node_class=cls,
        )
        _REGISTRY[node_type] = meta

        # Attach metadata to class
        cls.__cv_node_meta__ = meta
        return cls
    return decorator


# ── @cv_node_fn decorator ─────────────────────────────────────────────────────

def cv_node_fn(
    label: str = "",
    group: str = "custom",
    icon:  str  = "box",
    inputs:  list[str] = ("frame",),
    outputs: list[str] = ("frame",),
    port_types: dict[str, PortType] | None = None,
    default_config: dict | None = None,
) -> Callable[[Callable], Type]:
    """
    Function decorator that wraps a plain Python function as a BaseNode class.

    The function signature determines inputs:
      - Positional args (except 'self') → input port names
      - Keyword-only args (after '*') → configuration values (injected from config dict)
      - Special kwarg '_node_id' → receives the node's ID at runtime

    The function return value:
      - np.ndarray → ctx.frame
      - tuple      → (frame, detections)  or  (frame, detections, metadata)
      - None       → drop frame (not forwarded downstream)

    Example
    -------
    @cv_node_fn(label="Blur", group="processing", inputs=["frame"], outputs=["frame"])
    def gaussian_blur(frame, *, kernel_size=15):
        import cv2
        return cv2.GaussianBlur(frame, (kernel_size | 1, kernel_size | 1), 0)
    """
    def decorator(fn: Callable) -> Type:
        sig    = inspect.signature(fn)
        params = list(sig.parameters.values())

        # Separate positional params (→ input ports) and kw-only (→ config keys)
        pos_params = [p for p in params if p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        ) and p.name != "self"]
        kw_params  = [p for p in params if p.kind == inspect.Parameter.KEYWORD_ONLY
                      and not p.name.startswith("_")]

        import re
        snake = re.sub(r"(?<!^)(?=[A-Z])", "_", fn.__name__).lower()
        node_type = snake

        class _FnNode(BaseNode):
            _fn     = staticmethod(fn)
            _inputs  = list(inputs)
            _outputs = list(outputs)

            def initialize(self) -> None:
                # Pre-extract config values with defaults from function signature
                self._kw_vals: dict[str, Any] = {}
                for p in kw_params:
                    default = p.default if p.default is not inspect.Parameter.empty else None
                    self._kw_vals[p.name] = self.config.get(p.name, default)

            def process(self, ctx: FrameContext) -> Optional[FrameContext]:
                # Build positional args from ctx
                pos_args = []
                for p in pos_params:
                    if p.name == "frame":
                        pos_args.append(ctx.ensure_cpu())
                    elif p.name == "detections":
                        pos_args.append(ctx.detections)
                    elif p.name == "metadata":
                        pos_args.append(ctx.metadata)
                    else:
                        pos_args.append(ctx.metadata.get(p.name))

                # Keyword args from config + special values
                kw = dict(self._kw_vals)
                kw["_node_id"] = self.node_id  # always injected

                result = self._fn(*pos_args, **kw)
                return _unpack_fn_result(result, ctx)

        _FnNode.__name__     = fn.__name__
        _FnNode.__qualname__  = fn.__qualname__
        _FnNode.__module__    = fn.__module__
        _FnNode.__doc__       = fn.__doc__

        meta = NodeMeta(
            node_type=node_type,
            label=label or fn.__name__,
            group=group,
            icon=icon,
            inputs=list(inputs),
            outputs=list(outputs),
            port_types=port_types or {},
            default_config=default_config or {
                p.name: p.default for p in kw_params
                if p.default is not inspect.Parameter.empty
            },
            node_class=_FnNode,
        )
        _REGISTRY[node_type] = meta
        _FnNode.__cv_node_meta__ = meta

        return _FnNode
    return decorator


def _unpack_fn_result(
    result: Any,
    ctx: FrameContext,
) -> Optional[FrameContext]:
    """Map various return types from a cv_node_fn function to FrameContext mutations."""
    if result is None:
        return None  # drop frame

    if isinstance(result, np.ndarray):
        ctx.frame = result
        return ctx

    if isinstance(result, (list, tuple)):
        if len(result) >= 1 and isinstance(result[0], np.ndarray):
            ctx.frame = result[0]
        if len(result) >= 2 and isinstance(result[1], list):
            ctx.detections = result[1]
        if len(result) >= 3 and isinstance(result[2], dict):
            ctx.metadata.update(result[2])
        return ctx

    if isinstance(result, dict):
        ctx.metadata.update(result)
        return ctx

    if isinstance(result, FrameContext):
        return result

    # Scalar or other — store in metadata under "output"
    ctx.metadata["output"] = result
    return ctx


# ── NodeTestHarness ───────────────────────────────────────────────────────────

class NodeTestHarness:
    """
    Context manager for testing a node in isolation (no camera, no pipeline).

    Usage
    -----
    with NodeTestHarness(MyNode, config={"threshold": 0.4}) as h:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ctx   = h.make_context(frame=frame)
        result = h.run_frame(ctx)
        assert result is not None
        assert len(result.detections) == 0

    The harness calls initialize() on enter and teardown() on exit —
    exactly mirroring the production lifecycle.
    """

    def __init__(
        self,
        node_class: Type,
        config: dict | None = None,
        node_id: str = "test_node",
        node_type: str = "",
    ) -> None:
        self._node_class = node_class
        self._config     = config or {}
        self._node_id    = node_id
        self._node_type  = node_type or node_class.__name__
        self._node: Optional[BaseNode] = None

    def __enter__(self) -> "NodeTestHarness":
        self._node = self._node_class()
        self._node.setup(self._node_id, self._config, self._node_type)
        return self

    def __exit__(self, *_: Any) -> None:
        if self._node:
            with contextlib.suppress(Exception):
                self._node.teardown()
            self._node = None

    def make_context(
        self,
        frame: Optional[np.ndarray] = None,
        detections: Optional[list[Detection]] = None,
        metadata: Optional[dict] = None,
        frame_number: int = 0,
    ) -> FrameContext:
        """Build a FrameContext for testing."""
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        return FrameContext(
            frame=frame,
            frame_number=frame_number,
            timestamp=time.time(),
            session_id="test",
            detections=detections or [],
            metadata=metadata or {},
        )

    def run_frame(
        self,
        ctx: Optional[FrameContext] = None,
        frame: Optional[np.ndarray] = None,
        detections: Optional[list[Detection]] = None,
        metadata: Optional[dict] = None,
    ) -> Optional[FrameContext]:
        """
        Run one frame through the node.  Returns the output FrameContext,
        or None if the node dropped the frame.
        Raises exceptions from the node so tests can assert on errors.
        """
        if self._node is None:
            raise RuntimeError("NodeTestHarness not entered — use 'with' statement")

        if ctx is None:
            ctx = self.make_context(frame=frame, detections=detections, metadata=metadata)

        return self._node.process(ctx)

    def run_frames(
        self,
        frames: list[np.ndarray],
        **kwargs: Any,
    ) -> list[Optional[FrameContext]]:
        """Run multiple frames and return a list of results."""
        results = []
        for i, f in enumerate(frames):
            ctx = self.make_context(frame=f, frame_number=i, **kwargs)
            results.append(self.run_frame(ctx=ctx))
        return results


# ── Convenience helpers ───────────────────────────────────────────────────────

def load_test_image(path: Optional[str] = None, shape: tuple = (480, 640, 3)) -> np.ndarray:
    """
    Load an image for testing.  Returns a solid grey frame if path is None
    or the file can't be loaded.
    """
    if path:
        try:
            import cv2
            img = cv2.imread(path)
            if img is not None:
                return img
        except Exception:
            pass
    return np.full(shape, 128, dtype=np.uint8)


def make_detection(
    x1=10, y1=10, x2=100, y2=100,
    confidence=0.9,
    class_id=0,
    class_name="object",
    track_id=-1,
) -> Detection:
    """Create a Detection for use in tests."""
    return Detection(
        x1=x1, y1=y1, x2=x2, y2=y2,
        confidence=confidence,
        class_id=class_id,
        class_name=class_name,
        track_id=track_id,
    )
