import ctypes
import json
import logging
from pathlib import Path

from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext
from engine.core.node_registry import register
from engine.cpp_bridge.loader import load_cpp_node
from engine.cpp_bridge.context_shm import CppBridge

logger = logging.getLogger(__name__)


@register("cpp_function")
class CppNode(BaseNode):
    """Loads a compiled C++ .so and calls cv_flow_process() on every frame."""

    def initialize(self):
        so_hash = self.config.get("compiled_so_hash", "")
        if not so_hash:
            raise RuntimeError(f"CppNode {self.node_id}: no compiled_so_hash — compile first")

        from app.config import settings
        so_path = settings.compiled_dir / so_hash / "node.so"
        if not so_path.exists():
            raise FileNotFoundError(f"Compiled .so not found: {so_path}")

        self._lib = load_cpp_node(str(so_path))
        self._bridge = CppBridge()

        config_bytes = json.dumps(self.config).encode()
        self._lib.cv_flow_setup(config_bytes)

        try:
            ver = self._lib.cv_flow_version()
            logger.info("CppNode %s loaded: %s (hash=%s…)", self.node_id, ver.decode(), so_hash[:12])
        except AttributeError:
            logger.info("CppNode %s loaded (hash=%s…)", self.node_id, so_hash[:12])

    def process(self, ctx: FrameContext) -> FrameContext:
        c_ctx = self._bridge.python_to_c(ctx)
        self._lib.cv_flow_process(ctypes.byref(c_ctx))
        return self._bridge.c_to_python(ctx)

    def teardown(self):
        if hasattr(self, "_lib"):
            try:
                self._lib.cv_flow_teardown()
            except Exception as e:
                logger.warning("CppNode %s teardown error: %s", self.node_id, e)
