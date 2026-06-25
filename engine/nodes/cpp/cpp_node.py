"""
CppNode — loads a compiled C++ shared library and calls the cv_flow ABI each frame.

The user writes C++ code in the properties panel; the backend compiles it to a .so
and stores the hash in config["compiled_so_hash"].  The engine then loads that .so
and calls cv_flow_setup / cv_flow_process / cv_flow_teardown.

C++ code template (what the user writes):
  #include <opencv2/opencv.hpp>
  extern "C" {
    void cv_flow_setup(const char* config_json) { /* init */ }
    void cv_flow_process(CvFlowContext* ctx)     { /* per-frame */ }
    void cv_flow_teardown()                      { /* cleanup */ }
  }
"""
import ctypes
import json
import logging
import os
from pathlib import Path
from typing import Any

from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext

try:
    from engine.cpp_bridge.loader import load_cpp_node
    from engine.cpp_bridge.context_shm import CppBridge
    _BRIDGE_AVAILABLE = True
except ImportError:
    _BRIDGE_AVAILABLE = False

logger = logging.getLogger(__name__)


def _compiled_dir() -> Path:
    env = os.environ.get("CVFLOW_COMPILED_DIR")
    if env:
        return Path(env)
    return Path(__file__).parent.parent.parent.parent / "backend" / "storage" / "compiled"


class CppCodeNode(BaseNode):
    """
    Wraps a user-written C++ node that was compiled to a shared library (.so).
    Calls cv_flow_setup() once, cv_flow_process() each frame, cv_flow_teardown() on stop.
    """

    def setup(self, node_id: str, config: dict[str, Any], node_type: str) -> None:
        self.node_id   = node_id
        self.config    = config
        self.node_type = node_type

        if not _BRIDGE_AVAILABLE:
            raise RuntimeError(
                f"C++ bridge not available (engine/cpp_bridge missing). "
                f"Node {node_id} cannot run."
            )

        so_hash = config.get("compiled_so_hash", "")
        if not so_hash:
            raise RuntimeError(
                f"CppCodeNode {node_id}: no compiled_so_hash — compile the code first "
                f"via the backend /api/cpp/compile endpoint."
            )

        so_path = _compiled_dir() / so_hash / "node.so"
        if not so_path.exists():
            raise FileNotFoundError(f"Compiled .so not found: {so_path}")

        self._lib    = load_cpp_node(str(so_path))
        self._bridge = CppBridge()

        config_bytes = json.dumps(config).encode()
        self._lib.cv_flow_setup(config_bytes)

        try:
            ver = self._lib.cv_flow_version()
            logger.info(
                "CppCodeNode %r loaded: %s (hash=%s…)",
                node_id, ver.decode(), so_hash[:12],
            )
        except AttributeError:
            logger.info("CppCodeNode %r loaded (hash=%s…)", node_id, so_hash[:12])

    def process(self, ctx: FrameContext) -> FrameContext:
        c_ctx = self._bridge.python_to_c(ctx)
        self._lib.cv_flow_process(ctypes.byref(c_ctx))
        return self._bridge.c_to_python(ctx)

    def teardown(self) -> None:
        if hasattr(self, "_lib"):
            try:
                self._lib.cv_flow_teardown()
            except Exception as e:
                logger.warning("CppCodeNode %s teardown error: %s", self.node_id, e)
