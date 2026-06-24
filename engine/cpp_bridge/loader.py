"""Load a compiled C++ .so and wire up the cv_flow_* function signatures."""
import ctypes
import json
from pathlib import Path
from engine.core.frame_context import CVFlowCtxC


def load_cpp_node(so_path: str) -> ctypes.CDLL:
    lib = ctypes.CDLL(so_path)

    # void cv_flow_setup(const char* config_json)
    lib.cv_flow_setup.argtypes = [ctypes.c_char_p]
    lib.cv_flow_setup.restype  = None

    # void cv_flow_process(CVFlowCtx* ctx)
    lib.cv_flow_process.argtypes = [ctypes.POINTER(CVFlowCtxC)]
    lib.cv_flow_process.restype  = None

    # void cv_flow_teardown(void)
    lib.cv_flow_teardown.argtypes = []
    lib.cv_flow_teardown.restype  = None

    # const char* cv_flow_version(void)  — optional
    try:
        lib.cv_flow_version.argtypes = []
        lib.cv_flow_version.restype  = ctypes.c_char_p
    except AttributeError:
        pass

    return lib
