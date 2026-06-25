"""
Node registry — legacy shim, kept so old import statements don't crash.

The new node system uses PythonCodeNode and CppCodeNode directly in pipeline_builder.py.
@register is a no-op; get_node_class raises immediately to catch accidental use.
"""
from __future__ import annotations


def register(_node_type: str):
    """No-op decorator retained for backward compatibility."""
    def decorator(cls):
        return cls
    return decorator


def get_node_class(node_type: str):
    raise ValueError(
        f"get_node_class() is no longer used. "
        f"Unknown node type: {node_type!r}. "
        f"Only 'python_node' and 'cpp_node' are supported by the engine."
    )
