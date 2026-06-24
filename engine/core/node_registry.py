"""Maps node type strings to their BaseNode subclasses."""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.nodes.base import BaseNode

_registry: dict[str, type["BaseNode"]] = {}


def register(node_type: str):
    def decorator(cls):
        _registry[node_type] = cls
        return cls
    return decorator


def get_node_class(node_type: str) -> type["BaseNode"]:
    cls = _registry.get(node_type)
    if not cls:
        raise ValueError(f"Unknown node type: {node_type!r}. Registered: {list(_registry)}")
    return cls


def _import_all():
    """Import all node modules so their @register decorators execute."""
    import engine.nodes.input.camera          # noqa: F401
    import engine.nodes.input.video_file      # noqa: F401
    import engine.nodes.input.image_directory # noqa: F401
    import engine.nodes.processing.preprocess     # noqa: F401
    import engine.nodes.processing.model_inference # noqa: F401
    import engine.nodes.processing.postprocess_nms # noqa: F401
    import engine.nodes.spatial.draw_roi      # noqa: F401
    import engine.nodes.spatial.draw_line     # noqa: F401
    import engine.nodes.spatial.object_tracker # noqa: F401
    import engine.nodes.spatial.counter       # noqa: F401
    import engine.nodes.utility.python_function # noqa: F401
    import engine.nodes.utility.filter_node   # noqa: F401
    import engine.nodes.utility.param_node    # noqa: F401
    import engine.nodes.cpp.cpp_node          # noqa: F401
    import engine.nodes.output.stream_viewer  # noqa: F401
    import engine.nodes.output.video_writer   # noqa: F401
    import engine.nodes.output.trigger_webhook # noqa: F401


_import_all()
