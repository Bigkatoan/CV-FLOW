"""
Node registry — maps type string → BaseNode subclass.

@register(type_str) decorator stores class in _REGISTRY.
get_node_class(type_str) returns the class or raises ValueError.
get_registry() returns a copy of the full registry dict.

Previously this was a no-op shim; now it is functional.
All engine node modules that use @register will be auto-discovered
by pipeline_builder._ensure_nodes_discovered() at runtime.
"""
from __future__ import annotations

_REGISTRY: dict[str, type] = {}


def register(node_type: str):
    """Register a BaseNode subclass under a type string.

    Usage::

        @register("my_node")
        class MyNode(BaseNode):
            ...
    """
    def decorator(cls):
        _REGISTRY[node_type] = cls
        return cls
    return decorator


def get_node_class(node_type: str) -> type:
    """Return the class registered under *node_type*.

    Raises ValueError if not found (includes available types in message).
    """
    if node_type not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise ValueError(
            f"Unknown node type: {node_type!r}. "
            f"Registered types: {available}"
        )
    return _REGISTRY[node_type]


def get_registry() -> dict[str, type]:
    """Return a snapshot copy of the registry (safe to iterate while registering)."""
    return dict(_REGISTRY)
