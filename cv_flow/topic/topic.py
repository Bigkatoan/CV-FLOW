"""
cv_flow.topic.topic — Topic class and global registry.

Usage:
    from cv_flow.topic.topic import Topic, get_topic, list_topics, clear_topics

    Topic.from_file("topics/camera_frame.topic")
    cam = get_topic("camera_frame")
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from cv_flow.topic.types import TopicDef
from cv_flow.topic.parser import parse_topic_file, load_topics_dir

# ── Global registry ───────────────────────────────────────────────────────────
_REGISTRY: dict[str, "Topic"] = {}


class Topic:
    """
    Wraps TopicDef and auto-registers itself in the global registry.
    """

    def __init__(self, topic_def: TopicDef, *, overwrite: bool = False) -> None:
        name = topic_def.name
        if name in _REGISTRY and not overwrite:
            raise ValueError(
                f"Topic '{name}' is already registered. "
                "Pass overwrite=True to replace it."
            )
        self._def = topic_def
        _REGISTRY[name] = self

    # ── convenience constructors ──────────────────────────────────────────────

    @classmethod
    def from_file(cls, path: str | Path, *, overwrite: bool = False) -> "Topic":
        """Parse a .topic file and register it."""
        td = parse_topic_file(path)
        return cls(td, overwrite=overwrite)

    @classmethod
    def from_def(cls, topic_def: TopicDef, *, overwrite: bool = False) -> "Topic":
        """Register an existing TopicDef."""
        return cls(topic_def, overwrite=overwrite)

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._def.name

    @property
    def definition(self) -> TopicDef:
        return self._def

    @property
    def input_port(self):
        return self._def.input_port

    @property
    def output_port(self):
        return self._def.output_port

    @property
    def elastic(self) -> bool:
        return self._def.elastic

    @property
    def queue_depth(self) -> int:
        return self._def.queue_depth

    @property
    def drop_mode(self) -> bool:
        return self._def.drop_mode

    def __repr__(self) -> str:
        return f"Topic({self.name!r})"


# ── Registry helpers ──────────────────────────────────────────────────────────

def get_topic(name: str) -> Topic:
    """
    Look up a registered topic by name.

    Raises
    ------
    KeyError if the topic has not been registered.
    """
    if name not in _REGISTRY:
        registered = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(
            f"Topic '{name}' is not registered. "
            f"Registered topics: {registered}"
        )
    return _REGISTRY[name]


def list_topics() -> dict[str, Topic]:
    """Return a shallow copy of the full registry."""
    return dict(_REGISTRY)


def clear_topics() -> None:
    """Remove all topics from the registry (useful for tests)."""
    _REGISTRY.clear()


def load_topics(directory: str | Path, *, overwrite: bool = False) -> dict[str, Topic]:
    """
    Load all *.topic files in a directory and register them.

    Returns a {name → Topic} dict.
    """
    topic_defs = load_topics_dir(directory)
    return {
        name: Topic(td, overwrite=overwrite)
        for name, td in topic_defs.items()
    }
