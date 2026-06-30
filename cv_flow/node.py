"""
cv_flow.node — Node base class.

Nodes are the unit of computation in a CV-FLOW pipeline. Each node:
  1. Declares publishers (output topics) and subscribers (input topics).
  2. Implements spin_once() — called once per executor iteration.
  3. Optionally overrides initialize() / shutdown().
"""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cv_flow.topic.publisher import Publisher
    from cv_flow.topic.subscriber import Subscriber
    from cv_flow.topic.topic import Topic


class Node:
    """
    Base class for all CV-FLOW pipeline nodes.

    Subclass and override:
      initialize()  — called once before spin starts
      spin_once()   — called in a loop by the Executor
      shutdown()    — called once when pipeline stops

    Raise StopIteration inside spin_once() to signal normal termination.
    """

    def __init__(self, name: str | None = None) -> None:
        self.name       = name or f"{type(self).__name__}_{uuid.uuid4().hex[:6]}"
        self._session_id: str | None = None  # set by Executor before spin

        self._publishers:  list["Publisher"]  = []
        self._subscribers: list["Subscriber"] = []

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Override to perform setup before the spin loop starts."""

    def spin_once(self) -> None:
        """Override to implement node logic. Raise StopIteration to exit."""
        raise NotImplementedError(
            f"{type(self).__name__}.spin_once() is not implemented"
        )

    def shutdown(self) -> None:
        """Override to clean up resources when the pipeline stops."""

    # ── pub / sub factory ─────────────────────────────────────────────────────

    def advertise(
        self,
        topic,   # Topic | str
        *,
        queue_depth: int | None = None,
    ) -> "Publisher":
        """
        Create a Publisher on the given topic's output port.

        Parameters
        ----------
        topic       : Topic object or topic name string (looked up from registry).
        queue_depth : Override the topic's default queue_depth if provided.
        """
        from cv_flow.dam.bus import PortBus
        from cv_flow.topic.publisher import Publisher
        from cv_flow.topic.topic import get_topic

        if isinstance(topic, str):
            topic = get_topic(topic)

        td    = topic.definition
        port  = td.output_port
        depth = queue_depth if queue_depth is not None else td.queue_depth

        bus_name = f"{td.name}_{self._session_id or 'default'}"
        bus = PortBus(
            bus_name,
            slot_bytes=max(port.total_bytes, 1),
            queue_depth=depth,
            create=True,
            drop_mode=td.drop_mode,
        )
        pub = Publisher(bus, port)
        self._publishers.append(pub)
        return pub

    def subscribe(
        self,
        topic,   # Topic | str
        *,
        timeout_ms: int = 30,
    ) -> "Subscriber":
        """
        Create a Subscriber on the given topic's output port.

        Parameters
        ----------
        topic      : Topic object or topic name string.
        timeout_ms : Default read timeout for this subscriber.
        """
        from cv_flow.dam.bus import PortBus
        from cv_flow.topic.subscriber import Subscriber
        from cv_flow.topic.topic import get_topic

        if isinstance(topic, str):
            topic = get_topic(topic)

        td   = topic.definition
        port = td.output_port

        bus_name = f"{td.name}_{self._session_id or 'default'}"
        bus = PortBus(
            bus_name,
            slot_bytes=max(port.total_bytes, 1),
            create=False,
        )
        sub = Subscriber(bus, port)
        self._subscribers.append(sub)
        return sub

    # ── repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
