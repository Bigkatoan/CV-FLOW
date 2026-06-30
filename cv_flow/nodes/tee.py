"""
cv_flow.nodes.tee — Tee: fan a single topic out to N independent topics.

Every PortBus in this DAM model is a single-reader FIFO (the read cursor
lives in the shared-memory header, shared by whoever subscribes to that bus
name) — it is NOT a broadcast channel. If two different nodes both
`subscribe()` the same topic, they compete for the same queue instead of
each seeing every frame. This is the most common case where that bites:
a pipeline that needs the *original* frame both for inference (consumed by
Preprocess) and later for drawing on (consumed by DrawBbox) needs Tee to
republish the frame onto two independent topics first.
"""
from __future__ import annotations

from cv_flow.node import Node


class Tee(Node):
    """Subscribes to one topic, republishes every frame unchanged to N output topics."""

    def __init__(
        self,
        input_topic,
        output_topics: list,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self._input_topic = input_topic
        self._output_topics = list(output_topics)

    def initialize(self) -> None:
        self._sub = self.subscribe(self._input_topic)
        self._pubs = [self.advertise(t) for t in self._output_topics]

    def spin_once(self) -> None:
        data = self._sub.read(timeout_ms=30)
        if data is None:
            return
        for pub in self._pubs:
            pub.write(data)
