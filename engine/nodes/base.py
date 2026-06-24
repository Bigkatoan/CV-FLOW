"""BaseNode ABC — all engine nodes must implement this interface."""
from __future__ import annotations
from abc import ABC, abstractmethod
from engine.core.frame_context import FrameContext


class BaseNode(ABC):
    node_id: str = ""
    config: dict

    def setup(self, node_id: str, config: dict) -> None:
        self.node_id = node_id
        self.config = config
        self.initialize()

    def initialize(self) -> None:
        """Override to set up resources (open files, load weights, etc.)."""

    @abstractmethod
    def process(self, ctx: FrameContext) -> FrameContext:
        """Process one frame. Modify ctx in-place and return it."""

    def teardown(self) -> None:
        """Override to release resources."""
