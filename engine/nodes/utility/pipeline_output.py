"""Pipeline Output — marks the exit point(s) of a reusable pipeline template.

At runtime this node is a no-op; its only purpose is to be detected by the
frontend when the user saves a pipeline as a template node, so that its named
inputs become the output handles of the resulting composite node.
"""
import logging
from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext
from engine.core.node_registry import register

logger = logging.getLogger(__name__)


@register("pipeline_output")
class PipelineOutputNode(BaseNode):
    def initialize(self):
        logger.debug("[PipelineOutput] node %s — label=%s", self.node_id, self.config.get("label", "output"))

    def process(self, ctx: FrameContext) -> FrameContext:
        return ctx
