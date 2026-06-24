from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext
from engine.core.node_registry import register


@register("param")
class ParamNode(BaseNode):
    def initialize(self):
        self._params = self.config.get("params", {})

    def process(self, ctx: FrameContext) -> FrameContext:
        existing = ctx.metadata.get("params", {})
        ctx.metadata["params"] = {**self._params, **existing}   # existing takes precedence (live overrides)
        return ctx
