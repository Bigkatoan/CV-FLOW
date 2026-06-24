from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext
from engine.core.node_registry import register


@register("filter")
class FilterNode(BaseNode):
    def initialize(self):
        self._allowed = set(self.config.get("allowed_classes", []))
        self._min_conf = self.config.get("min_confidence", 0.0)
        self._min_area_pct = self.config.get("min_area_pct", 0.0)

    def process(self, ctx: FrameContext) -> FrameContext:
        if not ctx.detections:
            return ctx
        fh, fw = ctx.frame.shape[:2]
        frame_area = fw * fh
        kept = []
        for det in ctx.detections:
            if self._allowed and det.class_name not in self._allowed:
                continue
            if det.confidence < self._min_conf:
                continue
            if self._min_area_pct > 0 and (det.area / frame_area * 100) < self._min_area_pct:
                continue
            kept.append(det)
        ctx.detections = kept
        return ctx
