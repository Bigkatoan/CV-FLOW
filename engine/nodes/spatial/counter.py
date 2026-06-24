import asyncio
from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext
from engine.core.node_registry import register
from engine.nodes.spatial.draw_line import DrawLineNode


@register("counter")
class CounterNode(BaseNode):
    def initialize(self):
        self._trigger_type = self.config.get("trigger_type", "line_cross")
        self._trigger_id   = self.config.get("trigger_id", "line_1")
        self._count_classes = set(self.config.get("count_classes", []))
        self._count = 0
        self._prev_sides: dict[int, float] = {}   # track_id → last cross-product sign

    def process(self, ctx: FrameContext) -> FrameContext:
        if self._trigger_type == "line_cross":
            self._check_line_cross(ctx)
        elif self._trigger_type in ("zone_enter", "zone_exit"):
            self._check_zone(ctx)
        ctx.metadata[f"counter_{self.node_id}"] = self._count
        return ctx

    def _check_line_cross(self, ctx: FrameContext):
        line_key = f"line_{self._trigger_id}"
        line_data = ctx.metadata.get(line_key)
        if not line_data:
            return
        p0, p1 = line_data["p0"], line_data["p1"]

        for det in ctx.detections:
            if self._count_classes and det.class_name not in self._count_classes:
                continue
            if det.track_id < 0:
                continue
            cx, cy = det.center
            sign = (p1[0] - p0[0]) * (cy - p0[1]) - (p1[1] - p0[1]) * (cx - p0[0])
            prev = self._prev_sides.get(det.track_id)
            if prev is not None and prev * sign < 0:   # Sign changed → crossed line
                self._count += 1
            self._prev_sides[det.track_id] = sign

    def _check_zone(self, ctx: FrameContext):
        zone_key = f"zone_{self._trigger_id}_count"
        current_count = ctx.metadata.get(zone_key, 0)
        # Simplified: count rising edge events
        prev = ctx.metadata.get(f"_zone_{self._trigger_id}_prev", 0)
        if self._trigger_type == "zone_enter" and current_count > prev:
            self._count += current_count - prev
        ctx.metadata[f"_zone_{self._trigger_id}_prev"] = current_count
