import cv2
import logging
from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext
from engine.core.node_registry import register

logger = logging.getLogger(__name__)


@register("counter")
class CounterNode(BaseNode):
    def initialize(self):
        self._trigger_type  = self.config.get("trigger_type", "line_cross")
        self._trigger_id    = self.config.get("trigger_id", "line_1")
        self._count_classes = set(self.config.get("count_classes", []))
        self._label         = self.config.get("label", "Count")
        self._show_overlay  = self.config.get("show_overlay", True)
        self._count = 0
        self._prev_sides: dict[int, float] = {}   # track_id → last cross-product sign
        logger.info(
            "[Counter] node %s — type=%s trigger=%s label=%s",
            self.node_id, self._trigger_type, self._trigger_id, self._label,
        )

    def process(self, ctx: FrameContext) -> FrameContext:
        if self._trigger_type == "line_cross":
            self._check_line_cross(ctx)
        elif self._trigger_type in ("zone_enter", "zone_exit"):
            self._check_zone(ctx)

        ctx.metadata[f"counter_{self.node_id}"] = self._count

        if self._show_overlay and ctx.frame is not None:
            self._draw_overlay(ctx)

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
            if prev is not None and prev * sign < 0:
                self._count += 1
                logger.debug(
                    "[Counter] %s crossed line %s — total=%d (track_id=%d)",
                    det.class_name, self._trigger_id, self._count, det.track_id,
                )
            self._prev_sides[det.track_id] = sign

    def _check_zone(self, ctx: FrameContext):
        zone_key = f"zone_{self._trigger_id}_count"
        current_count = ctx.metadata.get(zone_key, 0)
        prev = ctx.metadata.get(f"_zone_{self._trigger_id}_prev", 0)
        if self._trigger_type == "zone_enter" and current_count > prev:
            self._count += current_count - prev
        ctx.metadata[f"_zone_{self._trigger_id}_prev"] = current_count

    def _draw_overlay(self, ctx: FrameContext):
        frame = ctx.frame
        if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            ctx.frame = frame

        type_short = {
            "line_cross":  "crossed",
            "zone_enter":  "entered",
            "zone_exit":   "exited",
        }.get(self._trigger_type, self._trigger_type)

        text = f"{self._label} ({type_short} \"{self._trigger_id}\"): {self._count}"
        font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        x, y = 14, 14
        # dark fill + accent border
        cv2.rectangle(frame, (x - 6, y - 4), (x + tw + 6, y + th + baseline + 6), (0, 0, 0), cv2.FILLED)
        cv2.rectangle(frame, (x - 6, y - 4), (x + tw + 6, y + th + baseline + 6), (88, 166, 255), 1)
        cv2.putText(frame, text, (x, y + th), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
