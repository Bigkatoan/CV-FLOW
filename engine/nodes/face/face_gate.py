"""Face Gate node — outputs a True/False access decision per detected face.

Connects after face_vector_db. Emits WS events and updates ctx.metadata
so downstream nodes (trigger_webhook, mqtt_publish) can act on the result.
"""
from __future__ import annotations
import logging
import time

from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext
from engine.core.node_registry import register
from engine.streaming import ws_server

logger = logging.getLogger(__name__)


@register("face_gate")
class FaceGateNode(BaseNode):
    """Evaluate face identity matches against an allow/block list."""

    def initialize(self):
        self._mode        = self.config.get("mode", "any_known")
        self._allowlist   = set(self.config.get("allowlist", []))
        self._blocklist   = set(self.config.get("blocklist", []))
        self._min_sim     = float(self.config.get("min_similarity", 0.5))
        self._rate_limit  = float(self.config.get("event_rate_limit_s", 1.0))
        self._draw_result = bool(self.config.get("draw_result", True))
        self._last_event: dict[str, float] = {}   # identity → last event timestamp

        import cv2
        self._cv2 = cv2
        logger.info("[FaceGate] mode=%s allowlist=%s blocklist=%s",
                    self._mode, self._allowlist, self._blocklist)

    def process(self, ctx: FrameContext) -> FrameContext:
        matches = ctx.metadata.get("face_matches", [])
        results = []
        now = time.time()

        for i, match in enumerate(matches):
            name    = match.get("name", "Unknown")
            sim     = match.get("similarity", 0.0)
            in_db   = match.get("in_db", False)
            allowed, reason = self._evaluate(name, sim, in_db)

            result = {"allowed": allowed, "identity": name,
                      "similarity": sim, "reason": reason}
            results.append(result)

            # Rate-limited WS event per identity
            if now - self._last_event.get(name, 0) >= self._rate_limit:
                self._last_event[name] = now
                ws_server.send_event(ctx.session_id, {
                    "type":      "face_gate_event",
                    "allowed":   allowed,
                    "identity":  name,
                    "similarity": sim,
                    "reason":    reason,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })

            # Draw result badge on frame
            if self._draw_result and ctx.frame is not None and i < len(ctx.detections):
                self._draw_badge(ctx.frame, ctx.detections[i], name, allowed, sim)

        ctx.metadata["face_gate_results"] = results

        # Single-face convenience key
        if results:
            ctx.metadata["face_gate_result"] = results[0]
            ctx.metadata["face_gate_allowed"] = results[0]["allowed"]
        return ctx

    def _evaluate(self, name: str, sim: float, in_db: bool) -> tuple[bool, str]:
        if sim < self._min_sim:
            return False, "low_confidence"
        if self._mode == "any_known":
            return (True, "known") if in_db else (False, "unknown")
        if self._mode == "whitelist":
            return (True, "in_allowlist") if name in self._allowlist else (False, "not_in_allowlist")
        if self._mode == "blacklist":
            return (False, "in_blocklist") if name in self._blocklist else (True, "not_blocked")
        return False, "unknown_mode"

    def _draw_badge(self, frame, det, name, allowed, sim):
        import cv2
        x1, y2 = int(det.x1), int(det.y2)
        color   = (56, 200, 56) if allowed else (56, 56, 230)
        icon    = "✓" if allowed else "✗"
        label   = f"{icon} {name} ({sim:.2f})"
        font    = cv2.FONT_HERSHEY_SIMPLEX
        scale, thick = 0.55, 2
        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
        cv2.rectangle(frame, (x1, y2), (x1 + tw + 8, y2 + th + 10), color, cv2.FILLED)
        cv2.putText(frame, label, (x1 + 4, y2 + th + 4), font, scale,
                    (255, 255, 255), thick, cv2.LINE_AA)
