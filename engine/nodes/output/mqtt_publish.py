import json
import logging
import time
from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext
from engine.core.node_registry import register

_log = logging.getLogger(__name__)

try:
    import paho.mqtt.client as mqtt_client
    _MQTT_AVAILABLE = True
except ImportError:
    _MQTT_AVAILABLE = False


@register("mqtt_publish")
class MQTTPublishNode(BaseNode):
    _client = None
    _last_publish: float = 0.0
    _prev_counts: dict

    def initialize(self):
        self._prev_counts = {}
        if not _MQTT_AVAILABLE:
            _log.warning("paho-mqtt not installed — mqtt_publish is a no-op")
            return
        self._client = mqtt_client.Client()
        broker = self.config.get("broker", "localhost")
        port = int(self.config.get("port", 1883))
        try:
            self._client.connect(broker, port, keepalive=60)
            self._client.loop_start()
        except Exception as exc:
            _log.warning("MQTT connect failed: %s", exc)
            self._client = None

    def process(self, ctx: FrameContext) -> FrameContext:
        if not self._client:
            return ctx
        trigger_on = self.config.get("trigger_on", "detection")
        rate_limit = float(self.config.get("rate_limit_s", 0.5))
        now = time.time()
        dets = getattr(ctx, "detections", []) or []
        counts = dict(getattr(ctx, "counters", {}) or {})

        should_pub = False
        if trigger_on == "every_frame":
            should_pub = True
        elif trigger_on == "detection":
            should_pub = bool(dets)
        elif trigger_on == "count_change":
            should_pub = counts != self._prev_counts
            self._prev_counts = counts

        if should_pub and (now - self._last_publish) >= rate_limit:
            payload = json.dumps({
                "frame_number": getattr(ctx, "frame_number", 0),
                "timestamp": getattr(ctx, "timestamp", now),
                "detection_count": len(dets),
                "counters": counts,
            })
            self._client.publish(
                self.config.get("topic", "cv_flow/events"),
                payload,
                qos=int(self.config.get("qos", 0)),
            )
            self._last_publish = now
        return ctx

    def teardown(self):
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
