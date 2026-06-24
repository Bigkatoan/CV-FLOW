import json
import logging
import time
from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext
from engine.core.node_registry import register

logger = logging.getLogger(__name__)


@register("trigger_webhook")
class TriggerWebhookNode(BaseNode):
    def initialize(self):
        self._protocol  = self.config.get("protocol", "http")
        self._url       = self.config.get("url", "")
        self._trigger   = self.config.get("trigger_on", "count_change")
        self._rate      = self.config.get("rate_limit_s", 2.0)
        self._template  = self.config.get("payload_template", "")
        self._last_sent = 0.0
        self._last_val  = None

        # MQTT client
        if self._protocol == "mqtt":
            try:
                import paho.mqtt.client as mqtt
                self._mqtt = mqtt.Client()
                broker = self.config.get("mqtt_broker", "localhost")
                self._mqtt.connect(broker, 1883, 60)
                self._mqtt.loop_start()
                self._mqtt_topic = self.config.get("mqtt_topic", "cv/events")
            except Exception as e:
                logger.warning("MQTT init failed: %s", e)
                self._mqtt = None

    def process(self, ctx: FrameContext) -> FrameContext:
        now = time.monotonic()
        if now - self._last_sent < self._rate:
            return ctx

        should_trigger = False
        payload: dict = {"timestamp": ctx.timestamp, "frame": ctx.frame_number, "session": ctx.session_id}

        if self._trigger == "count_change":
            val = ctx.metadata.get(f"counter_{self.node_id}", 0)
            if val != self._last_val:
                should_trigger = True
                payload["count"] = val
                self._last_val = val

        elif self._trigger == "detection" and ctx.detections:
            should_trigger = True
            payload["detection_count"] = len(ctx.detections)
            payload["classes"] = [d.class_name for d in ctx.detections[:10]]

        if should_trigger:
            self._last_sent = now
            self._send(payload)
        return ctx

    def _send(self, payload: dict):
        try:
            if self._protocol == "http":
                import httpx
                httpx.post(self._url, json=payload, timeout=5)
            elif self._protocol == "mqtt" and getattr(self, "_mqtt", None):
                self._mqtt.publish(self._mqtt_topic, json.dumps(payload))
        except Exception as e:
            logger.warning("Webhook send error: %s", e)

    def teardown(self):
        if self._protocol == "mqtt" and getattr(self, "_mqtt", None):
            self._mqtt.loop_stop()
            self._mqtt.disconnect()
