import json
import logging
import time
from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext
from engine.core.node_registry import register

_log = logging.getLogger(__name__)

try:
    from kafka import KafkaProducer
    _KAFKA_AVAILABLE = True
except ImportError:
    _KAFKA_AVAILABLE = False


@register("kafka_produce")
class KafkaProduceNode(BaseNode):
    _producer = None
    _last_produce: float = 0.0
    _prev_counts: dict

    def initialize(self):
        self._prev_counts = {}
        if not _KAFKA_AVAILABLE:
            _log.warning(
                "kafka-python not installed — kafka_produce is a no-op. "
                "Install with: pip install kafka-python"
            )
            return
        try:
            servers = self.config.get("bootstrap_servers", "localhost:9092")
            self._producer = KafkaProducer(
                bootstrap_servers=[s.strip() for s in servers.split(",")],
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
        except Exception as exc:
            _log.warning("Kafka connect failed: %s", exc)
            self._producer = None

    def process(self, ctx: FrameContext) -> FrameContext:
        if not self._producer:
            return ctx
        trigger_on = self.config.get("trigger_on", "detection")
        rate_limit = float(self.config.get("rate_limit_s", 0.0))
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

        if should_pub and (now - self._last_produce) >= rate_limit:
            self._producer.send(self.config.get("topic", "cv_flow_events"), {
                "frame_number": getattr(ctx, "frame_number", 0),
                "timestamp": getattr(ctx, "timestamp", now),
                "detection_count": len(dets),
                "counters": counts,
            })
            self._last_produce = now
        return ctx

    def teardown(self):
        if self._producer:
            self._producer.flush(timeout=5)
            self._producer.close()
