"""
cv_flow.nodes.output — StreamViewer, VideoWriter, MqttPublisher (sink nodes).
"""
from __future__ import annotations

import json
import time

from cv_flow.node import Node


class VideoWriter(Node):
    """Writes incoming bgr8 frames to an MP4/AVI file via cv2.VideoWriter."""

    def __init__(
        self,
        input_topic,
        *,
        output_path: str,
        fps: float = 30.0,
        codec: str = "mp4v",
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self._input_topic = input_topic
        self.output_path = output_path
        self.fps    = fps
        self.codec  = codec
        self._writer = None

    def initialize(self) -> None:
        self._sub = self.subscribe(self._input_topic)

    def spin_once(self) -> None:
        import cv2
        frame = self._sub.read(timeout_ms=30)
        if frame is None:
            return
        if self._writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self._writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (w, h))
        self._writer.write(frame)

    def shutdown(self) -> None:
        if self._writer is not None:
            self._writer.release()


class StreamViewer(Node):
    """
    JPEG-encodes frames and broadcasts them over a WebSocket server.

    Requires the `stream` extra (websockets). Frames are encoded
    synchronously in spin_once(); the WebSocket server runs in a
    background thread started in initialize().
    """

    def __init__(
        self,
        input_topic,
        *,
        port: int = 8765,
        quality: int = 80,
        max_fps: int = 30,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self._input_topic = input_topic
        self.port    = port
        self.quality = quality
        self.max_fps = max_fps
        self._clients: set = set()
        self._loop = None
        self._server_thread = None
        self._last_emit = 0.0

    def initialize(self) -> None:
        import asyncio
        import threading
        import websockets

        self._sub = self.subscribe(self._input_topic)
        self._loop = asyncio.new_event_loop()

        async def _handler(websocket):
            self._clients.add(websocket)
            try:
                await websocket.wait_closed()
            finally:
                self._clients.discard(websocket)

        async def _serve():
            async with websockets.serve(_handler, "0.0.0.0", self.port):
                await asyncio.Future()  # run forever

        def _run_loop():
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(_serve())

        self._server_thread = threading.Thread(target=_run_loop, daemon=True)
        self._server_thread.start()

    def spin_once(self) -> None:
        import asyncio
        import cv2

        period = 1.0 / self.max_fps if self.max_fps > 0 else 0.0
        if period > 0 and (time.monotonic() - self._last_emit) < period:
            return

        frame = self._sub.read(timeout_ms=30)
        if frame is None or not self._clients:
            return
        self._last_emit = time.monotonic()

        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            return
        payload = jpeg.tobytes()

        async def _broadcast():
            for ws in list(self._clients):
                try:
                    await ws.send(payload)
                except Exception:
                    self._clients.discard(ws)

        asyncio.run_coroutine_threadsafe(_broadcast(), self._loop)


class MqttPublisher(Node):
    """Serialises detections to JSON and publishes on an MQTT topic."""

    def __init__(
        self,
        input_topic,
        *,
        broker: str = "localhost",
        port: int = 1883,
        topic: str,
        qos: int = 0,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self._input_topic = input_topic
        self.broker = broker
        self.port   = port
        self.mqtt_topic = topic
        self.qos    = qos
        self._client = None

    def initialize(self) -> None:
        import paho.mqtt.client as mqtt
        self._sub = self.subscribe(self._input_topic)
        self._client = mqtt.Client()
        self._client.connect(self.broker, self.port)
        self._client.loop_start()

    def spin_once(self) -> None:
        dets = self._sub.read(timeout_ms=30)
        if dets is None:
            return
        payload = {
            k: (v.tolist() if hasattr(v, "tolist") else v)
            for k, v in dets.items()
        } if isinstance(dets, dict) else dets.tolist()
        self._client.publish(self.mqtt_topic, json.dumps(payload), qos=self.qos)

    def shutdown(self) -> None:
        if self._client is not None:
            self._client.loop_stop()
            self._client.disconnect()
