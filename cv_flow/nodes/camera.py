"""
cv_flow.nodes.camera — CameraSource, RtspSource, VideoFileSource.

All three wrap cv2.VideoCapture and publish bgr8 frames + a monotonic seq.
"""
from __future__ import annotations

import time

import numpy as np

from cv_flow.node import Node


class _CaptureSourceBase(Node):
    """Shared cv2.VideoCapture lifecycle for camera/RTSP/file sources."""

    def __init__(self, output_topic, *, name: str | None = None) -> None:
        super().__init__(name=name)
        self._output_topic = output_topic
        self._cap = None
        self._seq = 0

    def _open_capture(self):
        raise NotImplementedError

    def initialize(self) -> None:
        self._pub = self.advertise(self._output_topic)
        self._cap = self._open_capture()

    def spin_once(self) -> None:
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._on_read_failure()
            return
        self._seq += 1

        field_names = [f.name for f in self._pub._port_def.fields]
        if len(field_names) == 1:
            data = frame
        else:
            data = {n: (frame if n == "frame" else np.uint64(self._seq)) for n in field_names}
        self._pub.write(data, seq=self._seq)

    def _on_read_failure(self) -> None:
        raise StopIteration

    def shutdown(self) -> None:
        if self._cap is not None:
            self._cap.release()


class CameraSource(_CaptureSourceBase):
    """Reads frames from a USB/V4L2 camera device."""

    def __init__(
        self,
        output_topic,
        *,
        device_index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        name: str | None = None,
    ) -> None:
        super().__init__(output_topic, name=name)
        self.device_index = device_index
        self.width  = width
        self.height = height
        self.fps    = fps

    def _open_capture(self):
        import cv2
        cap = cv2.VideoCapture(self.device_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        return cap

    def _on_read_failure(self) -> None:
        raise StopIteration  # camera disconnected


class RtspSource(_CaptureSourceBase):
    """Reads an RTSP stream with automatic reconnect."""

    def __init__(
        self,
        output_topic,
        *,
        url: str,
        reconnect_delay_s: float = 2.0,
        name: str | None = None,
    ) -> None:
        super().__init__(output_topic, name=name)
        self.url = url
        self.reconnect_delay_s = reconnect_delay_s

    def _open_capture(self):
        import cv2
        return cv2.VideoCapture(self.url)

    def _on_read_failure(self) -> None:
        # Attempt reconnect rather than stopping the pipeline.
        if self._cap is not None:
            self._cap.release()
        time.sleep(self.reconnect_delay_s)
        self._cap = self._open_capture()


class VideoFileSource(_CaptureSourceBase):
    """Reads frames from a video file, optionally looping at EOF."""

    def __init__(
        self,
        output_topic,
        *,
        path: str,
        loop: bool = False,
        fps: float = 0.0,
        name: str | None = None,
    ) -> None:
        super().__init__(output_topic, name=name)
        self.path = path
        self.loop = loop
        self.fps  = fps
        self._frame_period: float | None = None
        self._last_emit = 0.0

    def _open_capture(self):
        import cv2
        cap = cv2.VideoCapture(self.path)
        native_fps = self.fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._frame_period = 1.0 / native_fps if native_fps > 0 else None
        return cap

    def _on_read_failure(self) -> None:
        if self.loop:
            import cv2
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            raise StopIteration

    def spin_once(self) -> None:
        if self._frame_period is not None:
            elapsed = time.monotonic() - self._last_emit
            remaining = self._frame_period - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last_emit = time.monotonic()
        super().spin_once()
