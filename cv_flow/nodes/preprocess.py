"""
cv_flow.nodes.preprocess — Preprocess (letterbox resize + normalize) and
GrayscaleConvert.
"""
from __future__ import annotations

import numpy as np

from cv_flow.node import Node

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def letterbox_resize(
    frame: np.ndarray,
    target_w: int,
    target_h: int,
    *,
    keep_aspect: bool = True,
) -> np.ndarray:
    """Resize a BGR frame to (target_h, target_w, 3), padding if keep_aspect."""
    import cv2

    h, w = frame.shape[:2]
    if not keep_aspect:
        return cv2.resize(frame, (target_w, target_h))

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(frame, (new_w, new_h))

    canvas = np.zeros((target_h, target_w, 3), dtype=frame.dtype)
    top  = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def normalize_chw(
    frame_bgr: np.ndarray,
    *,
    normalize: str = "imagenet",
) -> np.ndarray:
    """Convert a (H, W, 3) BGR uint8 frame into a (1, 3, H, W) float32 tensor."""
    rgb = frame_bgr[:, :, ::-1].astype(np.float32)

    if normalize == "imagenet":
        rgb = rgb / 255.0
        rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    elif normalize == "[0,1]":
        rgb = rgb / 255.0
    # "none" → leave raw pixel values

    chw = np.transpose(rgb, (2, 0, 1))  # HWC -> CHW
    return np.expand_dims(chw, axis=0).astype(np.float32)  # add batch dim


class Preprocess(Node):
    """
    Subscribes to a BGR frame topic, publishes a letterboxed + normalized
    CHW float32 tensor topic.
    """

    def __init__(
        self,
        input_topic,
        output_topic,
        *,
        width: int = 640,
        height: int = 640,
        normalize: str = "imagenet",
        keep_aspect: bool = True,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self._input_topic  = input_topic
        self._output_topic = output_topic
        self.width       = width
        self.height      = height
        self.normalize   = normalize
        self.keep_aspect = keep_aspect

    def initialize(self) -> None:
        self._sub = self.subscribe(self._input_topic)
        self._pub = self.advertise(self._output_topic)

    def spin_once(self) -> None:
        frame = self._sub.read(timeout_ms=30)
        if frame is None:
            return
        resized = letterbox_resize(frame, self.width, self.height,
                                   keep_aspect=self.keep_aspect)
        tensor = normalize_chw(resized, normalize=self.normalize)
        self._pub.write(tensor)


class GrayscaleConvert(Node):
    """Subscribes to a BGR frame topic, publishes a mono8 frame topic."""

    def __init__(self, input_topic, output_topic, *, name: str | None = None) -> None:
        super().__init__(name=name)
        self._input_topic  = input_topic
        self._output_topic = output_topic

    def initialize(self) -> None:
        self._sub = self.subscribe(self._input_topic)
        self._pub = self.advertise(self._output_topic)

    def spin_once(self) -> None:
        import cv2
        frame = self._sub.read(timeout_ms=30)
        if frame is None:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._pub.write(gray)
