"""
cv_flow.nodes.draw — DrawBbox: render bounding boxes + labels onto a BGR frame.
"""
from __future__ import annotations

import numpy as np

from cv_flow.node import Node


def draw_boxes(
    frame: np.ndarray,
    boxes: np.ndarray,
    *,
    scores: np.ndarray | None = None,
    class_ids: np.ndarray | None = None,
    label_names: dict[int, str] | None = None,
    thickness: int = 2,
    show_label: bool = True,
    show_confidence: bool = True,
) -> np.ndarray:
    """
    Draw bounding boxes onto a copy of `frame`. Boxes with class_id == -1
    (the NMS padding sentinel) are skipped.

    Parameters
    ----------
    frame     : (H, W, 3) BGR uint8 array.
    boxes     : (N, 4) xyxy float array.
    scores    : optional (N,) confidence scores.
    class_ids : optional (N,) class indices. -1 = no detection (padding).
    """
    import cv2

    out = frame.copy()
    n = len(boxes)
    for i in range(n):
        cls = int(class_ids[i]) if class_ids is not None else 0
        if class_ids is not None and cls < 0:
            continue

        x1, y1, x2, y2 = (int(v) for v in boxes[i])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), thickness)

        if show_label:
            label = label_names.get(cls, str(cls)) if label_names else str(cls)
            if show_confidence and scores is not None:
                label = f"{label} {scores[i]:.2f}"
            cv2.putText(
                out, label, (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
            )

    return out


class DrawBbox(Node):
    """
    Subscribes to a frame topic + a detections topic, publishes an
    annotated frame topic.
    """

    def __init__(
        self,
        frame_topic,
        dets_topic,
        output_topic,
        *,
        thickness: int = 2,
        show_label: bool = True,
        show_confidence: bool = True,
        label_map: dict[int, str] | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self._frame_topic  = frame_topic
        self._dets_topic   = dets_topic
        self._output_topic = output_topic
        self.thickness        = thickness
        self.show_label        = show_label
        self.show_confidence   = show_confidence
        self.label_map          = label_map or {}

    def initialize(self) -> None:
        self._frame_sub = self.subscribe(self._frame_topic)
        self._dets_sub  = self.subscribe(self._dets_topic)
        self._pub       = self.advertise(self._output_topic)

    def spin_once(self) -> None:
        frame = self._frame_sub.read(timeout_ms=30)
        if frame is None:
            return
        dets = self._dets_sub.read(timeout_ms=30)
        if dets is None:
            self._pub.write(frame)
            return

        boxes     = dets.get("boxes")
        scores    = dets.get("scores")
        class_ids = dets.get("class_ids")

        annotated = draw_boxes(
            frame, boxes,
            scores=scores, class_ids=class_ids,
            label_names=self.label_map,
            thickness=self.thickness,
            show_label=self.show_label,
            show_confidence=self.show_confidence,
        )
        self._pub.write(annotated)
