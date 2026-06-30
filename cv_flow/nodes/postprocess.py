"""
cv_flow.nodes.postprocess — NMS: Non-Maximum Suppression for YOLO output.
"""
from __future__ import annotations

import numpy as np

from cv_flow.node import Node


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def _iou_matrix(boxes: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    xx1 = np.maximum(x1[:, None], x1[None, :])
    yy1 = np.maximum(y1[:, None], y1[None, :])
    xx2 = np.minimum(x2[:, None], x2[None, :])
    yy2 = np.minimum(y2[:, None], y2[None, :])

    inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
    union = areas[:, None] + areas[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def nms(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45,
) -> np.ndarray:
    """
    Greedy NMS. Returns indices of boxes to keep, sorted by score descending.
    """
    if len(boxes_xyxy) == 0:
        return np.array([], dtype=np.int64)

    order = np.argsort(-scores)
    iou = _iou_matrix(boxes_xyxy)

    keep: list[int] = []
    suppressed = np.zeros(len(boxes_xyxy), dtype=bool)
    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(int(idx))
        suppressed |= iou[idx] > iou_threshold
        suppressed[idx] = True  # keep idx itself counted, but already added

    return np.array(keep, dtype=np.int64)


def run_nms(
    raw: np.ndarray,
    *,
    confidence_threshold: float = 0.4,
    iou_threshold: float = 0.45,
    max_detections: int = 512,
    format: str = "yolov8",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run NMS over raw YOLO model output.

    Parameters
    ----------
    raw : np.ndarray, shape (1, 84, N) for yolov8 (4 box coords + 80 classes)
          or (1, N, 85) for yolov5 (4 box coords + 1 obj conf + 80 classes).

    Returns
    -------
    (boxes_xyxy, scores, class_ids) each padded/truncated to max_detections.
    """
    arr = np.asarray(raw)
    if arr.ndim == 3:
        arr = arr[0]

    if format == "yolov8":
        # arr shape: (84, N) -> transpose to (N, 84). Assumes N (box count) >
        # feature count (4+num_classes), true for any real YOLOv8 output.
        if arr.shape[0] < arr.shape[1]:
            arr = arr.T
        boxes_xywh = arr[:, :4]
        class_scores = arr[:, 4:]
        scores = class_scores.max(axis=1)
        class_ids = class_scores.argmax(axis=1)
    else:  # yolov5: (N, 85) = box(4) + obj_conf(1) + class_scores(80)
        boxes_xywh = arr[:, :4]
        obj_conf = arr[:, 4]
        class_scores = arr[:, 5:]
        scores = obj_conf * class_scores.max(axis=1)
        class_ids = class_scores.argmax(axis=1)

    mask = scores >= confidence_threshold
    boxes_xywh, scores, class_ids = boxes_xywh[mask], scores[mask], class_ids[mask]
    boxes_xyxy = _xywh_to_xyxy(boxes_xywh)

    keep = nms(boxes_xyxy, scores, iou_threshold)
    keep = keep[:max_detections]

    out_boxes  = np.zeros((max_detections, 4), dtype=np.float32)
    out_scores = np.zeros((max_detections,),   dtype=np.float32)
    out_cls    = np.full((max_detections,), -1, dtype=np.int32)

    n = len(keep)
    if n > 0:
        out_boxes[:n]  = boxes_xyxy[keep]
        out_scores[:n] = scores[keep]
        out_cls[:n]    = class_ids[keep]

    return out_boxes, out_scores, out_cls


class NMS(Node):
    """
    Non-Maximum Suppression node.

    Subscribes to a raw YOLO output topic, publishes filtered
    boxes/scores/class_ids to a detections topic.
    """

    def __init__(
        self,
        input_topic,
        output_topic,
        *,
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        max_detections: int = 512,
        format: str = "yolov8",
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self._input_topic  = input_topic
        self._output_topic = output_topic
        self.confidence_threshold = confidence_threshold
        self.iou_threshold        = iou_threshold
        self.max_detections       = max_detections
        self.format               = format

    def initialize(self) -> None:
        self._sub = self.subscribe(self._input_topic)
        self._pub = self.advertise(self._output_topic)

    def spin_once(self) -> None:
        result = self._sub.read(timeout_ms=30)
        if result is None:
            return
        boxes, scores, class_ids = run_nms(
            result,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
            max_detections=self.max_detections,
            format=self.format,
        )
        self._pub.write({
            "boxes": boxes, "scores": scores, "class_ids": class_ids,
        })
