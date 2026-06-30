"""
cv_flow.nodes.tracking — ObjectTracker: greedy IOU-based multi-object tracker.

Lightweight ByteTrack-style tracker: matches new detections to existing
tracks by IOU, ages out unmatched tracks after max_age frames, and only
confirms a track after min_hits consecutive matches.
"""
from __future__ import annotations

import numpy as np

from cv_flow.node import Node


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class _Track:
    __slots__ = ("track_id", "box", "score", "class_id", "age", "hits", "time_since_update")

    def __init__(self, track_id: int, box, score, class_id):
        self.track_id = track_id
        self.box      = box
        self.score    = score
        self.class_id = class_id
        self.age      = 0
        self.hits     = 1
        self.time_since_update = 0


class ByteTrackLite:
    """
    Pure-Python greedy IOU tracker (no external dependency).

    Call update(boxes, scores, class_ids) once per frame; returns
    (boxes, scores, class_ids, track_ids) for confirmed tracks only.
    """

    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age       = max_age
        self.min_hits      = min_hits
        self.iou_threshold = iou_threshold
        self._tracks: list[_Track] = []
        self._next_id = 1

    def update(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Filter out padding sentinels (class_id == -1)
        valid = class_ids >= 0
        boxes, scores, class_ids = boxes[valid], scores[valid], class_ids[valid]

        matched_track_idx: set[int] = set()
        matched_det_idx:   set[int] = set()

        # Greedy matching by best IOU
        pairs = []
        for ti, track in enumerate(self._tracks):
            for di in range(len(boxes)):
                iou = _iou(track.box, boxes[di])
                if iou >= self.iou_threshold:
                    pairs.append((iou, ti, di))
        pairs.sort(key=lambda p: -p[0])

        for iou, ti, di in pairs:
            if ti in matched_track_idx or di in matched_det_idx:
                continue
            matched_track_idx.add(ti)
            matched_det_idx.add(di)
            track = self._tracks[ti]
            track.box, track.score, track.class_id = boxes[di], scores[di], class_ids[di]
            track.hits += 1
            track.time_since_update = 0

        # Age unmatched tracks
        for ti, track in enumerate(self._tracks):
            if ti not in matched_track_idx:
                track.time_since_update += 1
            track.age += 1

        # Drop stale tracks
        self._tracks = [t for t in self._tracks if t.time_since_update <= self.max_age]

        # Create new tracks for unmatched detections
        for di in range(len(boxes)):
            if di not in matched_det_idx:
                self._tracks.append(_Track(self._next_id, boxes[di], scores[di], class_ids[di]))
                self._next_id += 1

        confirmed = [t for t in self._tracks if t.hits >= self.min_hits]
        n = len(confirmed)
        out_boxes  = np.zeros((n, 4), dtype=np.float32)
        out_scores = np.zeros((n,),   dtype=np.float32)
        out_cls    = np.zeros((n,),   dtype=np.int32)
        out_ids    = np.zeros((n,),   dtype=np.int32)
        for i, t in enumerate(confirmed):
            out_boxes[i]  = t.box
            out_scores[i] = t.score
            out_cls[i]    = t.class_id
            out_ids[i]    = t.track_id

        return out_boxes, out_scores, out_cls, out_ids


class ObjectTracker(Node):
    """
    Subscribes to a detections topic, publishes tracked detections
    (with persistent track_id) to an output topic.
    """

    def __init__(
        self,
        input_topic,
        output_topic,
        *,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        max_tracks: int = 512,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self._input_topic  = input_topic
        self._output_topic = output_topic
        self.max_tracks = max_tracks
        self._tracker = ByteTrackLite(max_age=max_age, min_hits=min_hits,
                                      iou_threshold=iou_threshold)

    def initialize(self) -> None:
        self._sub = self.subscribe(self._input_topic)
        self._pub = self.advertise(self._output_topic)

    def spin_once(self) -> None:
        dets = self._sub.read(timeout_ms=30)
        if dets is None:
            return
        boxes, scores, class_ids, track_ids = self._tracker.update(
            dets["boxes"], dets["scores"], dets["class_ids"]
        )

        n = min(len(boxes), self.max_tracks)
        out_boxes  = np.zeros((self.max_tracks, 4), dtype=np.float32)
        out_scores = np.zeros((self.max_tracks,),   dtype=np.float32)
        out_cls    = np.full((self.max_tracks,), -1, dtype=np.int32)
        out_ids    = np.zeros((self.max_tracks,),   dtype=np.int32)
        out_boxes[:n]  = boxes[:n]
        out_scores[:n] = scores[:n]
        out_cls[:n]    = class_ids[:n]
        out_ids[:n]    = track_ids[:n]

        self._pub.write({
            "boxes": out_boxes, "scores": out_scores,
            "class_ids": out_cls, "track_ids": out_ids,
        })
