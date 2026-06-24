from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext, Detection
from engine.core.node_registry import register


@register("object_tracker")
class ObjectTrackerNode(BaseNode):
    """
    Wraps DeepSORT or ByteTrack behind a common interface.
    Track IDs are written back into ctx.detections[i].track_id.

    Phase 4 implementation — tracker libraries must be installed:
      pip install deep-sort-realtime  (DeepSORT)
      pip install bytetracker         (ByteTrack)
    """

    def initialize(self):
        self._algo = self.config.get("algorithm", "bytetrack")
        self._tracker = None
        self._init_tracker()

    def _init_tracker(self):
        algo = self._algo
        try:
            if algo == "deepsort":
                from deep_sort_realtime.deepsort_tracker import DeepSort
                self._tracker = DeepSort(
                    max_age=self.config.get("max_age", 30),
                    iou_threshold=self.config.get("iou_threshold", 0.3),
                )
            else:  # bytetrack
                from bytetracker import BYTETracker

                class Args:
                    track_thresh = 0.5
                    track_buffer = self.config.get("max_age", 30)
                    match_thresh = self.config.get("iou_threshold", 0.3)

                self._tracker = BYTETracker(Args())
        except ImportError as e:
            import logging
            logging.getLogger(__name__).warning(
                "Tracker library not installed (%s). track_id will remain -1.", e
            )
            self._tracker = None

    def process(self, ctx: FrameContext) -> FrameContext:
        if not ctx.detections or self._tracker is None:
            return ctx

        if self._algo == "deepsort":
            self._update_deepsort(ctx)
        else:
            self._update_bytetrack(ctx)
        return ctx

    def _update_deepsort(self, ctx: FrameContext):
        import numpy as np
        raw = [[d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1, d.confidence] for d in ctx.detections]
        tracks = self._tracker.update_tracks(raw, frame=ctx.frame)
        id_map: dict[int, int] = {}
        for t in tracks:
            if t.is_confirmed():
                ltrb = t.to_ltrb()
                # Match to nearest detection by IOU
                best, best_iou = -1, 0.0
                for i, d in enumerate(ctx.detections):
                    iou = _iou(ltrb, [d.x1, d.y1, d.x2, d.y2])
                    if iou > best_iou:
                        best_iou, best = iou, i
                if best >= 0:
                    ctx.detections[best].track_id = t.track_id

    def _update_bytetrack(self, ctx: FrameContext):
        import numpy as np
        fh, fw = ctx.frame.shape[:2]
        dets = [[d.x1, d.y1, d.x2, d.y2, d.confidence, d.class_id] for d in ctx.detections]
        dets_np = np.array(dets, dtype=np.float32) if dets else np.zeros((0, 6), dtype=np.float32)
        tracks = self._tracker.update(dets_np, img_info=(fh, fw), img_size=(fh, fw))
        for t in tracks:
            tlwh = t.tlwh
            x1, y1 = tlwh[0], tlwh[1]
            x2, y2 = tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]
            best, best_iou = -1, 0.0
            for i, d in enumerate(ctx.detections):
                iou = _iou([x1, y1, x2, y2], [d.x1, d.y1, d.x2, d.y2])
                if iou > best_iou:
                    best_iou, best = iou, i
            if best >= 0:
                ctx.detections[best].track_id = t.track_id


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)
