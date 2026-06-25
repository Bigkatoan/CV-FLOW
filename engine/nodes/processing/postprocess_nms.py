import numpy as np
import cv2
from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext, Detection
from engine.core.node_registry import register


@register("postprocess_nms")
@register("nms")
class PostprocessNMSNode(BaseNode):
    def process(self, ctx: FrameContext) -> FrameContext:
        outputs = ctx.metadata.get("model_output")
        model_config = ctx.metadata.get("model_config", {})
        if outputs is None or ctx.frame is None:
            return ctx

        task = model_config.get("task", "detection")
        if task != "detection":
            return ctx

        class_names  = model_config.get("class_names", [])
        iou_thr      = self.config.get("iou_threshold", 0.45)
        conf_thr     = self.config.get("conf_threshold", 0.25)
        max_det      = self.config.get("max_detections", 300)
        frame_h, frame_w = ctx.frame.shape[:2]

        # Input dimensions the model was trained/exported with
        input_shape  = model_config.get("input_shape", [1, 3, 640, 640])
        in_h, in_w   = input_shape[-2], input_shape[-1]

        # Detect output format from output_shapes + actual tensor shapes
        fmt = self._detect_format(outputs, model_config)

        ctx.detections = self._parse(
            fmt, outputs, class_names,
            iou_thr, conf_thr, max_det,
            frame_w, frame_h, in_w, in_h,
        )
        return ctx

    # ── Format detection ──────────────────────────────────────────────────────

    @staticmethod
    def _detect_format(outputs, model_config) -> str:
        output_shapes = model_config.get("output_shapes", [])

        # RT-DETR: 2 separate output tensors (boxes + scores)
        if len(outputs) >= 2 and len(output_shapes) >= 2:
            return "rtdetr"

        raw = outputs[0]
        if raw.ndim == 3:
            raw = raw[0]

        # YOLOv10: [N, 6] — already-decoded [x1,y1,x2,y2,score,cls]
        if raw.ndim == 2 and raw.shape[-1] == 6:
            return "yolov10"

        # YOLOv8/v9/v11: [84, 8400] (transposed after squeeze)
        return "yolov8"

    # ── Unified dispatcher ────────────────────────────────────────────────────

    def _parse(self, fmt, outputs, class_names, iou_thr, conf_thr, max_det,
               fw, fh, in_w, in_h) -> list[Detection]:
        if fmt == "yolov10":
            return self._parse_yolov10(outputs, class_names, conf_thr, max_det, fw, fh, in_w, in_h)
        if fmt == "rtdetr":
            return self._parse_rtdetr(outputs, class_names, conf_thr, max_det, fw, fh)
        return self._parse_yolov8(outputs, class_names, iou_thr, conf_thr, max_det, fw, fh, in_w, in_h)

    # ── YOLOv8 / v9 / v11: [1, 4+num_cls, num_anchors] ──────────────────────

    def _parse_yolov8(self, outputs, class_names, iou_thr, conf_thr, max_det,
                      fw, fh, in_w, in_h) -> list[Detection]:
        raw = outputs[0]
        if raw.ndim == 3:
            raw = raw[0]        # [4+C, N]
        raw = raw.T             # [N, 4+C]

        boxes  = raw[:, :4]    # cx, cy, w, h  (in input-resolution pixel space)
        scores = raw[:, 4:]    # class scores

        cls_ids   = np.argmax(scores, axis=1)
        cls_confs = scores[np.arange(len(scores)), cls_ids]
        mask      = cls_confs >= conf_thr

        if not mask.any():
            return []

        boxes     = boxes[mask]
        cls_ids   = cls_ids[mask]
        cls_confs = cls_confs[mask]

        # cx,cy,w,h → x1,y1,w,h  (scale from input size to frame size)
        sx, sy    = fw / in_w, fh / in_h
        x1 = (boxes[:, 0] - boxes[:, 2] / 2) * sx
        y1 = (boxes[:, 1] - boxes[:, 3] / 2) * sy
        bw = boxes[:, 2] * sx
        bh = boxes[:, 3] * sy
        cv2_boxes = np.stack([x1, y1, bw, bh], axis=1).tolist()

        indices = cv2.dnn.NMSBoxes(cv2_boxes, cls_confs.tolist(), conf_thr, iou_thr)
        if isinstance(indices, np.ndarray):
            indices = indices.flatten().tolist()
        indices = indices[:max_det]

        dets = []
        for i in indices:
            bx, by, bw2, bh2 = cv2_boxes[i]
            dets.append(Detection(
                x1=max(0.0, float(bx)),
                y1=max(0.0, float(by)),
                x2=min(float(fw), float(bx + bw2)),
                y2=min(float(fh), float(by + bh2)),
                confidence=float(cls_confs[i]),
                class_id=int(cls_ids[i]),
                class_name=(class_names[cls_ids[i]] if cls_ids[i] < len(class_names)
                             else str(cls_ids[i])),
            ))
        return dets

    # ── YOLOv10: [1, max_det, 6]  [x1,y1,x2,y2,score,cls]  already decoded ──

    def _parse_yolov10(self, outputs, class_names, conf_thr, max_det,
                       fw, fh, in_w, in_h) -> list[Detection]:
        raw = outputs[0]
        if raw.ndim == 3:
            raw = raw[0]          # [max_det, 6]

        sx, sy = fw / in_w, fh / in_h
        dets = []
        for row in raw[:max_det]:
            x1, y1, x2, y2, score, cls_id = row
            if score < conf_thr:
                continue
            cls_id = int(cls_id)
            dets.append(Detection(
                x1=max(0.0, float(x1) * sx),
                y1=max(0.0, float(y1) * sy),
                x2=min(float(fw), float(x2) * sx),
                y2=min(float(fh), float(y2) * sy),
                confidence=float(score),
                class_id=cls_id,
                class_name=(class_names[cls_id] if cls_id < len(class_names)
                             else str(cls_id)),
            ))
        return dets

    # ── RT-DETR: boxes [1,300,4] cx/cy/w/h norm + scores [1,300,num_cls] ─────

    def _parse_rtdetr(self, outputs, class_names, conf_thr, max_det,
                      fw, fh) -> list[Detection]:
        boxes_raw  = outputs[0][0]   # [300, 4]  normalized cx,cy,w,h
        scores_raw = outputs[1][0]   # [300, C]

        # Scores may already be probabilities from ONNX simplification
        scores = scores_raw
        if scores_raw.min() < 0 or scores_raw.max() > 1:
            scores = 1.0 / (1.0 + np.exp(-np.clip(scores_raw, -50, 50)))

        cls_ids   = np.argmax(scores, axis=1)
        cls_confs = scores[np.arange(len(scores)), cls_ids]
        mask      = cls_confs >= conf_thr

        dets = []
        for i in np.where(mask)[0][:max_det]:
            cx, cy, w, h = boxes_raw[i]
            x1 = (cx - w / 2) * fw
            y1 = (cy - h / 2) * fh
            x2 = (cx + w / 2) * fw
            y2 = (cy + h / 2) * fh
            cls_id = int(cls_ids[i])
            dets.append(Detection(
                x1=max(0.0, x1), y1=max(0.0, y1),
                x2=min(float(fw), x2), y2=min(float(fh), y2),
                confidence=float(cls_confs[i]),
                class_id=cls_id,
                class_name=(class_names[cls_id] if cls_id < len(class_names)
                             else str(cls_id)),
            ))
        return dets
