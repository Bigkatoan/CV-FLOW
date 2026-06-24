import numpy as np
import cv2
from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext, Detection
from engine.core.node_registry import register


@register("postprocess_nms")
class PostprocessNMSNode(BaseNode):
    def process(self, ctx: FrameContext) -> FrameContext:
        outputs = ctx.metadata.get("model_output")
        model_config = ctx.metadata.get("model_config", {})
        if outputs is None:
            return ctx

        task = model_config.get("task", "detection")
        class_names = model_config.get("class_names", [])
        iou_thr  = self.config.get("iou_threshold", 0.45)
        conf_thr = self.config.get("conf_threshold", 0.25)
        max_det  = self.config.get("max_detections", 300)
        frame_h, frame_w = ctx.frame.shape[:2]

        if task == "detection":
            ctx.detections = self._parse_detection(
                outputs, class_names, iou_thr, conf_thr, max_det, frame_w, frame_h
            )
        # Future: segmentation, pose branches here

        return ctx

    def _parse_detection(self, outputs, class_names, iou_thr, conf_thr, max_det, fw, fh) -> list[Detection]:
        """Parse YOLOv8-style output [1, 4+num_classes, num_anchors]."""
        raw = outputs[0]  # shape: [1, 84, 8400]
        if raw.ndim == 3:
            raw = raw[0]  # → [84, 8400]
        raw = raw.T       # → [8400, 84]

        boxes  = raw[:, :4]    # cx, cy, w, h
        scores = raw[:, 4:]    # class scores

        cls_ids   = np.argmax(scores, axis=1)
        cls_confs = scores[np.arange(len(scores)), cls_ids]
        mask = cls_confs >= conf_thr
        boxes     = boxes[mask]
        cls_ids   = cls_ids[mask]
        cls_confs = cls_confs[mask]

        if len(boxes) == 0:
            return []

        # Convert cx, cy, w, h → x1, y1, w, h (cv2.dnn.NMSBoxes format)
        input_shape = 640  # assume square; refine from model_config if needed
        x1 = (boxes[:, 0] - boxes[:, 2] / 2) * fw / input_shape
        y1 = (boxes[:, 1] - boxes[:, 3] / 2) * fh / input_shape
        bw = boxes[:, 2] * fw / input_shape
        bh = boxes[:, 3] * fh / input_shape
        cv2_boxes = np.stack([x1, y1, bw, bh], axis=1).tolist()

        indices = cv2.dnn.NMSBoxes(cv2_boxes, cls_confs.tolist(), conf_thr, iou_thr)
        if isinstance(indices, np.ndarray):
            indices = indices.flatten().tolist()
        indices = indices[:max_det]

        detections = []
        for i in indices:
            x, y, w, h = cv2_boxes[i]
            detections.append(Detection(
                x1=max(0.0, float(x)),
                y1=max(0.0, float(y)),
                x2=min(float(fw), float(x + w)),
                y2=min(float(fh), float(y + h)),
                confidence=float(cls_confs[i]),
                class_id=int(cls_ids[i]),
                class_name=class_names[cls_ids[i]] if cls_ids[i] < len(class_names) else str(cls_ids[i]),
            ))
        return detections
