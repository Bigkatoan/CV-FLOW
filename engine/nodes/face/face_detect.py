"""Face Detection node — SCRFD or any ONNX face detector with 5-point landmarks.

Default model: SCRFD-10G (auto-downloaded via insightface on first run).
Also accepts a manually uploaded ONNX via model_id in the Model Hub.
"""
from __future__ import annotations
import logging
import os
from pathlib import Path

import cv2
import numpy as np

from engine.nodes.base import BaseNode
from engine.core.frame_context import FrameContext, Detection
from engine.core.node_registry import register

logger = logging.getLogger(__name__)

# Default insightface model names per model_key
_INSIGHTFACE_NAMES = {
    "scrfd_10g":   "det_10g",
    "scrfd_500m":  "det_500m",
}


@register("face_detect")
class FaceDetectNode(BaseNode):
    """Detect faces and extract 5-point landmarks using SCRFD or compatible ONNX."""

    DEFAULT_MODEL_KEY = "scrfd_10g"

    def initialize(self):
        self._conf           = float(self.config.get("conf_threshold", 0.5))
        self._nms            = float(self.config.get("nms_threshold", 0.4))
        self._min_px         = int(self.config.get("min_face_size_px", 20))
        self._return_largest = bool(self.config.get("return_largest", False))
        self._model_key      = self.config.get("model_key", self.DEFAULT_MODEL_KEY)
        model_id             = self.config.get("model_id", "")

        self._det = None

        if model_id:
            self._load_custom_onnx(model_id)
        else:
            self._load_insightface()

    # ── Loaders ───────────────────────────────────────────────────────────────

    def _load_insightface(self):
        pack = "buffalo_l" if self._model_key == "scrfd_10g" else "buffalo_s"
        try:
            from insightface.app import FaceAnalysis
            providers = self._ort_providers()
            logger.info("[FaceDetect] Loading via FaceAnalysis pack=%s …", pack)
            app = FaceAnalysis(
                name=pack,
                allowed_modules=["detection"],
                providers=providers,
            )
            app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=self._conf)
            self._det = app
            logger.info("[FaceDetect] %s ready (det_thresh=%.2f)", pack, self._conf)
        except ImportError:
            logger.error(
                "[FaceDetect] insightface not installed. "
                "Run: pip install insightface onnxruntime"
            )
        except Exception as e:
            logger.error("[FaceDetect] Failed to load insightface model: %s", e, exc_info=True)

    def _load_custom_onnx(self, model_id: str):
        """Load a manually uploaded ONNX via model_id (YOLOv8-face compatible format)."""
        import onnxruntime as ort
        models_dir = Path(os.environ.get("CVFLOW_MODELS_DIR", "storage/models"))
        model_path = models_dir / model_id / "model.onnx"
        if not model_path.exists():
            logger.error("[FaceDetect] model.onnx not found at %s", model_path)
            return
        providers = self._ort_providers()
        self._session = ort.InferenceSession(str(model_path), providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        logger.info("[FaceDetect] custom ONNX loaded from %s", model_path)
        self._det = None  # use self._session path

    def _ort_providers(self):
        device = self.config.get("device", "cpu")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" \
               else ["CPUExecutionProvider"]

    # ── Process ───────────────────────────────────────────────────────────────

    def process(self, ctx: FrameContext) -> FrameContext:
        # Active gate — reads dynamic return_largest override from metadata
        if ctx.metadata.get("face_detect_active") is False:
            ctx.detections = []
            ctx.metadata["face_landmarks"]    = []
            ctx.metadata["face_bboxes"]       = []
            ctx.metadata["face_has_detection"] = False
            return ctx

        if ctx.frame is None:
            return ctx

        # return_largest may be overridden per-frame via metadata
        self._return_largest = bool(
            ctx.metadata.get("face_return_largest", self.config.get("return_largest", False))
        )

        frame = ctx.ensure_cpu()

        if self._det is not None:
            self._run_insightface(ctx, frame)
        elif hasattr(self, "_session"):
            self._run_custom_onnx(ctx, frame)
        else:
            logger.warning("[FaceDetect] No model loaded — skipping frame.")

        return ctx

    def _run_insightface(self, ctx: FrameContext, frame: np.ndarray):
        try:
            faces = self._det.get(frame)   # list of Face objects
        except Exception as e:
            logger.error("[FaceDetect] detect() failed: %s", e)
            return

        detections, landmarks = [], []
        for face in faces:
            x1, y1, x2, y2 = face.bbox
            w, h = x2 - x1, y2 - y1
            if w < self._min_px or h < self._min_px:
                continue
            detections.append(Detection(
                x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                confidence=float(face.det_score), class_id=0, class_name="face",
            ))
            kps = face.kps
            landmarks.append(kps.tolist() if kps is not None else None)   # [[x,y]×5]

        if self._return_largest and len(detections) > 1:
            areas = [(d.x2 - d.x1) * (d.y2 - d.y1) for d in detections]
            idx   = int(np.argmax(areas))
            detections = [detections[idx]]
            landmarks  = [landmarks[idx]]

        ctx.detections = detections
        ctx.metadata["face_landmarks"]    = landmarks
        ctx.metadata["face_bboxes"]       = [[d.x1, d.y1, d.x2, d.y2, d.confidence] for d in detections]
        ctx.metadata["face_has_detection"] = len(detections) > 0
        logger.debug("[FaceDetect] %d face(s) detected", len(detections))

    def _run_custom_onnx(self, ctx: FrameContext, frame: np.ndarray):
        """YOLOv8-face compatible: output [1, N_anchors, 20] (4bbox+1score+15lm)."""
        h, w = frame.shape[:2]
        inp = cv2.resize(frame, (640, 640))
        inp = inp.astype(np.float32) / 255.0
        inp = inp.transpose(2, 0, 1)[np.newaxis]

        out = self._session.run(None, {self._input_name: inp})[0][0]  # [N_anchors, 20]
        detections, landmarks = [], []

        for row in out:
            score = float(row[4])
            if score < self._conf:
                continue
            cx, cy, bw, bh = row[:4]
            x1 = (cx - bw / 2) / 640 * w
            y1 = (cy - bh / 2) / 640 * h
            x2 = (cx + bw / 2) / 640 * w
            y2 = (cy + bh / 2) / 640 * h
            if (x2 - x1) < self._min_px or (y2 - y1) < self._min_px:
                continue
            detections.append(Detection(
                x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                confidence=score, class_id=0, class_name="face",
            ))
            lm = []
            for k in range(5):
                lx = row[5 + k * 3] / 640 * w
                ly = row[5 + k * 3 + 1] / 640 * h
                lm.append([float(lx), float(ly)])
            landmarks.append(lm)

        # NMS
        if detections:
            boxes = [[d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1] for d in detections]
            scores = [d.confidence for d in detections]
            keep = cv2.dnn.NMSBoxes(boxes, scores, self._conf, self._nms)
            keep = [k[0] if isinstance(k, (list, np.ndarray)) else k for k in keep]
            ctx.detections = [detections[k] for k in keep]
            ctx.metadata["face_landmarks"] = [landmarks[k] for k in keep]
        else:
            ctx.detections = []
            ctx.metadata["face_landmarks"] = []

        if self._return_largest and len(ctx.detections) > 1:
            areas = [(d.x2 - d.x1) * (d.y2 - d.y1) for d in ctx.detections]
            idx   = int(np.argmax(areas))
            ctx.detections               = [ctx.detections[idx]]
            ctx.metadata["face_landmarks"] = [ctx.metadata["face_landmarks"][idx]]

        ctx.metadata["face_bboxes"]       = [[d.x1, d.y1, d.x2, d.y2, d.confidence] for d in ctx.detections]
        ctx.metadata["face_has_detection"] = len(ctx.detections) > 0
