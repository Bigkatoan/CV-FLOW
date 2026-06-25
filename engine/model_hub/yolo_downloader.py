"""Curated model catalog — download & export to ONNX for CV-FLOW."""
from __future__ import annotations
import json
import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Class lists ───────────────────────────────────────────────────────────────

COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush",
]

COCO_KEYPOINTS = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle",
]

# ── Catalog ───────────────────────────────────────────────────────────────────
# key = ultralytics model filename without .pt
# Fields:
#   name, desc, category, task, version, size_mb, badge
#   input_shape, output_shapes, class_names / keypoint_names

MODEL_CATALOG: dict[str, dict] = {

    # ── Object Detection — YOLO11 (latest) ───────────────────────────────────
    "yolo11n": {
        "name": "YOLO11 Nano", "desc": "Fastest · 2.6M params · CPU friendly",
        "category": "Object Detection", "task": "detection",
        "version": "11.0", "size_mb": 5, "badge": "Latest",
        "input_shape": [1, 3, 640, 640], "output_shapes": [[1, 84, 8400]],
        "class_names": COCO_CLASSES,
    },
    "yolo11s": {
        "name": "YOLO11 Small", "desc": "Fast & accurate · 9.4M params",
        "category": "Object Detection", "task": "detection",
        "version": "11.0", "size_mb": 19, "badge": "Latest",
        "input_shape": [1, 3, 640, 640], "output_shapes": [[1, 84, 8400]],
        "class_names": COCO_CLASSES,
    },
    "yolo11m": {
        "name": "YOLO11 Medium", "desc": "Balanced · 20.1M params",
        "category": "Object Detection", "task": "detection",
        "version": "11.0", "size_mb": 39, "badge": "Latest",
        "input_shape": [1, 3, 640, 640], "output_shapes": [[1, 84, 8400]],
        "class_names": COCO_CLASSES,
    },
    "yolo11l": {
        "name": "YOLO11 Large", "desc": "High accuracy · 25.3M params",
        "category": "Object Detection", "task": "detection",
        "version": "11.0", "size_mb": 49, "badge": "Latest",
        "input_shape": [1, 3, 640, 640], "output_shapes": [[1, 84, 8400]],
        "class_names": COCO_CLASSES,
    },
    "yolo11x": {
        "name": "YOLO11 XLarge", "desc": "Best accuracy · 56.9M params · GPU recommended",
        "category": "Object Detection", "task": "detection",
        "version": "11.0", "size_mb": 109, "badge": "Latest",
        "input_shape": [1, 3, 640, 640], "output_shapes": [[1, 84, 8400]],
        "class_names": COCO_CLASSES,
    },

    # ── Object Detection — YOLOv8 ─────────────────────────────────────────────
    "yolov8n": {
        "name": "YOLOv8 Nano", "desc": "Fastest · 3.2M params · great on CPU",
        "category": "Object Detection", "task": "detection",
        "version": "8.0", "size_mb": 6,
        "input_shape": [1, 3, 640, 640], "output_shapes": [[1, 84, 8400]],
        "class_names": COCO_CLASSES,
    },
    "yolov8s": {
        "name": "YOLOv8 Small", "desc": "Good speed/accuracy balance · 11.2M params",
        "category": "Object Detection", "task": "detection",
        "version": "8.0", "size_mb": 22,
        "input_shape": [1, 3, 640, 640], "output_shapes": [[1, 84, 8400]],
        "class_names": COCO_CLASSES,
    },
    "yolov8m": {
        "name": "YOLOv8 Medium", "desc": "Balanced · 25.9M params",
        "category": "Object Detection", "task": "detection",
        "version": "8.0", "size_mb": 52,
        "input_shape": [1, 3, 640, 640], "output_shapes": [[1, 84, 8400]],
        "class_names": COCO_CLASSES,
    },
    "yolov8l": {
        "name": "YOLOv8 Large", "desc": "High accuracy · 43.7M params",
        "category": "Object Detection", "task": "detection",
        "version": "8.0", "size_mb": 87,
        "input_shape": [1, 3, 640, 640], "output_shapes": [[1, 84, 8400]],
        "class_names": COCO_CLASSES,
    },
    "yolov8x": {
        "name": "YOLOv8 XLarge", "desc": "Best accuracy · 68.2M params · GPU recommended",
        "category": "Object Detection", "task": "detection",
        "version": "8.0", "size_mb": 136,
        "input_shape": [1, 3, 640, 640], "output_shapes": [[1, 84, 8400]],
        "class_names": COCO_CLASSES,
    },

    # ── Object Detection — YOLOv9 ─────────────────────────────────────────────
    "yolov9c": {
        "name": "YOLOv9-C", "desc": "Programmable gradient info · compact variant",
        "category": "Object Detection", "task": "detection",
        "version": "9.0", "size_mb": 51,
        "input_shape": [1, 3, 640, 640], "output_shapes": [[1, 84, 8400]],
        "class_names": COCO_CLASSES,
    },
    "yolov9e": {
        "name": "YOLOv9-E", "desc": "Extended accuracy · 57.3M params",
        "category": "Object Detection", "task": "detection",
        "version": "9.0", "size_mb": 115,
        "input_shape": [1, 3, 640, 640], "output_shapes": [[1, 84, 8400]],
        "class_names": COCO_CLASSES,
    },

    # ── Object Detection — YOLOv10 ────────────────────────────────────────────
    "yolov10n": {
        "name": "YOLOv10 Nano", "desc": "NMS-free · real-time friendly · 2.3M params",
        "category": "Object Detection", "task": "detection",
        "version": "10.0", "size_mb": 5,
        "input_shape": [1, 3, 640, 640], "output_shapes": [[1, 300, 6]],
        "class_names": COCO_CLASSES,
    },
    "yolov10s": {
        "name": "YOLOv10 Small", "desc": "NMS-free · 7.2M params",
        "category": "Object Detection", "task": "detection",
        "version": "10.0", "size_mb": 15,
        "input_shape": [1, 3, 640, 640], "output_shapes": [[1, 300, 6]],
        "class_names": COCO_CLASSES,
    },
    "yolov10m": {
        "name": "YOLOv10 Medium", "desc": "NMS-free · balanced · 15.4M params",
        "category": "Object Detection", "task": "detection",
        "version": "10.0", "size_mb": 32,
        "input_shape": [1, 3, 640, 640], "output_shapes": [[1, 300, 6]],
        "class_names": COCO_CLASSES,
    },

    # ── Object Detection — RT-DETR ────────────────────────────────────────────
    "rtdetr-l": {
        "name": "RT-DETR Large", "desc": "Transformer-based · 32M params · high accuracy",
        "category": "Object Detection", "task": "detection",
        "version": "1.0", "size_mb": 83,
        "input_shape": [1, 3, 640, 640],
        "output_shapes": [[1, 300, 4], [1, 300, 80]],
        "class_names": COCO_CLASSES,
    },
    "rtdetr-x": {
        "name": "RT-DETR XLarge", "desc": "Transformer-based · 67M params · best accuracy",
        "category": "Object Detection", "task": "detection",
        "version": "1.0", "size_mb": 136,
        "input_shape": [1, 3, 640, 640],
        "output_shapes": [[1, 300, 4], [1, 300, 80]],
        "class_names": COCO_CLASSES,
    },

    # ── Instance Segmentation — YOLO11 ────────────────────────────────────────
    "yolo11n-seg": {
        "name": "YOLO11n Segment", "desc": "Pixel-level masks · fastest · 2.9M params",
        "category": "Segmentation", "task": "segmentation",
        "version": "11.0", "size_mb": 7, "badge": "Latest",
        "input_shape": [1, 3, 640, 640],
        "output_shapes": [[1, 116, 8400], [1, 32, 160, 160]],
        "class_names": COCO_CLASSES,
    },
    "yolo11s-seg": {
        "name": "YOLO11s Segment", "desc": "Balanced masks · 10.1M params",
        "category": "Segmentation", "task": "segmentation",
        "version": "11.0", "size_mb": 22, "badge": "Latest",
        "input_shape": [1, 3, 640, 640],
        "output_shapes": [[1, 116, 8400], [1, 32, 160, 160]],
        "class_names": COCO_CLASSES,
    },
    "yolo11m-seg": {
        "name": "YOLO11m Segment", "desc": "Accurate masks · 22.4M params",
        "category": "Segmentation", "task": "segmentation",
        "version": "11.0", "size_mb": 43, "badge": "Latest",
        "input_shape": [1, 3, 640, 640],
        "output_shapes": [[1, 116, 8400], [1, 32, 160, 160]],
        "class_names": COCO_CLASSES,
    },

    # ── Instance Segmentation — YOLOv8 ───────────────────────────────────────
    "yolov8n-seg": {
        "name": "YOLOv8n Segment", "desc": "Fast pixel masks · 3.4M params",
        "category": "Segmentation", "task": "segmentation",
        "version": "8.0", "size_mb": 7,
        "input_shape": [1, 3, 640, 640],
        "output_shapes": [[1, 116, 8400], [1, 32, 160, 160]],
        "class_names": COCO_CLASSES,
    },
    "yolov8s-seg": {
        "name": "YOLOv8s Segment", "desc": "Balanced · 11.8M params",
        "category": "Segmentation", "task": "segmentation",
        "version": "8.0", "size_mb": 24,
        "input_shape": [1, 3, 640, 640],
        "output_shapes": [[1, 116, 8400], [1, 32, 160, 160]],
        "class_names": COCO_CLASSES,
    },
    "yolov8m-seg": {
        "name": "YOLOv8m Segment", "desc": "Accurate masks · 27.3M params",
        "category": "Segmentation", "task": "segmentation",
        "version": "8.0", "size_mb": 53,
        "input_shape": [1, 3, 640, 640],
        "output_shapes": [[1, 116, 8400], [1, 32, 160, 160]],
        "class_names": COCO_CLASSES,
    },

    # ── Pose Estimation — YOLO11 ─────────────────────────────────────────────
    "yolo11n-pose": {
        "name": "YOLO11n Pose", "desc": "17 COCO keypoints · fastest · 2.9M params",
        "category": "Pose Estimation", "task": "pose",
        "version": "11.0", "size_mb": 7, "badge": "Latest",
        "input_shape": [1, 3, 640, 640], "output_shapes": [[1, 56, 8400]],
        "keypoint_names": COCO_KEYPOINTS, "class_names": ["person"],
    },
    "yolo11s-pose": {
        "name": "YOLO11s Pose", "desc": "17 COCO keypoints · 9.9M params",
        "category": "Pose Estimation", "task": "pose",
        "version": "11.0", "size_mb": 23, "badge": "Latest",
        "input_shape": [1, 3, 640, 640], "output_shapes": [[1, 56, 8400]],
        "keypoint_names": COCO_KEYPOINTS, "class_names": ["person"],
    },
    "yolo11m-pose": {
        "name": "YOLO11m Pose", "desc": "17 COCO keypoints · 20.9M params",
        "category": "Pose Estimation", "task": "pose",
        "version": "11.0", "size_mb": 43, "badge": "Latest",
        "input_shape": [1, 3, 640, 640], "output_shapes": [[1, 56, 8400]],
        "keypoint_names": COCO_KEYPOINTS, "class_names": ["person"],
    },

    # ── Pose Estimation — YOLOv8 ─────────────────────────────────────────────
    "yolov8n-pose": {
        "name": "YOLOv8n Pose", "desc": "17 COCO keypoints · fastest · 3.3M params",
        "category": "Pose Estimation", "task": "pose",
        "version": "8.0", "size_mb": 7,
        "input_shape": [1, 3, 640, 640], "output_shapes": [[1, 56, 8400]],
        "keypoint_names": COCO_KEYPOINTS, "class_names": ["person"],
    },
    "yolov8s-pose": {
        "name": "YOLOv8s Pose", "desc": "17 COCO keypoints · 11.6M params",
        "category": "Pose Estimation", "task": "pose",
        "version": "8.0", "size_mb": 23,
        "input_shape": [1, 3, 640, 640], "output_shapes": [[1, 56, 8400]],
        "keypoint_names": COCO_KEYPOINTS, "class_names": ["person"],
    },
    "yolov8m-pose": {
        "name": "YOLOv8m Pose", "desc": "17 COCO keypoints · 26.4M params",
        "category": "Pose Estimation", "task": "pose",
        "version": "8.0", "size_mb": 53,
        "input_shape": [1, 3, 640, 640], "output_shapes": [[1, 56, 8400]],
        "keypoint_names": COCO_KEYPOINTS, "class_names": ["person"],
    },

    # ── Image Classification — YOLO11 ─────────────────────────────────────────
    "yolo11n-cls": {
        "name": "YOLO11n Classify", "desc": "ImageNet 1000 classes · fastest · 1.6M params",
        "category": "Classification", "task": "classification",
        "version": "11.0", "size_mb": 4, "badge": "Latest",
        "input_shape": [1, 3, 224, 224], "output_shapes": [[1, 1000]],
        "class_names": [f"class_{i}" for i in range(1000)],
    },
    "yolo11s-cls": {
        "name": "YOLO11s Classify", "desc": "ImageNet 1000 classes · 5.5M params",
        "category": "Classification", "task": "classification",
        "version": "11.0", "size_mb": 19, "badge": "Latest",
        "input_shape": [1, 3, 224, 224], "output_shapes": [[1, 1000]],
        "class_names": [f"class_{i}" for i in range(1000)],
    },
    "yolo11m-cls": {
        "name": "YOLO11m Classify", "desc": "ImageNet 1000 classes · 10.4M params",
        "category": "Classification", "task": "classification",
        "version": "11.0", "size_mb": 40, "badge": "Latest",
        "input_shape": [1, 3, 224, 224], "output_shapes": [[1, 1000]],
        "class_names": [f"class_{i}" for i in range(1000)],
    },

    # ── Image Classification — YOLOv8 ─────────────────────────────────────────
    "yolov8n-cls": {
        "name": "YOLOv8n Classify", "desc": "ImageNet 1000 classes · fastest · 2.7M params",
        "category": "Classification", "task": "classification",
        "version": "8.0", "size_mb": 5,
        "input_shape": [1, 3, 224, 224], "output_shapes": [[1, 1000]],
        "class_names": [f"class_{i}" for i in range(1000)],
    },
    "yolov8s-cls": {
        "name": "YOLOv8s Classify", "desc": "ImageNet 1000 classes · 6.4M params",
        "category": "Classification", "task": "classification",
        "version": "8.0", "size_mb": 19,
        "input_shape": [1, 3, 224, 224], "output_shapes": [[1, 1000]],
        "class_names": [f"class_{i}" for i in range(1000)],
    },
    "yolov8m-cls": {
        "name": "YOLOv8m Classify", "desc": "ImageNet 1000 classes · 17M params",
        "category": "Classification", "task": "classification",
        "version": "8.0", "size_mb": 43,
        "input_shape": [1, 3, 224, 224], "output_shapes": [[1, 1000]],
        "class_names": [f"class_{i}" for i in range(1000)],
    },
}

# Keep the old name for backward compat with existing API
YOLO_MODELS = MODEL_CATALOG


def download_yolo_model(model_key: str, models_dir: Path) -> dict:
    """
    Download a model via ultralytics, export to ONNX, and save into models_dir.
    Returns the full config dict (including model_id).
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError(
            "ultralytics is not installed.\n"
            "Run:  pip install ultralytics\n"
            "then click Download again."
        )

    if model_key not in MODEL_CATALOG:
        raise ValueError(
            f"Unknown model: {model_key!r}. "
            f"Available: {list(MODEL_CATALOG)}"
        )

    meta = MODEL_CATALOG[model_key]
    logger.info("Downloading + exporting %s to ONNX (may take a minute)…", model_key)

    # Classification models use 224px, all others use 640px
    imgsz = 224 if meta["task"] == "classification" else 640

    with tempfile.TemporaryDirectory() as tmpdir:
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            model = YOLO(f"{model_key}.pt")
            export_result = model.export(
                format="onnx",
                imgsz=imgsz,
                simplify=True,
                dynamic=False,
                opset=12,
            )
        finally:
            os.chdir(old_cwd)

        onnx_src = Path(export_result) if export_result else None
        if not onnx_src or not onnx_src.exists():
            candidates = list(Path(tmpdir).rglob("*.onnx"))
            if not candidates:
                raise RuntimeError(
                    f"ONNX export completed but no .onnx file found in {tmpdir}"
                )
            onnx_src = candidates[0]

        model_id = str(uuid.uuid4())
        dest_dir = models_dir / model_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(onnx_src), str(dest_dir / "model.onnx"))

        config: dict = {
            "model_id":     model_id,
            "name":         meta["name"],
            "version":      meta["version"],
            "task":         meta["task"],
            "format":       "onnx",
            "input_name":   "images",
            "input_shape":  meta["input_shape"],
            "output_shapes": meta["output_shapes"],
            "class_names":  meta.get("class_names", []),
        }
        if "keypoint_names" in meta:
            config["keypoint_names"] = meta["keypoint_names"]

        (dest_dir / "config.json").write_text(json.dumps(config, indent=2))
        logger.info("Saved %s as model_id=%s", model_key, model_id)
        return config
