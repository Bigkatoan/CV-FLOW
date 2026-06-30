"""
cv_flow.nodes._catalog — NODE_CATALOG: metadata for the visual pipeline editor.

Each entry describes a built-in node's category, description, inputs, outputs,
and configurable parameters. This catalog is consumed by the backend API to
power the drag-and-drop visual editor and AI deployment guide generator.
"""
from __future__ import annotations

NODE_CATALOG: dict[str, dict] = {

    # ── Input nodes ───────────────────────────────────────────────────────────

    "CameraSource": {
        "category":    "input",
        "description": (
            "Reads frames from a USB/V4L2 camera. Publishes BGR frame + "
            "timestamp + monotonic seq number on the output topic."
        ),
        "runtime":      "python",
        "elastic_capable": False,
        "inputs":  [],
        "outputs": [
            {
                "slot": "frame_out",
                "dtype": "bgr8",
                "description": "Raw BGR frame from camera",
                "required_shape_keys": ["height", "width"],
            }
        ],
        "parameters": [
            {"name": "device_index", "type": "int",   "default": 0,
             "description": "Camera index (0 = first camera)"},
            {"name": "width",        "type": "int",   "default": 1280},
            {"name": "height",       "type": "int",   "default": 720},
            {"name": "fps",          "type": "int",   "default": 30},
        ],
        "example": "CameraSource(device_index=0, width=1280, height=720, fps=30)",
    },

    "RtspSource": {
        "category":    "input",
        "description": "Reads an RTSP video stream with automatic reconnect on failure.",
        "runtime":      "python",
        "elastic_capable": False,
        "inputs":  [],
        "outputs": [
            {"slot": "frame_out", "dtype": "bgr8",
             "description": "Decoded BGR frame from RTSP stream"}
        ],
        "parameters": [
            {"name": "url",              "type": "str",   "required": True,
             "description": "RTSP URL, e.g. rtsp://192.168.1.1:554/stream"},
            {"name": "reconnect_delay_s", "type": "float", "default": 2.0,
             "description": "Seconds to wait before reconnecting after failure"},
        ],
        "example": 'RtspSource(url="rtsp://192.168.1.1:554/stream")',
    },

    "VideoFileSource": {
        "category":    "input",
        "description": "Reads frames from a video file (MP4, AVI, etc.) and loops if requested.",
        "runtime":      "python",
        "elastic_capable": False,
        "inputs":  [],
        "outputs": [
            {"slot": "frame_out", "dtype": "bgr8", "description": "Decoded BGR frame"}
        ],
        "parameters": [
            {"name": "path",   "type": "str",  "required": True,
             "description": "Path to the video file"},
            {"name": "loop",   "type": "bool", "default": False,
             "description": "Restart from beginning when EOF is reached"},
            {"name": "fps",    "type": "float", "default": 0.0,
             "description": "Playback rate override (0 = use file's native fps)"},
        ],
    },

    # ── Processing nodes ──────────────────────────────────────────────────────

    "Preprocess": {
        "category":    "processing",
        "description": (
            "Letterbox-resizes and normalises a BGR frame into a CHW float32 "
            "tensor suitable for YOLO, ResNet, etc."
        ),
        "runtime":      "python",
        "elastic_capable": True,
        "inputs":  [{"slot": "frame_in", "dtype": "bgr8",
                     "description": "Input BGR frame"}],
        "outputs": [{"slot": "tensor_out", "dtype": "float32",
                     "shape_note": "(1, 3, H, W) CHW format, normalised"}],
        "parameters": [
            {"name": "width",      "type": "int",  "default": 640},
            {"name": "height",     "type": "int",  "default": 640},
            {"name": "normalize",  "type": "str",  "default": "imagenet",
             "choices": ["imagenet", "[0,1]", "none"],
             "description": "Normalisation scheme"},
            {"name": "keep_aspect", "type": "bool", "default": True,
             "description": "True = letterbox padding, False = stretch to fit"},
        ],
    },

    "GrayscaleConvert": {
        "category":    "processing",
        "description": "Converts a BGR frame to grayscale (mono8).",
        "runtime":      "python",
        "elastic_capable": False,
        "inputs":  [{"slot": "frame_in",  "dtype": "bgr8"}],
        "outputs": [{"slot": "frame_out", "dtype": "mono8"}],
        "parameters": [],
    },

    # ── Inference nodes ───────────────────────────────────────────────────────

    "YoloInference": {
        "category":    "inference",
        "description": (
            "Runs a YOLO model (ONNX format) on a float32 CHW tensor. "
            "Elastic-capable: can be scaled to N workers via RoundRobinBus."
        ),
        "runtime":      "python",
        "elastic_capable": True,
        "inputs":  [{"slot": "tensor_in", "dtype": "float32",
                     "shape_note": "(1, 3, H, W) — output of Preprocess"}],
        "outputs": [{"slot": "raw_out",   "dtype": "float32",
                     "shape_note": "(1, 84, 8400) raw YOLO v8 output"}],
        "parameters": [
            {"name": "model_path", "type": "str", "required": True,
             "description": "Path to the ONNX model file"},
            {"name": "device",     "type": "str", "default": "cpu",
             "choices": ["cpu", "cuda:0"],
             "description": "Inference device"},
            {"name": "providers",  "type": "list", "default": ["CPUExecutionProvider"],
             "description": "ONNX Runtime execution providers (ordered by priority)"},
        ],
        "example": 'YoloInference(model_path="models/yolov8n.onnx", device="cpu")',
    },

    "OnnxInference": {
        "category":    "inference",
        "description": "Generic ONNX model runner. Input/output shapes are model-defined.",
        "runtime":      "python",
        "elastic_capable": True,
        "inputs":  [{"slot": "tensor_in", "dtype": "float32",
                     "description": "Input tensor — shape must match model's first input"}],
        "outputs": [{"slot": "tensor_out", "dtype": "float32",
                     "description": "Model output tensor"}],
        "parameters": [
            {"name": "model_path",    "type": "str", "required": True},
            {"name": "input_name",    "type": "str", "default": "images",
             "description": "ONNX input node name"},
            {"name": "output_name",   "type": "str", "default": "output0"},
            {"name": "device",        "type": "str", "default": "cpu",
             "choices": ["cpu", "cuda:0"]},
        ],
    },

    # ── Post-processing nodes ─────────────────────────────────────────────────

    "NMS": {
        "category":    "postprocess",
        "description": (
            "Non-Maximum Suppression. Converts raw YOLO output tensor "
            "into boxes, scores, and class_ids."
        ),
        "runtime":      "python",
        "elastic_capable": False,
        "inputs":  [{"slot": "raw_in",      "dtype": "float32",
                     "shape_note": "(1, 84, 8400) raw YOLO output"}],
        "outputs": [
            {"slot": "boxes_out",     "dtype": "float32",
             "shape_note": "(N, 4) xyxy bounding boxes"},
            {"slot": "scores_out",    "dtype": "float32",
             "shape_note": "(N,) confidence scores"},
            {"slot": "class_ids_out", "dtype": "int32",
             "shape_note": "(N,) class indices"},
        ],
        "parameters": [
            {"name": "confidence_threshold", "type": "float", "default": 0.4},
            {"name": "iou_threshold",        "type": "float", "default": 0.45},
            {"name": "max_detections",       "type": "int",   "default": 512},
            {"name": "format",               "type": "str",   "default": "yolov8",
             "choices": ["yolov8", "yolov5"]},
        ],
    },

    # ── Tracking nodes ────────────────────────────────────────────────────────

    "ObjectTracker": {
        "category":    "tracking",
        "description": (
            "ByteTrack tracker. Associates detections across frames, "
            "assigns persistent track_ids, and handles occlusion."
        ),
        "runtime":      "python",
        "elastic_capable": False,
        "inputs":  [{"slot": "dets_in", "dtype": "float32",
                     "shape_note": "(N, 6) — boxes(4) + score(1) + class_id(1)"}],
        "outputs": [{"slot": "tracked_out", "dtype": "float32",
                     "shape_note": "(M, 7) — boxes(4) + score(1) + class_id(1) + track_id(1)"}],
        "parameters": [
            {"name": "max_age",   "type": "int", "default": 30,
             "description": "Frames to keep a track alive with no matching detection"},
            {"name": "min_hits",  "type": "int", "default": 3,
             "description": "Detections required before confirming a track"},
            {"name": "iou_threshold", "type": "float", "default": 0.3},
        ],
    },

    # ── Visualisation nodes ───────────────────────────────────────────────────

    "DrawBbox": {
        "category":    "visualization",
        "description": "Draws bounding boxes + class labels onto a BGR frame.",
        "runtime":      "python",
        "elastic_capable": False,
        "inputs":  [
            {"slot": "frame_in", "dtype": "bgr8",    "description": "Input frame"},
            {"slot": "dets_in",  "dtype": "float32", "description": "Detections"},
        ],
        "outputs": [{"slot": "frame_out", "dtype": "bgr8",
                     "description": "Annotated frame"}],
        "parameters": [
            {"name": "thickness",         "type": "int",  "default": 2},
            {"name": "show_label",        "type": "bool", "default": True},
            {"name": "show_confidence",   "type": "bool", "default": True},
            {"name": "label_map",         "type": "str",  "default": "",
             "description": "Path to class-name text file (one name per line)"},
        ],
    },

    # ── Output / sink nodes ───────────────────────────────────────────────────

    "StreamViewer": {
        "category":    "output",
        "description": (
            "JPEG-encodes each frame and streams it over WebSocket. "
            "View live at ws://<host>:<port>."
        ),
        "runtime":      "python",
        "elastic_capable": False,
        "inputs":  [{"slot": "frame_in", "dtype": "bgr8"}],
        "outputs": [],
        "parameters": [
            {"name": "port",    "type": "int",  "default": 8765},
            {"name": "quality", "type": "int",  "default": 80,
             "description": "JPEG quality 1-100"},
            {"name": "max_fps", "type": "int",  "default": 30},
        ],
        "example": "StreamViewer(port=8765, quality=80)",
    },

    "VideoWriter": {
        "category":    "output",
        "description": "Writes frames to an MP4 file using OpenCV VideoWriter.",
        "runtime":      "python",
        "elastic_capable": False,
        "inputs":  [{"slot": "frame_in", "dtype": "bgr8"}],
        "outputs": [],
        "parameters": [
            {"name": "output_path", "type": "str",   "required": True,
             "description": "Path for the output video file"},
            {"name": "fps",         "type": "float", "default": 30.0},
            {"name": "codec",       "type": "str",   "default": "mp4v",
             "description": "FourCC codec code"},
        ],
    },

    "MqttPublisher": {
        "category":    "output",
        "description": "Serialises detection results to JSON and publishes on an MQTT topic.",
        "runtime":      "python",
        "elastic_capable": False,
        "inputs":  [{"slot": "dets_in", "dtype": "float32"}],
        "outputs": [],
        "parameters": [
            {"name": "broker",   "type": "str", "default": "localhost"},
            {"name": "port",     "type": "int", "default": 1883},
            {"name": "topic",    "type": "str", "required": True,
             "description": "MQTT topic to publish on"},
            {"name": "qos",      "type": "int", "default": 0,
             "choices": [0, 1, 2]},
        ],
    },

}

# ── Validation helpers ────────────────────────────────────────────────────────

_REQUIRED_NODE_FIELDS = {"category", "description", "inputs", "outputs", "parameters"}
_REQUIRED_PARAM_FIELDS = {"name", "type"}


def validate_catalog() -> list[str]:
    """
    Validate NODE_CATALOG schema. Returns a list of error strings (empty = OK).
    """
    errors: list[str] = []
    for node_type, meta in NODE_CATALOG.items():
        missing = _REQUIRED_NODE_FIELDS - meta.keys()
        if missing:
            errors.append(f"{node_type}: missing fields {missing}")
            continue

        for param in meta["parameters"]:
            pm = _REQUIRED_PARAM_FIELDS - param.keys()
            has_default  = "default" in param
            has_required = param.get("required", False)
            if pm:
                errors.append(f"{node_type}.parameters[{param.get('name','?')}]: missing {pm}")
            if not has_default and not has_required:
                errors.append(
                    f"{node_type}.parameters[{param.get('name','?')}]: "
                    "must have 'default' or 'required': True"
                )

        if meta.get("elastic_capable") and not meta["inputs"]:
            errors.append(
                f"{node_type}: elastic_capable=True but has no inputs"
            )

    return errors
