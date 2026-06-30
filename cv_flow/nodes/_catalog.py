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
            "Reads frames from a USB/V4L2 camera (device_index) or a Jetson CSI "
            "camera (gstreamer_pipeline, e.g. via build_nvargus_pipeline()). "
            "Publishes BGR frame + timestamp + monotonic seq number."
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
             "description": "USB/V4L2 camera index (0 = first camera). Ignored if "
                            "gstreamer_pipeline is set."},
            {"name": "width",        "type": "int",   "default": 1280},
            {"name": "height",       "type": "int",   "default": 720},
            {"name": "fps",          "type": "int",   "default": 30},
            {"name": "gstreamer_pipeline", "type": "str", "default": "",
             "description": (
                 "GStreamer pipeline string for CSI cameras (Jetson nvarguscamerasrc). "
                 "When set, takes priority over device_index/width/height/fps. "
                 "Requires an OpenCV build with GStreamer support — the JetPack "
                 "apt-installed OpenCV has it, the generic PyPI opencv-python wheel "
                 "does not. See build_nvargus_pipeline() helper."
             )},
        ],
        "example": "CameraSource(device_index=0, width=1280, height=720, fps=30)",
    },

    "RtspSource": {
        "category":    "input",
        "description": (
            "Reads an RTSP video stream with automatic reconnect on failure "
            "(exponential backoff, capped at max_reconnect_delay_s, resets on success)."
        ),
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
             "description": "Initial seconds to wait before reconnecting after failure"},
            {"name": "max_reconnect_delay_s", "type": "float", "default": 30.0,
             "description": "Cap for the doubling backoff delay between reconnect attempts"},
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

    "Tee": {
        "category":    "processing",
        "description": (
            "Fans a single topic out to N independent output topics, republishing "
            "every frame unchanged. Required whenever more than one downstream node "
            "needs the SAME upstream data — e.g. a detect-and-draw pipeline where "
            "both Preprocess (for inference) and DrawBbox (for the final overlay) "
            "need the original camera frame: every topic/bus in this DAM model is a "
            "single-reader FIFO, so two nodes subscribing to the same topic directly "
            "would compete for the same queue instead of each seeing every frame."
        ),
        "runtime":      "python",
        "elastic_capable": False,
        "inputs":  [{"slot": "in", "dtype": "any", "description": "Any single-field topic"}],
        "outputs": [{"slot": "out", "dtype": "any",
                     "description": "Same data republished to each output topic"}],
        "parameters": [
            {"name": "output_topics", "type": "list", "required": True,
             "description": "List of topic names to republish the input onto"},
        ],
        "example": 'Tee("camera_frame", ["camera_frame_infer", "camera_frame_draw"])',
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
             "description": (
                 "Inference device. device=\"cuda:0\" tries ONNX Runtime providers in order "
                 "TensorrtExecutionProvider -> CUDAExecutionProvider -> CPUExecutionProvider "
                 "(first one available on the host is used); device=\"cpu\" forces "
                 "CPUExecutionProvider only."
             )},
            {"name": "trt_cache_dir", "type": "str", "default": "",
             "description": (
                 "If set, enables TensorRT engine disk caching at this path so the (slow, "
                 "~1-2 min) engine build only happens once instead of on every process start."
             )},
        ],
        "example": 'YoloInference(model_path="models/yolov8n.onnx", device="cuda:0", '
                   'trt_cache_dir=".trt_cache")',
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
             "choices": ["cpu", "cuda:0"],
             "description": (
                 "device=\"cuda:0\" tries TensorrtExecutionProvider -> CUDAExecutionProvider -> "
                 "CPUExecutionProvider in order; device=\"cpu\" forces CPU only."
             )},
            {"name": "trt_cache_dir", "type": "str", "default": "",
             "description": "If set, enables TensorRT engine disk caching at this path."},
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
            {"name": "output_layout",        "type": "str",   "default": "features_first",
             "choices": ["features_first", "boxes_first", "auto"],
             "description": (
                 "Only used when format=\"yolov8\". \"features_first\" (default) matches "
                 "the standard YOLOv8 ONNX export shape (1, 84, N). \"auto\" falls back to "
                 "a shape-comparison heuristic that breaks when box count < 84 — avoid it "
                 "unless you know your model's raw output layout varies."
             )},
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
