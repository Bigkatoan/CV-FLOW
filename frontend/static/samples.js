// Pre-built sample pipelines
export const SAMPLES = [
  {
    name: "Camera Stream",
    description: "Webcam → Stream Viewer. Requires a connected camera.",
    nodes: [
      { id: "cam_1", type: "camera", position: { x: 80, y: 200 },
        data: { label: "Camera", config: { source_type: "usb", device_index: 0, fps_limit: 30 } } },
      { id: "stream_1", type: "stream_viewer", position: { x: 380, y: 200 },
        data: { label: "Stream Viewer", config: { jpeg_quality: 80, max_fps: 30 } } },
    ],
    edges: [
      { id: "e1", source: "cam_1", target: "stream_1", sourceHandle: "out", targetHandle: "in" },
    ],
  },
  {
    name: "Video File Stream",
    description: "Play a local video file and stream it. Edit the file_path config.",
    nodes: [
      { id: "vid_1", type: "video_file", position: { x: 80, y: 200 },
        data: { label: "Video File", config: { file_path: "C:/video.mp4", loop: true } } },
      { id: "stream_1", type: "stream_viewer", position: { x: 380, y: 200 },
        data: { label: "Stream Viewer", config: { jpeg_quality: 80, max_fps: 30 } } },
    ],
    edges: [
      { id: "e1", source: "vid_1", target: "stream_1", sourceHandle: "out", targetHandle: "in" },
    ],
  },
  {
    name: "Object Detection",
    description: "Camera → ONNX Model → NMS → Draw boxes → Stream. Requires a model uploaded.",
    nodes: [
      { id: "cam_1", type: "camera", position: { x: 60, y: 200 },
        data: { label: "Camera", config: { source_type: "usb", device_index: 0, fps_limit: 30 } } },
      { id: "pre_1", type: "preprocess", position: { x: 280, y: 200 },
        data: { label: "Preprocess", config: { normalize: "imagenet" } } },
      { id: "model_1", type: "model_inference", position: { x: 500, y: 200 },
        data: { label: "Model Inference", config: { model_id: "", device: "cpu", conf_threshold: 0.5 } } },
      { id: "nms_1", type: "nms", position: { x: 720, y: 200 },
        data: { label: "NMS Filter", config: { iou_threshold: 0.45, conf_threshold: 0.25, max_detections: 300 } } },
      { id: "stream_1", type: "stream_viewer", position: { x: 940, y: 200 },
        data: { label: "Stream Viewer", config: { jpeg_quality: 80, max_fps: 30 } } },
    ],
    edges: [
      { id: "e1", source: "cam_1",   target: "pre_1",    sourceHandle: "out", targetHandle: "in" },
      { id: "e2", source: "pre_1",   target: "model_1",  sourceHandle: "out", targetHandle: "in" },
      { id: "e3", source: "model_1", target: "nms_1",    sourceHandle: "out", targetHandle: "in" },
      { id: "e4", source: "nms_1",   target: "stream_1", sourceHandle: "out", targetHandle: "in" },
    ],
  },
  {
    name: "Line Counter",
    description: "Detect & track objects, count line crossings. Requires a model uploaded.",
    nodes: [
      { id: "cam_1",     type: "camera",         position: { x: 60,  y: 200 },
        data: { label: "Camera",          config: { source_type: "usb", device_index: 0, fps_limit: 30 } } },
      { id: "model_1",   type: "model_inference", position: { x: 280, y: 200 },
        data: { label: "Model Inference", config: { model_id: "", device: "cpu", conf_threshold: 0.5 } } },
      { id: "nms_1",     type: "nms",            position: { x: 500, y: 200 },
        data: { label: "NMS Filter",      config: { iou_threshold: 0.45, conf_threshold: 0.25, max_detections: 300 } } },
      { id: "tracker_1", type: "object_tracker", position: { x: 720, y: 200 },
        data: { label: "Tracker",         config: { algorithm: "bytetrack", max_age: 30, min_hits: 3 } } },
      { id: "line_1",    type: "draw_line",      position: { x: 500, y: 380 },
        data: { label: "Trip Line",       config: { line_id: "line_1", line: [[0,360],[1280,360]], direction: "both" } } },
      { id: "counter_1", type: "counter",        position: { x: 720, y: 380 },
        data: { label: "Counter",         config: { trigger_type: "line_cross", trigger_id: "line_1", reset_on_start: true } } },
      { id: "stream_1",  type: "stream_viewer",  position: { x: 940, y: 290 },
        data: { label: "Stream Viewer",   config: { jpeg_quality: 80, max_fps: 30 } } },
    ],
    edges: [
      { id: "e1", source: "cam_1",     target: "model_1",   sourceHandle: "out", targetHandle: "in" },
      { id: "e2", source: "model_1",   target: "nms_1",     sourceHandle: "out", targetHandle: "in" },
      { id: "e3", source: "nms_1",     target: "tracker_1", sourceHandle: "out", targetHandle: "in" },
      { id: "e4", source: "nms_1",     target: "line_1",    sourceHandle: "out", targetHandle: "in" },
      { id: "e5", source: "tracker_1", target: "stream_1",  sourceHandle: "out", targetHandle: "in" },
      { id: "e6", source: "line_1",    target: "counter_1", sourceHandle: "out", targetHandle: "in" },
    ],
  },
];
