import type { NodeMeta, NodeType, AnyNodeConfig } from '@/types/nodes'

interface RegistryEntry {
  meta: NodeMeta
  defaultConfig: AnyNodeConfig
}

const registry: Record<NodeType, RegistryEntry> = {
  /* ── Input ─────────────────────────────────────────────────────────────── */
  camera: {
    meta: { type: 'camera', group: 'input', label: 'Camera', description: 'RTSP or USB camera source', color: 'bg-blue-900', icon: '📷' },
    defaultConfig: { source_type: 'usb', device_index: 0, fps_limit: 30 },
  },
  video_file: {
    meta: { type: 'video_file', group: 'input', label: 'Video File', description: 'Read frames from .mp4 file', color: 'bg-blue-900', icon: '🎞️' },
    defaultConfig: { file_path: '', loop: false },
  },
  image_directory: {
    meta: { type: 'image_directory', group: 'input', label: 'Image Dir', description: 'Read images from a folder', color: 'bg-blue-900', icon: '🗂️' },
    defaultConfig: { directory_path: '', pattern: '*.jpg', delay_ms: 100 },
  },

  /* ── Processing ────────────────────────────────────────────────────────── */
  preprocess: {
    meta: { type: 'preprocess', group: 'processing', label: 'Preprocess', description: 'Resize, crop, normalize', color: 'bg-green-900', icon: '⚙️' },
    defaultConfig: { normalize: 'none' },
  },
  model_inference: {
    meta: { type: 'model_inference', group: 'processing', label: 'Model Inference', description: 'Run ONNX model', color: 'bg-green-900', icon: '🧠' },
    defaultConfig: { model_id: '', device: 'cpu', conf_threshold: 0.5 },
  },
  postprocess_nms: {
    meta: { type: 'postprocess_nms', group: 'processing', label: 'NMS Filter', description: 'Non-max suppression', color: 'bg-green-900', icon: '🔲' },
    defaultConfig: { iou_threshold: 0.45, conf_threshold: 0.25, max_detections: 300 },
  },

  /* ── Spatial Logic ─────────────────────────────────────────────────────── */
  draw_roi: {
    meta: { type: 'draw_roi', group: 'spatial', label: 'ROI Zone', description: 'Polygon zone filter', color: 'bg-orange-900', icon: '⬡' },
    defaultConfig: { zone_id: 'zone_1', polygon: [[10, 10], [90, 10], [90, 90], [10, 90]], draw_on_frame: true, filter_outside: true },
  },
  draw_line: {
    meta: { type: 'draw_line', group: 'spatial', label: 'Trip Line', description: 'Directional crossing line', color: 'bg-orange-900', icon: '↕' },
    defaultConfig: { line_id: 'line_1', line: [[10, 50], [90, 50]], direction: 'both' },
  },
  object_tracker: {
    meta: { type: 'object_tracker', group: 'spatial', label: 'Tracker', description: 'Persistent object IDs (ByteTrack)', color: 'bg-orange-900', icon: '🎯' },
    defaultConfig: { algorithm: 'bytetrack', max_age: 30, min_hits: 3, iou_threshold: 0.3 },
  },
  counter: {
    meta: { type: 'counter', group: 'spatial', label: 'Counter', description: 'Count zone/line events', color: 'bg-orange-900', icon: '🔢' },
    defaultConfig: { trigger_type: 'line_cross', trigger_id: 'line_1', reset_on_start: true },
  },

  /* ── Utility ───────────────────────────────────────────────────────────── */
  python_function: {
    meta: { type: 'python_function', group: 'utility', label: 'Python Fn', description: 'Custom Python processing', color: 'bg-purple-900', icon: '🐍' },
    defaultConfig: { code: 'def process(frame, detections, params):\n    # frame: numpy BGR array\n    # detections: list of Detection objects\n    return frame, detections\n' },
  },
  filter: {
    meta: { type: 'filter', group: 'utility', label: 'Filter', description: 'Filter by class or confidence', color: 'bg-purple-900', icon: '🔍' },
    defaultConfig: { allowed_classes: [], min_confidence: 0.0 },
  },
  param: {
    meta: { type: 'param', group: 'utility', label: 'Param', description: 'Inject parameters into pipeline', color: 'bg-purple-900', icon: '🔧' },
    defaultConfig: { params: {} },
  },

  /* ── C++ ───────────────────────────────────────────────────────────────── */
  cpp_function: {
    meta: { type: 'cpp_function', group: 'cpp', label: 'C++ Node', description: 'High-performance C++ processing', color: 'bg-cyan-900', icon: '⚡' },
    defaultConfig: {
      source_code: '#include <cv_flow/helpers.hpp>\n#include <opencv2/imgproc.hpp>\n\nextern "C" {\n\nvoid cv_flow_setup(const char* config_json) {}\n\nvoid cv_flow_process(CVFlowCtx* ctx) {\n    auto frame = cvflow::frame_mat(ctx);\n    // Modify frame in-place here\n}\n\nvoid cv_flow_teardown(void) {}\n\nconst char* cv_flow_version(void) { return "1.0.0"; }\n\n} /* extern "C" */\n',
      compile_status: 'uncompiled',
      compile_flags: ['-O2', '-march=native'],
    },
  },

  /* ── Output ────────────────────────────────────────────────────────────── */
  stream_viewer: {
    meta: { type: 'stream_viewer', group: 'output', label: 'Stream Viewer', description: 'Live preview in browser', color: 'bg-red-900', icon: '📺' },
    defaultConfig: { jpeg_quality: 80, max_fps: 30 },
  },
  video_writer: {
    meta: { type: 'video_writer', group: 'output', label: 'Video Writer', description: 'Save output to .mp4', color: 'bg-red-900', icon: '💾' },
    defaultConfig: { output_path: './output.mp4', codec: 'mp4v', fps: 30 },
  },
  trigger_webhook: {
    meta: { type: 'trigger_webhook', group: 'output', label: 'Webhook', description: 'HTTP/MQTT event trigger', color: 'bg-red-900', icon: '🔔' },
    defaultConfig: { protocol: 'http', trigger_on: 'count_change', rate_limit_s: 2.0 },
  },
}

export function getNodeMeta(type: NodeType): NodeMeta {
  return registry[type].meta
}

export function getDefaultConfig(type: NodeType): AnyNodeConfig {
  /* Return a deep clone so each node gets independent state */
  return JSON.parse(JSON.stringify(registry[type].defaultConfig))
}

export function getAllNodeMetas(): NodeMeta[] {
  return Object.values(registry).map((e) => e.meta)
}

export function getNodesByGroup(group: NodeMeta['group']): NodeMeta[] {
  return getAllNodeMetas().filter((m) => m.group === group)
}

export default registry
