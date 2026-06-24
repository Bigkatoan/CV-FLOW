/* All node type identifiers — must match pipeline_schema.json enum */
export type NodeType =
  | 'camera' | 'video_file' | 'image_directory'
  | 'preprocess' | 'model_inference' | 'postprocess_nms'
  | 'draw_roi' | 'draw_line' | 'object_tracker' | 'counter'
  | 'python_function' | 'filter' | 'param'
  | 'cpp_function'
  | 'stream_viewer' | 'video_writer' | 'trigger_webhook'

export type NodeGroup = 'input' | 'processing' | 'spatial' | 'utility' | 'output' | 'cpp'

export interface NodeMeta {
  type: NodeType
  group: NodeGroup
  label: string
  description: string
  color: string       /* Tailwind bg class */
  icon: string        /* Emoji or text icon */
}

/* --- Per-node config interfaces --- */

export interface CameraConfig {
  source_type: 'rtsp' | 'usb'
  url?: string
  device_index?: number
  fps_limit?: number
}

export interface VideoFileConfig {
  file_path: string
  loop?: boolean
  fps_limit?: number
}

export interface ImageDirectoryConfig {
  directory_path: string
  pattern?: string
  delay_ms?: number
}

export interface ResizeConfig { width: number; height: number; keep_aspect?: boolean }
export interface CropConfig   { x: number; y: number; width: number; height: number }

export interface PreprocessConfig {
  resize?: ResizeConfig
  crop?: CropConfig
  normalize?: 'none' | '0_1' | 'imagenet' | 'custom'
  mean?: number[]
  std?: number[]
}

export interface ModelInferenceConfig {
  model_id: string
  device?: 'cpu' | 'cuda'
  batch_size?: number
  conf_threshold?: number
}

export interface PostprocessNMSConfig {
  iou_threshold?: number
  conf_threshold?: number
  max_detections?: number
}

export interface DrawROIConfig {
  zone_id: string
  polygon: [number, number][]   /* [[x_pct, y_pct], ...] */
  color?: number[]
  draw_on_frame?: boolean
  filter_outside?: boolean
}

export interface DrawLineConfig {
  line_id: string
  line: [[number, number], [number, number]]   /* [[x0_pct, y0_pct], [x1_pct, y1_pct]] */
  direction?: 'both' | 'positive' | 'negative'
  color?: number[]
}

export interface ObjectTrackerConfig {
  algorithm?: 'deepsort' | 'bytetrack'
  max_age?: number
  min_hits?: number
  iou_threshold?: number
}

export interface CounterConfig {
  trigger_type: 'line_cross' | 'zone_enter' | 'zone_exit'
  trigger_id: string
  count_classes?: string[]
  reset_on_start?: boolean
}

export interface PythonFunctionConfig {
  code: string
}

export interface FilterConfig {
  allowed_classes?: string[]
  min_confidence?: number
  min_area_pct?: number
}

export interface ParamConfig {
  params: Record<string, unknown>
}

export interface CppFunctionConfig {
  source_code: string
  compiled_so_hash?: string
  compile_status?: 'uncompiled' | 'compiling' | 'ok' | 'error'
  compile_stderr?: string
  compile_flags?: string[]
  extra_libs?: string[]
}

export interface StreamViewerConfig {
  jpeg_quality?: number
  max_fps?: number
}

export interface VideoWriterConfig {
  output_path: string
  codec?: string
  fps?: number
}

export interface TriggerWebhookConfig {
  protocol: 'http' | 'mqtt'
  url?: string
  mqtt_broker?: string
  mqtt_topic?: string
  trigger_on: 'count_change' | 'detection' | 'zone_enter' | 'line_cross'
  rate_limit_s?: number
  payload_template?: string
}

export type AnyNodeConfig =
  | CameraConfig | VideoFileConfig | ImageDirectoryConfig
  | PreprocessConfig | ModelInferenceConfig | PostprocessNMSConfig
  | DrawROIConfig | DrawLineConfig | ObjectTrackerConfig | CounterConfig
  | PythonFunctionConfig | FilterConfig | ParamConfig
  | CppFunctionConfig
  | StreamViewerConfig | VideoWriterConfig | TriggerWebhookConfig

/* Data stored in React Flow node.data */
export interface CVFlowNodeData {
  nodeType: NodeType
  config: AnyNodeConfig
}
