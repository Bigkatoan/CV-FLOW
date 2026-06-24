/* All WebSocket message types — discriminated union on 'type' field */

export interface WsFrameMessage {
  type: 'frame'
  session_id: string
  frame_number: number
  timestamp: number
  data: string           /* base64-encoded JPEG */
}

export interface WsDetection {
  class_name: string
  confidence: number
  bbox: [number, number, number, number]   /* xyxy */
  track_id: number
}

export interface WsDetectionResultMessage {
  type: 'detection_result'
  session_id: string
  node_id: string
  frame_number: number
  detections: WsDetection[]
}

export interface WsCounterUpdateMessage {
  type: 'counter_update'
  session_id: string
  node_id: string
  counter_id: string
  value: number
  delta: number
}

export interface WsStatusChangeMessage {
  type: 'status_change'
  session_id: string
  status: 'running' | 'paused' | 'stopped' | 'error'
  message?: string
}

export interface WsLogMessage {
  type: 'log'
  session_id: string
  level: 'info' | 'warning' | 'error'
  node_id?: string
  message: string
}

export interface WsCompileResultMessage {
  type: 'compile_result'
  node_id: string
  status: 'ok' | 'error'
  so_hash?: string
  stderr?: string
}

/* Client → Server messages */
export interface WsPingMessage        { type: 'ping' }
export interface WsResetCounterMsg    { type: 'reset_counter'; node_id: string }
export interface WsUpdateParamMsg     { type: 'update_param'; node_id: string; key: string; value: unknown }

export type ServerMessage =
  | WsFrameMessage
  | WsDetectionResultMessage
  | WsCounterUpdateMessage
  | WsStatusChangeMessage
  | WsLogMessage
  | WsCompileResultMessage

export type ClientMessage =
  | WsPingMessage
  | WsResetCounterMsg
  | WsUpdateParamMsg
