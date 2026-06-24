import React from 'react'
import { type NodeProps } from 'reactflow'
import { BaseNode } from '../base/BaseNode'
import type { CVFlowNodeData, CameraConfig } from '@/types/nodes'

export function CameraNode(props: NodeProps<CVFlowNodeData>) {
  const cfg = props.data.config as CameraConfig
  return (
    <BaseNode {...props} hasInput={false}>
      <div className="space-y-0.5">
        <div className="text-gray-400">Type: <span className="text-white">{cfg.source_type?.toUpperCase()}</span></div>
        {cfg.source_type === 'rtsp' && (
          <div className="text-gray-400 truncate max-w-[160px]" title={cfg.url}>
            {cfg.url || <span className="text-yellow-400 italic">No URL set</span>}
          </div>
        )}
        {cfg.source_type === 'usb' && (
          <div className="text-gray-400">Device: <span className="text-white">/dev/video{cfg.device_index ?? 0}</span></div>
        )}
        {cfg.fps_limit && <div className="text-gray-400">FPS: <span className="text-white">{cfg.fps_limit}</span></div>}
      </div>
    </BaseNode>
  )
}
