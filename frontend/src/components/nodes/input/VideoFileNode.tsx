import React from 'react'
import { type NodeProps } from 'reactflow'
import { BaseNode } from '../base/BaseNode'
import type { CVFlowNodeData, VideoFileConfig } from '@/types/nodes'

export function VideoFileNode(props: NodeProps<CVFlowNodeData>) {
  const cfg = props.data.config as VideoFileConfig
  const filename = cfg.file_path?.split(/[\\/]/).pop() || ''
  return (
    <BaseNode {...props} hasInput={false}>
      <div className="text-gray-400 truncate max-w-[160px]" title={cfg.file_path}>
        {filename || <span className="text-yellow-400 italic">No file selected</span>}
      </div>
      {cfg.loop && <div className="text-gray-400 text-xs">Loop: ON</div>}
    </BaseNode>
  )
}
