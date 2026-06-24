import React from 'react'
import { type NodeProps } from 'reactflow'
import { BaseNode, StatusDot } from '../base/BaseNode'
import type { CVFlowNodeData, ModelInferenceConfig } from '@/types/nodes'

export function ModelInferenceNode(props: NodeProps<CVFlowNodeData>) {
  const cfg = props.data.config as ModelInferenceConfig
  const hasModel = !!cfg.model_id

  return (
    <BaseNode
      {...props}
      statusBadge={<StatusDot status={hasModel ? 'ok' : 'warn'} />}
    >
      <div className="space-y-0.5">
        <div className="text-gray-400">
          Model: <span className={hasModel ? 'text-white' : 'text-yellow-400 italic'}>
            {hasModel ? cfg.model_id.slice(0, 8) + '…' : 'Not selected'}
          </span>
        </div>
        <div className="text-gray-400">
          Device: <span className={cfg.device === 'cuda' ? 'text-green-400' : 'text-white'}>
            {(cfg.device ?? 'cpu').toUpperCase()}
          </span>
        </div>
        <div className="text-gray-400">Conf: <span className="text-white">{cfg.conf_threshold ?? 0.5}</span></div>
      </div>
    </BaseNode>
  )
}
