import React from 'react'
import { type NodeProps } from 'reactflow'
import { BaseNode } from '../base/BaseNode'
import type { CVFlowNodeData, CounterConfig } from '@/types/nodes'
import { useExecutionStore } from '@/store/executionStore'

export function CounterNode(props: NodeProps<CVFlowNodeData>) {
  const cfg = props.data.config as CounterConfig
  const counters = useExecutionStore((s) => s.counters)

  /* Find any counter keys for this node */
  const nodeCounters = Object.entries(counters).filter(([k]) => k.startsWith(props.id + ':'))

  return (
    <BaseNode {...props}>
      <div className="space-y-1">
        <div className="text-gray-400 text-xs">
          Trigger: <span className="text-white">{cfg.trigger_type}</span>
        </div>
        <div className="text-gray-400 text-xs">
          ID: <span className="text-white">{cfg.trigger_id}</span>
        </div>
        {nodeCounters.length > 0 ? (
          nodeCounters.map(([key, val]) => (
            <div key={key} className="flex items-center gap-2 mt-1">
              <span className="text-gray-400 text-xs">{key.split(':')[1]}:</span>
              <span className="text-2xl font-bold text-green-400">{val}</span>
            </div>
          ))
        ) : (
          <div className="text-3xl font-bold text-green-400 mt-1">0</div>
        )}
      </div>
    </BaseNode>
  )
}
