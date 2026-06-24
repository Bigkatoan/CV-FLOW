import React from 'react'
import { Handle, Position, type NodeProps } from 'reactflow'
import type { CVFlowNodeData } from '@/types/nodes'
import { getNodeMeta } from '@/utils/nodeRegistry'
import { usePipelineStore } from '@/store/pipelineStore'

interface BaseNodeProps extends NodeProps<CVFlowNodeData> {
  children?: React.ReactNode
  hasInput?: boolean
  hasOutput?: boolean
  statusBadge?: React.ReactNode
}

export function BaseNode({
  id,
  data,
  selected,
  children,
  hasInput = true,
  hasOutput = true,
  statusBadge,
}: BaseNodeProps) {
  const meta = getNodeMeta(data.nodeType)
  const setSelectedNodeId = usePipelineStore((s) => s.setSelectedNodeId)

  return (
    <div
      onClick={() => setSelectedNodeId(id)}
      className={`
        min-w-[180px] rounded-lg border-2 shadow-xl
        ${selected ? 'border-blue-400' : 'border-gray-600'}
        ${meta.color} text-white cursor-pointer select-none
        transition-all duration-150
      `}
    >
      {/* Handle IN */}
      {hasInput && (
        <Handle
          type="target"
          position={Position.Left}
          id="in"
          style={{ top: '50%', left: -6 }}
        />
      )}

      {/* Title bar */}
      <div className="flex items-center gap-2 px-3 py-2 border-b border-white/10">
        <span className="text-base">{meta.icon}</span>
        <span className="font-semibold text-sm tracking-wide">{meta.label}</span>
        <div className="ml-auto flex items-center gap-1">
          {statusBadge}
        </div>
      </div>

      {/* Body */}
      {children && (
        <div className="px-3 py-2 text-xs text-gray-300">
          {children}
        </div>
      )}

      {/* Handle OUT */}
      {hasOutput && (
        <Handle
          type="source"
          position={Position.Right}
          id="out"
          style={{ top: '50%', right: -6 }}
        />
      )}
    </div>
  )
}

/* Status dot helper */
export function StatusDot({ status }: { status: 'ok' | 'error' | 'warn' | 'idle' }) {
  const colors = { ok: 'bg-green-400', error: 'bg-red-400', warn: 'bg-yellow-400', idle: 'bg-gray-500' }
  return <span className={`w-2 h-2 rounded-full ${colors[status]}`} />
}
