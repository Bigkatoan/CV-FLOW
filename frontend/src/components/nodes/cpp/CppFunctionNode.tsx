import React from 'react'
import { type NodeProps } from 'reactflow'
import { BaseNode, StatusDot } from '../base/BaseNode'
import type { CVFlowNodeData, CppFunctionConfig } from '@/types/nodes'

const statusMap = {
  uncompiled: { dot: 'idle' as const,  label: 'Not compiled',   color: 'text-gray-400' },
  compiling:  { dot: 'warn' as const,  label: 'Compiling…',     color: 'text-yellow-400' },
  ok:         { dot: 'ok' as const,    label: 'Compiled',       color: 'text-green-400' },
  error:      { dot: 'error' as const, label: 'Compile error',  color: 'text-red-400' },
}

export function CppFunctionNode(props: NodeProps<CVFlowNodeData>) {
  const cfg = props.data.config as CppFunctionConfig
  const st = statusMap[cfg.compile_status ?? 'uncompiled']
  const linesCount = (cfg.source_code ?? '').split('\n').length

  return (
    <BaseNode
      {...props}
      statusBadge={<StatusDot status={st.dot} />}
    >
      <div className="space-y-0.5">
        <div className={`font-medium ${st.color}`}>{st.label}</div>
        <div className="text-gray-400">{linesCount} lines</div>
        {cfg.compiled_so_hash && (
          <div className="text-gray-500 font-mono text-xs truncate max-w-[160px]">
            {cfg.compiled_so_hash.slice(0, 12)}…
          </div>
        )}
      </div>
    </BaseNode>
  )
}
