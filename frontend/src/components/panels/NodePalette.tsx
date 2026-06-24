import React from 'react'
import type { NodeMeta, NodeGroup } from '@/types/nodes'
import { getAllNodeMetas } from '@/utils/nodeRegistry'

const groups: { id: NodeGroup; label: string }[] = [
  { id: 'input',      label: 'Input' },
  { id: 'processing', label: 'Processing & AI' },
  { id: 'spatial',    label: 'Spatial Logic' },
  { id: 'utility',    label: 'Utility' },
  { id: 'cpp',        label: 'C++ (High Perf)' },
  { id: 'output',     label: 'Output' },
]

function NodePaletteItem({ meta }: { meta: NodeMeta }) {
  const onDragStart = (e: React.DragEvent) => {
    e.dataTransfer.setData('application/cvflow-node', meta.type)
    e.dataTransfer.effectAllowed = 'move'
  }

  return (
    <div
      draggable
      onDragStart={onDragStart}
      className={`
        flex items-center gap-2 px-3 py-2 rounded-md cursor-grab active:cursor-grabbing
        ${meta.color} border border-white/10 hover:border-white/30
        transition-all duration-100 select-none
      `}
      title={meta.description}
    >
      <span>{meta.icon}</span>
      <span className="text-xs font-medium text-white">{meta.label}</span>
    </div>
  )
}

export function NodePalette() {
  const allMetas = getAllNodeMetas()

  return (
    <div className="w-52 h-full bg-[#13161f] border-r border-gray-700 flex flex-col overflow-hidden">
      <div className="px-3 py-3 border-b border-gray-700">
        <h2 className="text-xs font-bold text-gray-400 uppercase tracking-widest">Nodes</h2>
      </div>

      <div className="flex-1 overflow-y-auto p-2 space-y-4">
        {groups.map((group) => {
          const metas = allMetas.filter((m) => m.group === group.id)
          if (metas.length === 0) return null
          return (
            <div key={group.id}>
              <p className="text-xs text-gray-500 uppercase tracking-wider px-1 mb-1.5">
                {group.label}
              </p>
              <div className="space-y-1">
                {metas.map((m) => <NodePaletteItem key={m.type} meta={m} />)}
              </div>
            </div>
          )
        })}
      </div>

      <div className="px-3 py-2 border-t border-gray-700 text-xs text-gray-600 text-center">
        Drag node onto canvas
      </div>
    </div>
  )
}
