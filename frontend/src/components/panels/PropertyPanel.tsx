import React from 'react'
import { usePipelineStore } from '@/store/pipelineStore'
import { getNodeMeta } from '@/utils/nodeRegistry'

export function PropertyPanel() {
  const { nodes, selectedNodeId, updateNodeConfig, removeNode } = usePipelineStore()
  const selectedNode = nodes.find((n) => n.id === selectedNodeId)

  if (!selectedNode) {
    return (
      <div className="w-64 h-full bg-[#13161f] border-l border-gray-700 flex items-center justify-center">
        <p className="text-gray-600 text-sm text-center px-4">
          Select a node to edit its properties
        </p>
      </div>
    )
  }

  const meta = getNodeMeta(selectedNode.data.nodeType)
  const cfg = selectedNode.data.config

  const updateField = (key: string, value: unknown) => {
    updateNodeConfig(selectedNode.id, { [key]: value } as never)
  }

  return (
    <div className="w-64 h-full bg-[#13161f] border-l border-gray-700 flex flex-col overflow-hidden">
      {/* Header */}
      <div className={`px-3 py-3 border-b border-gray-700 ${meta.color}`}>
        <div className="flex items-center gap-2">
          <span className="text-lg">{meta.icon}</span>
          <div>
            <h3 className="text-sm font-bold text-white">{meta.label}</h3>
            <p className="text-xs text-gray-400">{meta.description}</p>
          </div>
        </div>
        <div className="mt-1 text-xs text-gray-500 font-mono">{selectedNode.id}</div>
      </div>

      {/* Properties form */}
      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {Object.entries(cfg).map(([key, val]) => (
          <div key={key}>
            <label className="block text-xs text-gray-400 mb-1 uppercase tracking-wide">
              {key.replace(/_/g, ' ')}
            </label>
            <PropertyField
              fieldKey={key}
              value={val}
              onChange={(v) => updateField(key, v)}
            />
          </div>
        ))}
      </div>

      {/* Delete */}
      <div className="px-3 py-2 border-t border-gray-700">
        <button
          onClick={() => removeNode(selectedNode.id)}
          className="w-full py-1.5 text-xs text-red-400 hover:text-red-300 hover:bg-red-900/20 rounded transition-colors"
        >
          Delete Node
        </button>
      </div>
    </div>
  )
}

function PropertyField({
  fieldKey, value, onChange,
}: {
  fieldKey: string
  value: unknown
  onChange: (v: unknown) => void
}) {
  if (typeof value === 'boolean') {
    return (
      <label className="flex items-center gap-2 cursor-pointer">
        <input
          type="checkbox"
          checked={value}
          onChange={(e) => onChange(e.target.checked)}
          className="rounded"
        />
        <span className="text-xs text-gray-300">{value ? 'Enabled' : 'Disabled'}</span>
      </label>
    )
  }

  if (typeof value === 'number') {
    return (
      <input
        type="number"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full bg-gray-800 text-white text-xs px-2 py-1.5 rounded border border-gray-600 focus:border-blue-400 outline-none"
      />
    )
  }

  if (typeof value === 'string' && (fieldKey === 'code' || fieldKey === 'source_code')) {
    return (
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        rows={6}
        className="w-full bg-gray-800 text-white text-xs px-2 py-1.5 rounded border border-gray-600 focus:border-blue-400 outline-none font-mono resize-y"
      />
    )
  }

  if (typeof value === 'string') {
    return (
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full bg-gray-800 text-white text-xs px-2 py-1.5 rounded border border-gray-600 focus:border-blue-400 outline-none"
      />
    )
  }

  if (Array.isArray(value)) {
    return (
      <input
        type="text"
        value={JSON.stringify(value)}
        onChange={(e) => {
          try { onChange(JSON.parse(e.target.value)) } catch { /* ignore parse error */ }
        }}
        className="w-full bg-gray-800 text-white text-xs px-2 py-1.5 rounded border border-gray-600 focus:border-blue-400 outline-none font-mono"
      />
    )
  }

  if (typeof value === 'object' && value !== null) {
    return (
      <textarea
        value={JSON.stringify(value, null, 2)}
        onChange={(e) => {
          try { onChange(JSON.parse(e.target.value)) } catch { /* ignore */ }
        }}
        rows={4}
        className="w-full bg-gray-800 text-white text-xs px-2 py-1.5 rounded border border-gray-600 focus:border-blue-400 outline-none font-mono resize-y"
      />
    )
  }

  return <span className="text-xs text-gray-500">{String(value)}</span>
}
