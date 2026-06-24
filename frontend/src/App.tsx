import React, { useState, useCallback } from 'react'
import { PipelineCanvas }  from '@/components/canvas/PipelineCanvas'
import { NodePalette }     from '@/components/panels/NodePalette'
import { PropertyPanel }   from '@/components/panels/PropertyPanel'
import { ExecutionPanel }  from '@/components/panels/ExecutionPanel'
import { usePipelineStore } from '@/store/pipelineStore'
import { pipelinesApi }    from '@/api/pipelines'
import { serializePipeline }    from '@/utils/pipelineSerializer'
import { deserializePipeline }  from '@/utils/pipelineDeserializer'

export default function App() {
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle')
  const {
    nodes, edges, pipelineId, pipelineName, isDirty,
    loadPipeline, setPipelineName, markSaved,
  } = usePipelineStore()

  const handleSave = useCallback(async () => {
    setSaveStatus('saving')
    try {
      const id = pipelineId ?? crypto.randomUUID()
      const pipeline = serializePipeline(id, pipelineName, nodes, edges)
      const saved = pipelineId
        ? await pipelinesApi.update(id, pipeline)
        : await pipelinesApi.create(pipeline)
      markSaved(saved.id)
      setSaveStatus('saved')
      setTimeout(() => setSaveStatus('idle'), 2000)
    } catch {
      setSaveStatus('error')
    }
  }, [nodes, edges, pipelineId, pipelineName, markSaved])

  const handleLoad = useCallback(async () => {
    const id = window.prompt('Enter Pipeline ID to load:')
    if (!id) return
    try {
      const pipeline = await pipelinesApi.get(id)
      const { nodes: n, edges: e } = deserializePipeline(pipeline)
      loadPipeline(pipeline.id, pipeline.name, n, e)
    } catch {
      alert('Failed to load pipeline')
    }
  }, [loadPipeline])

  return (
    <div className="flex flex-col h-screen bg-[#0f1117] text-white overflow-hidden">
      {/* Top navbar */}
      <header className="flex items-center gap-3 px-4 py-2 bg-[#1a1d27] border-b border-gray-700 shrink-0">
        <span className="font-bold text-blue-400 tracking-wider text-sm">CV-FLOW</span>

        <input
          type="text"
          value={pipelineName}
          onChange={(e) => setPipelineName(e.target.value)}
          className="bg-transparent text-white text-sm border-b border-gray-600 focus:border-blue-400 outline-none px-1 w-48"
        />

        {isDirty && <span className="text-yellow-400 text-xs">● Unsaved</span>}

        <div className="flex items-center gap-2 ml-2">
          <button
            onClick={handleSave}
            disabled={saveStatus === 'saving'}
            className="px-3 py-1 text-xs bg-blue-700 hover:bg-blue-600 rounded transition-colors disabled:opacity-50"
          >
            {saveStatus === 'saving' ? 'Saving…' : saveStatus === 'saved' ? '✓ Saved' : 'Save'}
          </button>
          <button
            onClick={handleLoad}
            className="px-3 py-1 text-xs bg-gray-700 hover:bg-gray-600 rounded transition-colors"
          >
            Load
          </button>
        </div>

        <div className="ml-auto text-xs text-gray-500">
          {nodes.length} nodes · {edges.length} edges
        </div>
      </header>

      {/* Main area */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left: Node palette */}
        <NodePalette />

        {/* Center: Canvas + execution log */}
        <div className="flex flex-col flex-1 overflow-hidden">
          <div className="flex-1 overflow-hidden">
            <PipelineCanvas />
          </div>
          <div className="h-44 overflow-hidden border-t border-gray-700">
            <ExecutionPanel />
          </div>
        </div>

        {/* Right: Property panel */}
        <PropertyPanel />
      </div>
    </div>
  )
}
