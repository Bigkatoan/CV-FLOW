import React, { useState, useCallback } from 'react'
import { usePipelineStore } from '@/store/pipelineStore'
import { useExecutionStore } from '@/store/executionStore'
import { useWebSocket } from '@/hooks/useWebSocket'
import { serializePipeline } from '@/utils/pipelineSerializer'
import { pipelinesApi } from '@/api/pipelines'
import { executionApi } from '@/api/execution'

function LogLine({ log }: { log: { level: string; message: string; node_id?: string } }) {
  const color = { info: 'text-gray-300', warning: 'text-yellow-400', error: 'text-red-400' }[log.level] ?? 'text-gray-300'
  return (
    <div className={`text-xs font-mono ${color} flex gap-2`}>
      <span className="text-gray-600 shrink-0">[{log.level[0].toUpperCase()}]</span>
      {log.node_id && <span className="text-blue-400 shrink-0">{log.node_id}</span>}
      <span className="break-all">{log.message}</span>
    </div>
  )
}

export function ExecutionPanel() {
  const [isRunning, setIsRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const { nodes, edges, pipelineId, pipelineName } = usePipelineStore()
  const { sessionId, status, logs, frameNumber, setSession, reset: resetExecution } = useExecutionStore()
  const { sendMessage } = useWebSocket(sessionId)

  const handleRun = useCallback(async () => {
    setError(null)
    try {
      /* Save pipeline first if not saved */
      let id = pipelineId ?? crypto.randomUUID()
      const pipeline = serializePipeline(id, pipelineName, nodes, edges)
      const saved = pipelineId
        ? await pipelinesApi.update(id, pipeline)
        : await pipelinesApi.create(pipeline)
      id = saved.id

      /* Start execution */
      const { session_id } = await executionApi.start(id)
      setSession(session_id)
      setIsRunning(true)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to start pipeline')
    }
  }, [nodes, edges, pipelineId, pipelineName, setSession])

  const handleStop = useCallback(async () => {
    if (!sessionId) return
    try {
      await executionApi.stop(sessionId)
    } catch { /* ignore */ }
    setIsRunning(false)
    resetExecution()
  }, [sessionId, resetExecution])

  return (
    <div className="h-full flex flex-col bg-[#13161f] border-t border-gray-700">
      {/* Controls bar */}
      <div className="flex items-center gap-2 px-3 py-2 border-b border-gray-700">
        <button
          onClick={isRunning ? handleStop : handleRun}
          className={`
            px-4 py-1.5 rounded text-sm font-semibold transition-colors
            ${isRunning
              ? 'bg-red-700 hover:bg-red-600 text-white'
              : 'bg-green-700 hover:bg-green-600 text-white'}
          `}
        >
          {isRunning ? '⏹ Stop' : '▶ Run'}
        </button>

        <div className="flex items-center gap-1.5 text-xs text-gray-400">
          <span className={`w-2 h-2 rounded-full ${
            status === 'running' ? 'bg-green-400 animate-pulse' :
            status === 'error'   ? 'bg-red-400' :
            'bg-gray-600'
          }`} />
          <span className="capitalize">{status}</span>
          {frameNumber > 0 && <span className="text-gray-600">| Frame {frameNumber}</span>}
        </div>

        <button
          onClick={() => useExecutionStore.getState().clearLogs()}
          className="ml-auto text-xs text-gray-500 hover:text-gray-300 transition-colors"
        >
          Clear logs
        </button>
      </div>

      {error && (
        <div className="px-3 py-2 text-xs text-red-400 bg-red-900/20 border-b border-red-900">
          {error}
        </div>
      )}

      {/* Log output */}
      <div className="flex-1 overflow-y-auto p-2 space-y-0.5 font-mono">
        {logs.length === 0 ? (
          <p className="text-gray-600 text-xs text-center mt-4">No logs yet</p>
        ) : (
          logs.map((log, i) => <LogLine key={i} log={log} />)
        )}
      </div>
    </div>
  )
}
