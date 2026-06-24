import React, { useCallback, useMemo } from 'react'
import ReactFlow, {
  Background, Controls, MiniMap,
  type OnDrop, type OnDragOver,
  ReactFlowProvider,
} from 'reactflow'
import 'reactflow/dist/style.css'
import { usePipelineStore } from '@/store/pipelineStore'
import { CVFlowNode } from '@/components/nodes/CVFlowNode'
import { getDefaultConfig } from '@/utils/nodeRegistry'
import type { CVFlowNodeData, NodeType } from '@/types/nodes'

const nodeTypes = { cvflowNode: CVFlowNode }

let nodeIdCounter = 1
function genId() { return `node_${nodeIdCounter++}` }

function PipelineCanvasInner() {
  const { nodes, edges, onNodesChange, onEdgesChange, onConnect, addNode, setSelectedNodeId } =
    usePipelineStore()

  const onDragOver: OnDragOver = useCallback((e) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'move'
  }, [])

  const onDrop: OnDrop = useCallback((e) => {
    e.preventDefault()
    const nodeType = e.dataTransfer.getData('application/cvflow-node') as NodeType
    if (!nodeType) return

    const bounds = (e.target as HTMLElement).closest('.react-flow')?.getBoundingClientRect()
    if (!bounds) return

    /* Approximate canvas position (ignores zoom/pan for simplicity; improve with useReactFlow) */
    const x = e.clientX - bounds.left - 90
    const y = e.clientY - bounds.top - 30

    const id = genId()
    addNode({
      id,
      type: 'cvflowNode',
      position: { x, y },
      data: {
        nodeType,
        config: getDefaultConfig(nodeType),
      } as CVFlowNodeData,
    })
    setSelectedNodeId(id)
  }, [addNode, setSelectedNodeId])

  const memoizedNodeTypes = useMemo(() => nodeTypes, [])

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={memoizedNodeTypes}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onConnect={onConnect}
      onDrop={onDrop}
      onDragOver={onDragOver}
      onPaneClick={() => setSelectedNodeId(null)}
      fitView
      deleteKeyCode="Delete"
      className="bg-[#0f1117]"
    >
      <Background color="#1e2130" gap={20} />
      <Controls />
      <MiniMap
        nodeColor={(n) => {
          const type = (n.data as CVFlowNodeData)?.nodeType ?? ''
          if (['camera', 'video_file', 'image_directory'].includes(type)) return '#1e3a5f'
          if (['preprocess', 'model_inference', 'postprocess_nms'].includes(type)) return '#1a3d2b'
          if (type === 'cpp_function') return '#1a2d3d'
          if (['stream_viewer', 'video_writer', 'trigger_webhook'].includes(type)) return '#3d1a1a'
          return '#2d2d2d'
        }}
        maskColor="rgba(0,0,0,0.6)"
      />
    </ReactFlow>
  )
}

export function PipelineCanvas() {
  return (
    <ReactFlowProvider>
      <PipelineCanvasInner />
    </ReactFlowProvider>
  )
}
