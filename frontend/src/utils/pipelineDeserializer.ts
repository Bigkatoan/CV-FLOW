import type { Node, Edge } from 'reactflow'
import type { CVFlowNodeData } from '@/types/nodes'
import type { PipelineJSON } from '@/types/pipeline'

export function deserializePipeline(pipeline: PipelineJSON): {
  nodes: Node<CVFlowNodeData>[]
  edges: Edge[]
} {
  const nodes: Node<CVFlowNodeData>[] = pipeline.nodes.map((pn) => ({
    id: pn.id,
    type: 'cvflowNode',       /* single React Flow node component handles all types */
    position: pn.position,
    data: {
      nodeType: pn.type,
      config: pn.config,
    },
  }))

  const edges: Edge[] = pipeline.edges.map((pe) => ({
    id: pe.id,
    source: pe.source,
    target: pe.target,
    sourceHandle: pe.sourceHandle,
    targetHandle: pe.targetHandle,
    animated: true,
  }))

  return { nodes, edges }
}
