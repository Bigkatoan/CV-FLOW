import type { Node, Edge } from 'reactflow'
import type { CVFlowNodeData } from '@/types/nodes'
import type { PipelineJSON, PipelineNode, PipelineEdge } from '@/types/pipeline'

export function serializePipeline(
  id: string,
  name: string,
  nodes: Node<CVFlowNodeData>[],
  edges: Edge[],
): PipelineJSON {
  const pipelineNodes: PipelineNode[] = nodes.map((n) => ({
    id: n.id,
    type: n.data.nodeType,
    label: n.data.nodeType,   /* label stored in data to keep React Flow id separate */
    position: n.position,
    config: n.data.config,
  }))

  const pipelineEdges: PipelineEdge[] = edges.map((e) => ({
    id: e.id,
    source: e.source,
    target: e.target,
    sourceHandle: e.sourceHandle ?? 'out',
    targetHandle: e.targetHandle ?? 'in',
  }))

  return {
    version: '1.0',
    id,
    name,
    created_at: new Date().toISOString(),
    nodes: pipelineNodes,
    edges: pipelineEdges,
  }
}
