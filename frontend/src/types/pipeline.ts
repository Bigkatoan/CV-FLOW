import type { AnyNodeConfig, NodeType } from './nodes'

export interface PipelineNode {
  id: string
  type: NodeType
  label: string
  position: { x: number; y: number }
  config: AnyNodeConfig
}

export interface PipelineEdge {
  id: string
  source: string
  target: string
  sourceHandle: string
  targetHandle: string
}

export interface PipelineJSON {
  version: '1.0'
  id: string
  name: string
  description?: string
  created_at?: string
  updated_at?: string
  nodes: PipelineNode[]
  edges: PipelineEdge[]
}

export interface PipelineListItem {
  id: string
  name: string
  description?: string
  created_at: string
  updated_at: string
}
