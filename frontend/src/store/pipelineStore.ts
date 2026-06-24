import { create } from 'zustand'
import {
  type Node, type Edge,
  applyNodeChanges, applyEdgeChanges,
  type NodeChange, type EdgeChange,
  type Connection, addEdge,
} from 'reactflow'
import type { CVFlowNodeData } from '@/types/nodes'

interface PipelineState {
  nodes: Node<CVFlowNodeData>[]
  edges: Edge[]
  selectedNodeId: string | null
  pipelineId: string | null
  pipelineName: string
  isDirty: boolean

  setNodes: (nodes: Node<CVFlowNodeData>[]) => void
  setEdges: (edges: Edge[]) => void
  onNodesChange: (changes: NodeChange[]) => void
  onEdgesChange: (changes: EdgeChange[]) => void
  onConnect: (connection: Connection) => void
  addNode: (node: Node<CVFlowNodeData>) => void
  updateNodeConfig: (nodeId: string, config: Partial<CVFlowNodeData['config']>) => void
  removeNode: (nodeId: string) => void
  setSelectedNodeId: (id: string | null) => void
  loadPipeline: (id: string, name: string, nodes: Node<CVFlowNodeData>[], edges: Edge[]) => void
  setPipelineName: (name: string) => void
  markSaved: (id: string) => void
  reset: () => void
}

export const usePipelineStore = create<PipelineState>((set, get) => ({
  nodes: [],
  edges: [],
  selectedNodeId: null,
  pipelineId: null,
  pipelineName: 'Untitled Pipeline',
  isDirty: false,

  setNodes: (nodes) => set({ nodes, isDirty: true }),
  setEdges: (edges) => set({ edges, isDirty: true }),

  onNodesChange: (changes) => set((s) => ({
    nodes: applyNodeChanges(changes, s.nodes) as Node<CVFlowNodeData>[],
    isDirty: true,
  })),

  onEdgesChange: (changes) => set((s) => ({
    edges: applyEdgeChanges(changes, s.edges),
    isDirty: true,
  })),

  onConnect: (connection) => set((s) => ({
    edges: addEdge({ ...connection, animated: true }, s.edges),
    isDirty: true,
  })),

  addNode: (node) => set((s) => ({
    nodes: [...s.nodes, node],
    isDirty: true,
  })),

  updateNodeConfig: (nodeId, config) => set((s) => ({
    nodes: s.nodes.map((n) =>
      n.id === nodeId
        ? { ...n, data: { ...n.data, config: { ...n.data.config, ...config } } }
        : n
    ),
    isDirty: true,
  })),

  removeNode: (nodeId) => set((s) => ({
    nodes: s.nodes.filter((n) => n.id !== nodeId),
    edges: s.edges.filter((e) => e.source !== nodeId && e.target !== nodeId),
    selectedNodeId: s.selectedNodeId === nodeId ? null : s.selectedNodeId,
    isDirty: true,
  })),

  setSelectedNodeId: (id) => set({ selectedNodeId: id }),

  loadPipeline: (id, name, nodes, edges) => set({
    pipelineId: id, pipelineName: name, nodes, edges,
    selectedNodeId: null, isDirty: false,
  }),

  setPipelineName: (name) => set({ pipelineName: name, isDirty: true }),

  markSaved: (id) => set({ pipelineId: id, isDirty: false }),

  reset: () => set({
    nodes: [], edges: [], selectedNodeId: null,
    pipelineId: null, pipelineName: 'Untitled Pipeline', isDirty: false,
  }),
}))
