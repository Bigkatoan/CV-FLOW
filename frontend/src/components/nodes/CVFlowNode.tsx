import React from 'react'
import { type NodeProps } from 'reactflow'
import type { CVFlowNodeData } from '@/types/nodes'

/* Input nodes */
import { CameraNode }          from './input/CameraNode'
import { VideoFileNode }       from './input/VideoFileNode'

/* Processing nodes */
import { ModelInferenceNode }  from './processing/ModelInferenceNode'

/* Spatial nodes */
import { CounterNode }         from './spatial/CounterNode'

/* C++ node */
import { CppFunctionNode }     from './cpp/CppFunctionNode'

/* Output nodes */
import { StreamViewerNode }    from './output/StreamViewerNode'

/* Generic fallback node for types without a dedicated component */
import { BaseNode }            from './base/BaseNode'

const nodeComponents: Partial<Record<string, React.ComponentType<NodeProps<CVFlowNodeData>>>> = {
  camera:           CameraNode,
  video_file:       VideoFileNode,
  model_inference:  ModelInferenceNode,
  counter:          CounterNode,
  cpp_function:     CppFunctionNode,
  stream_viewer:    StreamViewerNode,
}

export function CVFlowNode(props: NodeProps<CVFlowNodeData>) {
  const Component = nodeComponents[props.data.nodeType]
  if (Component) return <Component {...props} />
  /* Generic fallback — shows node type label only */
  return <BaseNode {...props} />
}
