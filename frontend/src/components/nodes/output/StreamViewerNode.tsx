import React, { useEffect, useRef } from 'react'
import { type NodeProps } from 'reactflow'
import { BaseNode } from '../base/BaseNode'
import type { CVFlowNodeData } from '@/types/nodes'
import { setStreamFrameCallback } from '@/hooks/useWebSocket'

export function StreamViewerNode(props: NodeProps<CVFlowNodeData>) {
  const imgRef = useRef<HTMLImageElement>(null)

  useEffect(() => {
    setStreamFrameCallback((b64data) => {
      if (imgRef.current) {
        imgRef.current.src = `data:image/jpeg;base64,${b64data}`
      }
    })
    return () => setStreamFrameCallback(null)
  }, [])

  return (
    <BaseNode {...props} hasOutput={false}>
      <div className="mt-1 rounded overflow-hidden bg-black">
        <img
          ref={imgRef}
          alt="Live stream"
          className="w-48 h-27 object-contain"
          style={{ width: 192, height: 108 }}
          src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
        />
      </div>
    </BaseNode>
  )
}
