import { useRef, useCallback, useEffect } from 'react'
import type { ServerMessage, ClientMessage } from '@/types/websocket'
import { useExecutionStore } from '@/store/executionStore'

const WS_BASE = `ws://${window.location.hostname}:8765`

interface UseWebSocketReturn {
  sendMessage: (msg: ClientMessage) => void
  streamUrl: string | null
}

/* streamFrameRef is set by StreamViewer component to receive frames */
export let streamFrameCallback: ((data: string) => void) | null = null
export function setStreamFrameCallback(cb: typeof streamFrameCallback) {
  streamFrameCallback = cb
}

export function useWebSocket(sessionId: string | null): UseWebSocketReturn {
  const eventsWsRef = useRef<WebSocket | null>(null)
  const { setStatus, addLog, updateCounter, tickFrame } = useExecutionStore()

  const connect = useCallback((sid: string) => {
    /* Events channel */
    const evWs = new WebSocket(`${WS_BASE}/ws/events/${sid}`)
    eventsWsRef.current = evWs

    evWs.onmessage = (event) => {
      try {
        const msg: ServerMessage = JSON.parse(event.data as string)
        switch (msg.type) {
          case 'frame':
            streamFrameCallback?.(msg.data)
            tickFrame(msg.frame_number)
            break
          case 'status_change':
            setStatus(msg.status)
            break
          case 'counter_update':
            updateCounter(msg.node_id, msg.counter_id, msg.value)
            break
          case 'log':
            addLog(msg)
            break
          default:
            break
        }
      } catch {
        /* Ignore malformed messages */
      }
    }

    evWs.onerror = () => setStatus('error')
    evWs.onclose = () => setStatus('stopped')

    /* Keep-alive ping every 20s */
    const pingInterval = setInterval(() => {
      if (evWs.readyState === WebSocket.OPEN) {
        evWs.send(JSON.stringify({ type: 'ping' }))
      }
    }, 20_000)

    return () => {
      clearInterval(pingInterval)
      evWs.close()
    }
  }, [setStatus, addLog, updateCounter, tickFrame])

  useEffect(() => {
    if (!sessionId) return
    const cleanup = connect(sessionId)
    return cleanup
  }, [sessionId, connect])

  const sendMessage = useCallback((msg: ClientMessage) => {
    if (eventsWsRef.current?.readyState === WebSocket.OPEN) {
      eventsWsRef.current.send(JSON.stringify(msg))
    }
  }, [])

  return {
    sendMessage,
    streamUrl: sessionId ? `${WS_BASE}/ws/stream/${sessionId}` : null,
  }
}
