import { create } from 'zustand'
import type { WsLogMessage } from '@/types/websocket'

type SessionStatus = 'idle' | 'running' | 'paused' | 'stopped' | 'error'

interface CounterState { [counterKey: string]: number }   /* key: `${nodeId}:${counterId}` */

interface ExecutionState {
  sessionId: string | null
  status: SessionStatus
  logs: WsLogMessage[]
  counters: CounterState
  frameNumber: number
  fps: number

  setSession: (id: string) => void
  setStatus: (status: SessionStatus, message?: string) => void
  addLog: (log: WsLogMessage) => void
  clearLogs: () => void
  updateCounter: (nodeId: string, counterId: string, value: number) => void
  resetCounters: () => void
  tickFrame: (frameNumber: number) => void
  reset: () => void
}

export const useExecutionStore = create<ExecutionState>((set) => ({
  sessionId: null,
  status: 'idle',
  logs: [],
  counters: {},
  frameNumber: 0,
  fps: 0,

  setSession: (id) => set({ sessionId: id }),

  setStatus: (status) => set({ status }),

  addLog: (log) => set((s) => ({
    logs: [...s.logs.slice(-499), log],   /* Keep last 500 */
  })),

  clearLogs: () => set({ logs: [] }),

  updateCounter: (nodeId, counterId, value) => set((s) => ({
    counters: { ...s.counters, [`${nodeId}:${counterId}`]: value },
  })),

  resetCounters: () => set({ counters: {} }),

  tickFrame: (frameNumber) => set({ frameNumber }),

  reset: () => set({
    sessionId: null, status: 'idle', logs: [],
    counters: {}, frameNumber: 0, fps: 0,
  }),
}))
