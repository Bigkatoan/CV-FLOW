import client from './client'

export interface SessionStatus {
  session_id: string
  pipeline_id: string
  status: 'running' | 'stopped' | 'error' | 'completed'
  started_at: string
  stopped_at?: string
  error_msg?: string
}

export const executionApi = {
  start: (pipelineId: string, paramsOverride?: Record<string, unknown>): Promise<{ session_id: string }> =>
    client.post('/execution/start', { pipeline_id: pipelineId, params_override: paramsOverride }).then((r) => r.data),

  stop: (sessionId: string): Promise<void> =>
    client.post(`/execution/stop/${sessionId}`).then(() => undefined),

  status: (sessionId: string): Promise<SessionStatus> =>
    client.get(`/execution/status/${sessionId}`).then((r) => r.data),

  logs: (sessionId: string): Promise<string[]> =>
    client.get(`/execution/logs/${sessionId}`).then((r) => r.data),

  sessions: (): Promise<SessionStatus[]> =>
    client.get('/execution/sessions').then((r) => r.data),
}
