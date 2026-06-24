import client from './client'

export interface ModelEntry {
  id: string
  name: string
  version: string
  task: 'detection' | 'classification' | 'segmentation' | 'pose'
  config: Record<string, unknown>
  uploaded_at: string
}

export const modelsApi = {
  list: (): Promise<ModelEntry[]> =>
    client.get('/models').then((r) => r.data),

  get: (id: string): Promise<ModelEntry> =>
    client.get(`/models/${id}`).then((r) => r.data),

  upload: (formData: FormData, onProgress?: (pct: number) => void): Promise<ModelEntry> =>
    client.post('/models/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (e) => {
        if (onProgress && e.total) onProgress(Math.round((e.loaded * 100) / e.total))
      },
    }).then((r) => r.data),

  delete: (id: string): Promise<void> =>
    client.delete(`/models/${id}`).then(() => undefined),

  reload: (id: string): Promise<void> =>
    client.post(`/models/${id}/reload`).then(() => undefined),
}
