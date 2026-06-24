import client from './client'
import type { PipelineJSON, PipelineListItem } from '@/types/pipeline'

export const pipelinesApi = {
  list: (): Promise<PipelineListItem[]> =>
    client.get('/pipelines').then((r) => r.data),

  get: (id: string): Promise<PipelineJSON> =>
    client.get(`/pipelines/${id}`).then((r) => r.data),

  create: (pipeline: PipelineJSON): Promise<PipelineJSON> =>
    client.post('/pipelines', pipeline).then((r) => r.data),

  update: (id: string, pipeline: PipelineJSON): Promise<PipelineJSON> =>
    client.put(`/pipelines/${id}`, pipeline).then((r) => r.data),

  delete: (id: string): Promise<void> =>
    client.delete(`/pipelines/${id}`).then(() => undefined),

  validate: (pipeline: PipelineJSON): Promise<{ valid: boolean; errors: string[] }> =>
    client.post('/pipelines/validate', pipeline).then((r) => r.data),
}
