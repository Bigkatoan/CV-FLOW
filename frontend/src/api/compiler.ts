import client from './client'

export interface CompileRequest {
  node_id: string
  source_code: string
  compile_flags?: string[]
  extra_libs?: string[]
}

export interface CompileResult {
  status: 'ok' | 'error'
  so_hash?: string
  stderr_output?: string
  compiled_at?: string
}

export const compilerApi = {
  compile: (req: CompileRequest): Promise<CompileResult> =>
    client.post('/compile', req).then((r) => r.data),

  status: (hash: string): Promise<CompileResult> =>
    client.get(`/compile/${hash}/status`).then((r) => r.data),

  downloadSdk: (): string => '/api/compile/sdk',   /* Direct download URL */
}
