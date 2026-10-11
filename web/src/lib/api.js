/** Thin client for the DGFlow control-plane JSON API. */
import { deploymentConfig, deploymentConfigError } from './deployment.js'

function deploymentRequest(action, configuration, signal) {
  const config = deploymentConfig(configuration)
  const error = deploymentConfigError(config)
  if (error) throw new Error(error)
  return request(`/deployment/${action}`, { method: 'POST', body: JSON.stringify(config), signal })
}

function describeError(response, data) {
  const detail = Array.isArray(data.detail)
    ? data.detail
        .map(item => `${(item.loc || []).filter(part => part !== 'body').join('.')}: ${item.msg || '参数无效'}`)
        .join('；')
    : data.detail
  const message =
    typeof detail === 'string' ? detail
    : typeof data.error === 'string' ? data.error
    : typeof data.message === 'string' ? data.message
    : `请求失败（HTTP ${response.status}）`
  const error = new Error(message)
  error.status = response.status
  return error
}

async function request(path, options = {}) {
  const response = await fetch(`/api${path}`, {
    ...options,
    headers: {
      ...(options.body ? { 'Content-Type': 'application/json' } : {}),
      ...options.headers,
    },
  })
  const body = await response.text()
  let data
  try {
    data = body ? JSON.parse(body) : {}
  } catch {
    throw new Error(`接口响应格式错误（HTTP ${response.status}），请确认后端服务已启动。`)
  }
  if (!response.ok) throw describeError(response, data)
  return data
}

export const api = {
  status: signal => request('/status', { signal }),
  tensorBoardStatus: signal => request('/tensorboard/status', { signal }),
  syncTensorBoard: (id, signal) => request(`/runs/${encodeURIComponent(id)}/tensorboard`, { method: 'POST', signal }),
  prepareData: (dataset = 'mnist', signal) => request('/data/prepare', { method: 'POST', body: JSON.stringify({ dataset }), signal }),
  prepareCompute: signal => request('/compute/prepare', { method: 'POST', body: '{}', signal }),
  initializeDeployment: (configuration, signal) => deploymentRequest('init', configuration, signal),
  startDeployment: (configuration, signal) => deploymentRequest('start', configuration, signal),
  listRuns: signal => request('/runs', { signal }),
  getRun: (id, signal) => request(`/runs/${encodeURIComponent(id)}`, { signal }),
  getMonitor: (id, signal) => request(`/runs/${encodeURIComponent(id)}/monitor`, { signal }),
  startRun: (config, signal) => request('/runs', { method: 'POST', body: JSON.stringify(config), signal }),
  stopRun: (id, signal) => request(`/runs/${encodeURIComponent(id)}/stop`, { method: 'POST', signal }),
}

export const exportUrl = id => `/api/runs/${encodeURIComponent(id)}/export`

export function errorText(error) {
  return error instanceof Error ? error.message : '请求失败，请稍后重试。'
}
