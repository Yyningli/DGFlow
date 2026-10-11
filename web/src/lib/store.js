/** Shared control-plane state: one poller feeds every view. */
import { computed, reactive, ref } from 'vue'
import { api, errorText } from './api.js'
import { isActiveRun } from './format.js'
import { cacheRunRecord, reconcileRunCache } from './runRecords.js'
import { deploymentConfig, deploymentState, deployedTopology, runCompatibilityError, sameDeployment, TOPOLOGY_FIELDS } from './deployment.js'
import { trainingBackendError, trainingBackendStatus } from './trainingBackends.js'
import { datasetName, datasetReady as isDatasetReady, modelConfigurationError, withPreparedDataset } from './datasets.js'
import { trainingEligibility } from './trainingEligibility.js'

const status = ref(null)
const connected = ref(false)
const connectionError = ref('')
const initialLoading = ref(true)
const runs = ref([])
const listError = ref('')
const selectedRunId = ref('')
const currentRun = ref(null)
const loadingRun = ref(false)
const runError = ref('')
const lastUpdated = ref(null)
const operationError = ref('')
const notification = ref('')
const preparing = ref(false)
const selectedDataset = ref('mnist')
const submitting = ref(false)
const deploying = ref(false)
const stopping = ref(false)
const stopRequestedId = ref('')

/** Full records keyed by run id — reused by the comparison view.
 *  Reactive so views that read from it re-render when a record arrives. */
const runCache = reactive(new Map())

let timer = null
let pollController = null
let actionController = null
let stopped = false
let pollCount = 0
let listLoaded = false

const hasActiveRun = computed(() =>
  Boolean(status.value?.active_run_id) || isActiveRun(currentRun.value) || runs.value.some(isActiveRun),
)

const datasetReady = computed(() => isDatasetReady(status.value, selectedDataset.value))
const nodes = computed(() => status.value?.nodes || [])

function defaultRunId(activeRunId) {
  if (activeRunId) return activeRunId
  let latest = null
  let latestTime = -Infinity
  for (const run of runs.value) {
    if (!run?.run_id) continue
    const created = typeof run.created_at === 'string' ? Date.parse(run.created_at) : NaN
    const time = Number.isFinite(created) ? created : -Infinity
    if (!latest || time > latestTime) { latest = run; latestTime = time }
  }
  return latest?.run_id || ''
}

async function refreshRuns(signal) {
  try {
    const data = await api.listRuns(signal)
    runs.value = Array.isArray(data.runs) ? data.runs : []
    reconcileRunCache(runCache, runs.value)
    listLoaded = true
    listError.value = ''
  } catch (error) {
    if (error.name !== 'AbortError') listError.value = `无法读取实验记录：${errorText(error)}`
  }
}

async function poll() {
  if (stopped) return
  pollController = new AbortController()
  const { signal } = pollController
  const requestTimeout = window.setTimeout(() => pollController?.abort(), 12000)
  try {
    const data = await api.status(signal)
    status.value = data
    connected.value = true
    connectionError.value = ''
    if (!listLoaded || pollCount % 5 === 0) await refreshRuns(signal)
    if (!selectedRunId.value) selectedRunId.value = defaultRunId(data.active_run_id)
    if (selectedRunId.value) {
      const requestedId = selectedRunId.value
      try {
        const run = await api.getRun(requestedId, signal)
        if (selectedRunId.value === requestedId) {
          const previous = currentRun.value?.status
          currentRun.value = cacheRunRecord(runCache, requestedId, run)
          loadingRun.value = false
          runError.value = ''
          if (previous && previous !== run.status) await refreshRuns(signal)
        }
      } catch (error) {
        if (error.name === 'AbortError') throw error
        if (selectedRunId.value === requestedId) {
          runError.value =
            error.status === 404
              ? '该实验记录不存在或已被移除，请重新选择实验。'
              : `暂时无法更新实验：${errorText(error)}`
          loadingRun.value = false
          if (error.status === 404) currentRun.value = null
        }
      }
    }
    lastUpdated.value = new Date().toISOString()
  } catch (error) {
    if (!stopped) {
      connected.value = false
      connectionError.value =
        error.name === 'AbortError' ? '接口响应超时，请检查后端服务。' : errorText(error)
    }
  } finally {
    clearTimeout(requestTimeout)
    initialLoading.value = false
    pollCount += 1
    if (!stopped) timer = window.setTimeout(poll, 2000)
  }
}

function start() {
  if (timer || stopped) return
  poll()
}

function stop() {
  stopped = true
  clearTimeout(timer)
  timer = null
  pollController?.abort()
  actionController?.abort()
}

function chooseRun(id) {
  selectedRunId.value = id
  currentRun.value = id ? runCache.get(id) || null : null
  loadingRun.value = Boolean(id)
  runError.value = ''
}

function clearMessages() {
  operationError.value = ''
  notification.value = ''
}

async function prepareData(dataset = selectedDataset.value) {
  if (preparing.value || submitting.value || deploying.value || hasActiveRun.value || !connected.value) return
  const datasetError = modelConfigurationError({ dataset })
  if (datasetError) { operationError.value = datasetError; return }
  preparing.value = true
  clearMessages()
  actionController = new AbortController()
  try {
    const result = await api.prepareData(dataset, actionController.signal)
    if (result.status === 'ready') {
      status.value = withPreparedDataset(status.value, dataset)
      notification.value = `${datasetName(dataset)} 数据已准备完成，可以创建任务。`
    } else {
      notification.value = '数据准备请求已提交，数据状态以后端返回为准。'
    }
  } catch (error) {
    if (error.name !== 'AbortError') operationError.value = `数据准备失败：${errorText(error)}`
  } finally {
    preparing.value = false
  }
}

async function refreshStatus(signal) {
  const data = await api.status(signal)
  status.value = data
  connected.value = true
  connectionError.value = ''
  lastUpdated.value = new Date().toISOString()
  return data
}

async function deployNodes(configuration, startNodes = true) {
  const config = deploymentConfig(configuration, deployedTopology(status.value) || undefined)
  const state = deploymentState(status.value, config, connected.value, hasActiveRun.value,
    deploying.value || submitting.value || preparing.value || stopping.value)
  if (state.reason) { operationError.value = state.reason; return false }
  deploying.value = true
  clearMessages()
  const controller = new AbortController()
  actionController = controller
  const timeout = window.setTimeout(() => controller.abort(), 120000)
  try {
    const response = await (startNodes ? api.startDeployment : api.initializeDeployment)(config, controller.signal)
    if (TOPOLOGY_FIELDS.some(field => response[field] !== undefined && response[field] !== config[field])) throw new Error('后端返回的部署拓扑与所选配置不一致。')
    await refreshStatus(controller.signal)
    if (!sameDeployment(deployedTopology(status.value), config)) throw new Error('实际部署的节点数量或门限与所选配置不一致。')
    notification.value = startNodes
      ? `已应用 ${config.client_count} 个客户端、${config.authority_count} 个边缘服务器、${config.aggregator_count} 个云服务器；节点在线情况以后端状态为准。`
      : `已初始化三层拓扑的真实身份；启动节点后可运行实验。`
    return true
  } catch (error) {
    operationError.value = error.name === 'AbortError'
      ? '部署操作超时，请刷新实际节点状态确认结果后再试。'
      : `部署操作失败：${errorText(error)}`
    // Reconfiguration may have committed before a node startup failed.
    // Only a new status response may update the displayed real topology.
    const refreshController = new AbortController()
    const refreshTimeout = window.setTimeout(() => refreshController.abort(), 10000)
    try { await refreshStatus(refreshController.signal) }
    catch { connected.value = false; connectionError.value = '部署后的节点状态暂未确认，正在自动重试。' }
    finally { clearTimeout(refreshTimeout) }
    return false
  } finally {
    clearTimeout(timeout)
    deploying.value = false
  }
}

async function startRun(config) {
  if (submitting.value || deploying.value) return null
  const modelError = modelConfigurationError(config)
  if (modelError) { operationError.value = modelError; return null }
  const compatibilityError = connected.value ? runCompatibilityError(status.value, config.mode) : ''
  if (compatibilityError) { operationError.value = compatibilityError; return null }
  const backendError = trainingBackendError(config.backend ?? 'numpy',
    trainingBackendStatus(status.value?.capabilities?.training_backends, connected.value))
  if (backendError) { operationError.value = backendError; return null }
  const participants = trainingEligibility(status.value, config, connected.value)
  if (!participants.ready) { operationError.value = participants.reason; return null }
  submitting.value = true
  clearMessages()
  actionController = new AbortController()
  try {
    const result = await api.startRun(config, actionController.signal)
    if (!result.run_id) throw new Error('后端未返回实验编号，请刷新实验列表确认任务状态。')
    status.value = { ...status.value, active_run_id: result.run_id }
    selectedRunId.value = result.run_id
    currentRun.value = null
    loadingRun.value = true
    stopRequestedId.value = ''
    await refreshRuns()
    return result.run_id
  } catch (error) {
    if (error.name !== 'AbortError') operationError.value = `创建实验失败：${errorText(error)}`
    return null
  } finally {
    submitting.value = false
  }
}

async function stopRun() {
  if (stopping.value || !isActiveRun(currentRun.value) || stopRequestedId.value === currentRun.value?.run_id) return
  stopping.value = true
  operationError.value = ''
  const id = currentRun.value.run_id
  actionController = new AbortController()
  try {
    await api.stopRun(id, actionController.signal)
    stopRequestedId.value = id
    notification.value = '停止请求已发送，等待后端结束当前任务。'
  } catch (error) {
    if (error.name !== 'AbortError') operationError.value = `停止请求失败：${errorText(error)}`
  } finally {
    stopping.value = false
  }
}

export const consoleStore = {
  status, connected, connectionError, initialLoading,
  runs, listError, selectedRunId, currentRun, loadingRun, runError,
  lastUpdated, operationError, notification,
  preparing, selectedDataset, submitting, deploying, stopping, stopRequestedId,
  hasActiveRun, datasetReady, nodes, runCache,
  start, stop, chooseRun, refreshRuns, refreshStatus, deployNodes, prepareData, startRun, stopRun, clearMessages,
}
