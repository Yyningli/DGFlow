import { computed, onScopeDispose, ref, watch } from 'vue'
import { api, errorText } from './api.js'

export const tensorBoardUrl = value => value === '/tensorboard/' ? value : ''

export const regexLiteral = value => String(value).replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

// TensorBoard 2.21 stores this native preference in same-origin localStorage.
// Supply the initial preference only; an explicit user choice always wins.
export function ensureTensorBoardAutoReload(storage) {
  try {
    const settings = storage ?? globalThis.localStorage
    if (!settings) return false
    const parse = key => {
      const value = settings.getItem(key)
      if (value === null) return {}
      const parsed = JSON.parse(value)
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) throw new Error('invalid settings')
      return parsed
    }
    const legacy = parse('_tb_global_settings.timeseries'), current = parse('_tb_global_settings')
    if ([legacy, current].some(value => Object.prototype.hasOwnProperty.call(value, 'autoReload'))) return false
    settings.setItem('_tb_global_settings', JSON.stringify({ ...current, autoReload: true }))
    return true
  } catch { return false }
}

export function tensorBoardFilter(runId, selection = { kind: 'run' }, nodes = []) {
  if (selection.kind === 'all') return ''
  const root = regexLiteral(runId)
  // TensorBoard uses OS-relative run names: Windows event directories contain
  // backslashes, while Linux uses slashes. Match either without broadening IDs.
  const separator = '[/\\\\]'
  if (selection.kind === 'node') return `^${root}${separator}nodes${separator}${regexLiteral(selection.id)}$`
  if (selection.kind === 'machine') return `^${root}${separator}machines${separator}${regexLiteral(selection.id)}$`
  if (selection.kind === 'role') {
    const ids = nodes.filter(node => node.role === selection.id).map(node => regexLiteral(node.id))
    return ids.length ? `^${root}${separator}nodes${separator}(?:${ids.join('|')})$` : '^$'
  }
  return `^${root}(?:${separator}(?:nodes|machines)${separator}[^/\\\\]+)?$`
}

export function tensorBoardFrameUrl(base, runId, selection, nodes) {
  if (!tensorBoardUrl(base)) return ''
  const query = new URLSearchParams({ smoothing: '0' })
  const filter = runId ? tensorBoardFilter(runId, selection, nodes) : ''
  if (filter) query.set('runFilter', filter)
  return `${base}?${query}#timeseries`
}

export function useTensorBoard(runId, {
  selection = ref({ kind: 'run' }), nodes = ref([]), schedule = setTimeout, cancel = clearTimeout,
} = {}) {
  ensureTensorBoardAutoReload()
  const expanded = ref(true), busy = ref(false), baseUrl = ref(''), frameLoaded = ref(false)
  const frameUrl = computed(() => tensorBoardFrameUrl(baseUrl.value, runId.value, selection.value, nodes.value))
  const frameFailed = ref(false)
  const syncFailed = ref(false)
  const frameKey = ref(0), syncedRun = ref(''), message = ref('正在准备 TensorBoard…')
  let timer = null, timeout = null, controller = null, generation = 0, disposed = false

  function cancelWork() {
    generation += 1
    if (timer !== null) cancel(timer)
    if (timeout !== null) cancel(timeout)
    timer = timeout = null
    controller?.abort()
    controller = null
    busy.value = false
  }

  function unavailable(reason, preserve = false) {
    syncFailed.value = true
    if (!preserve) {
      baseUrl.value = ''
      frameLoaded.value = false
      frameFailed.value = false
    }
    message.value = (reason || 'TensorBoard 服务暂不可用。') + (preserve ? ' 已加载面板保留已有日志。' : '')
  }

  async function synchronize(reload = false) {
    if (!expanded.value || disposed) return
    cancelWork()
    const token = generation, selected = runId.value
    controller = new AbortController()
    const signal = controller.signal
    const activeController = controller
    timeout = schedule(() => activeController.abort(), 30000)
    busy.value = true
    if (reload) { frameLoaded.value = false; frameFailed.value = false; frameKey.value += 1 }
    if (!frameLoaded.value && !frameFailed.value) message.value = '正在检查服务并同步实验日志…'
    try {
      const status = await api.tensorBoardStatus(signal)
      if (token !== generation || disposed) return
      // Synchronization is also the recovery path after a previous export failure.
      const result = selected ? await api.syncTensorBoard(selected, signal) : status
      if (token !== generation || disposed) return
      const url = tensorBoardUrl(result.url)
      if (!result.available || !url) { unavailable(result.reason || status.reason || 'TensorBoard 未返回有效的同源页面地址。', !reload && Boolean(baseUrl.value)); return }
      syncedRun.value = selected
      syncFailed.value = false
      baseUrl.value = url
      message.value = frameFailed.value ? '嵌入页面加载失败，请刷新或在新窗口打开。'
        : frameLoaded.value ? '实验日志已同步。' : '实验日志已同步，正在加载 TensorBoard 页面…'
    } catch (error) {
      if (token === generation && !disposed) unavailable(error?.name === 'AbortError' ? 'TensorBoard 同步超时，正在等待重试。' : errorText(error), !reload && Boolean(baseUrl.value))
    } finally {
      if (token === generation && !disposed) {
        if (timeout !== null) cancel(timeout)
        timeout = null
        busy.value = false
        controller = null
        if (expanded.value) timer = schedule(() => synchronize(), 10000)
      }
    }
  }

  function confirmFrame(success) {
    if (!frameUrl.value || disposed) return
    frameLoaded.value = success
    frameFailed.value = !success
    if (!success) message.value = '嵌入页面加载失败，请刷新或在新窗口打开。'
    else if (!syncFailed.value) message.value = 'TensorBoard 页面已加载。'
  }

  watch([expanded, runId], () => {
    cancelWork()
    baseUrl.value = ''
    frameLoaded.value = false
    frameFailed.value = false
    syncFailed.value = false
    syncedRun.value = ''
    frameKey.value += 1
    if (expanded.value) void synchronize()
  }, { flush: 'sync', immediate: true })
  watch(selection, () => {
    cancelWork()
    if (expanded.value) void synchronize()
  }, { flush: 'sync' })
  watch(frameUrl, () => { frameLoaded.value = false; frameFailed.value = false }, { flush: 'sync' })
  onScopeDispose(() => { disposed = true; cancelWork() })
  return { expanded, busy, frameUrl, frameLoaded, frameFailed, syncFailed, frameKey, syncedRun, message, confirmFrame,
    refresh: () => synchronize(true) }
}

// Prepare historical runs for TensorBoard's native Runs selector. Export one at
// a time, only while this view exists; completed exports never navigate the frame.
export function useTensorBoardHistory(runs, selectedRun, {
  schedule = setTimeout, cancel = clearTimeout,
} = {}) {
  const exported = new Set()
  const attempts = new Map()
  let timer = null, timeout = null, controller = null, disposed = false

  function pending() {
    return runs.value.filter(run => run.run_id && run.run_id !== selectedRun.value && !exported.has(run.run_id))
      .sort((left, right) => (attempts.get(left.run_id) || 0) - (attempts.get(right.run_id) || 0))[0]?.run_id
  }
  function queue(delay = 500) {
    if (disposed || timer !== null || controller || !pending()) return
    timer = schedule(() => { timer = null; void next() }, delay)
  }
  async function next() {
    const id = pending()
    if (disposed || !id) return
    attempts.set(id, (attempts.get(id) || 0) + 1)
    controller = new AbortController()
    const active = controller
    timeout = schedule(() => active.abort(), 30000)
    let success = false
    try {
      const result = await api.syncTensorBoard(id, active.signal)
      if (!disposed && result.available && tensorBoardUrl(result.url)) {
        exported.add(id)
        success = true
      }
    } catch { /* Native TensorBoard keeps already exported runs during a failure. */ }
    finally {
      if (timeout !== null) cancel(timeout)
      timeout = null
      controller = null
      queue(success ? 500 : 10000)
    }
  }
  watch([runs, selectedRun], () => queue(), { immediate: true, deep: true })
  onScopeDispose(() => {
    disposed = true
    if (timer !== null) cancel(timer)
    if (timeout !== null) cancel(timeout)
    timer = timeout = null
    controller?.abort()
  })
}
