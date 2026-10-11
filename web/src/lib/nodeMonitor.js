import { onScopeDispose, ref, watch } from 'vue'
import { api, errorText } from './api.js'

export const monitorStatus = status => ({
  starting: '正在采集', online: '采集中', unavailable: '采集异常', recorded: '已记录', missing: '未记录',
  stopped: '采集已停止', completed: '已记录',
})[status] || '未记录'

export function useNodeMonitor(runId, { schedule = setTimeout, cancel = clearTimeout } = {}) {
  const snapshot = ref(null), busy = ref(false), error = ref(''), updatedAt = ref('')
  let timer = null, timeout = null, controller = null, generation = 0, disposed = false
  function cancelWork() {
    generation += 1
    if (timer !== null) cancel(timer)
    if (timeout !== null) cancel(timeout)
    timer = timeout = null
    controller?.abort()
    controller = null
  }
  async function refresh() {
    if (!runId.value || disposed) return
    cancelWork()
    const token = generation, selected = runId.value, activeController = new AbortController()
    controller = activeController
    timeout = schedule(() => activeController.abort(), 8000)
    busy.value = true
    try {
      const result = await api.getMonitor(selected, activeController.signal)
      if (disposed || token !== generation) return
      if (result.run_id !== selected) throw new Error('监控快照与所选实验不匹配。')
      snapshot.value = result
      updatedAt.value = new Date().toISOString()
      error.value = ''
    } catch (failure) {
      if (!disposed && token === generation) error.value = failure?.name === 'AbortError'
        ? '监控接口响应超时，正在等待重试。' : errorText(failure)
    } finally {
      if (!disposed && token === generation) {
        if (timeout !== null) cancel(timeout)
        timeout = null
        controller = null
        busy.value = false
        timer = schedule(refresh, 5000)
      }
    }
  }
  watch(runId, () => {
    cancelWork()
    snapshot.value = null
    error.value = ''
    updatedAt.value = ''
    busy.value = false
    if (runId.value) void refresh()
  }, { immediate: true, flush: 'sync' })
  onScopeDispose(() => { disposed = true; cancelWork() })
  return { snapshot, busy, error, updatedAt, refresh }
}
