import assert from 'node:assert/strict'
import test from 'node:test'
import { effectScope, ref } from 'vue'

const module = await import('../src/lib/tensorboard.js').catch(error => {
  if (error.code === 'ERR_MODULE_NOT_FOUND') return {}
  throw error
})
const flush = () => new Promise(resolve => setImmediate(resolve))
const response = data => ({ ok: true, text: async () => JSON.stringify(data) })

function session(id = 'run/a') {
  assert.equal(typeof module.useTensorBoard, 'function', 'TensorBoard lifecycle is implemented')
  const scope = effectScope(), runId = ref(id), scheduled = new Map()
  let next = 0
  const state = scope.run(() => module.useTensorBoard(runId, {
    schedule: (fn, delay) => { scheduled.set(++next, { fn, delay }); return next },
    cancel: id => scheduled.delete(id),
  }))
  return { state, runId, scope, scheduled }
}

test('TensorBoard opens automatically, filters the selected run and waits for iframe confirmation', async () => {
  const original = globalThis.fetch, calls = []
  globalThis.fetch = async (url, options) => { calls.push({ url, options }); return response({ available: true, url: '/tensorboard/' }) }
  let current
  try {
    current = session()
    await flush()
    assert.deepEqual(calls.map(call => call.url), ['/api/tensorboard/status', '/api/runs/run%2Fa/tensorboard'])
    assert.equal(calls[1].options.method, 'POST')
    const filter = new RegExp(new URL(current.state.frameUrl.value, 'http://localhost').searchParams.get('runFilter'))
    assert.ok(filter.test('run/a'))
    assert.ok(filter.test('run/a\\nodes\\client1'))
    assert.equal(filter.test('run/a-other'), false)
    assert.equal(new URL(current.state.frameUrl.value, 'http://localhost').searchParams.get('smoothing'), '0')
    assert.equal(current.state.frameLoaded.value, false)
    current.state.confirmFrame(true)
    assert.equal(current.state.frameLoaded.value, true)
    assert.equal([...current.scheduled.values()][0].delay, 10000)
    const oldKey = current.state.frameKey.value
    await current.state.refresh()
    assert.ok(current.state.frameKey.value > oldKey)
    assert.equal(current.state.frameLoaded.value, false)
  } finally { current?.scope.stop(); globalThis.fetch = original }
  assert.equal(current.scheduled.size, 0)
})

test('unavailable TensorBoard presents the backend reason without creating an iframe', async () => {
  const original = globalThis.fetch, calls = []
  globalThis.fetch = async url => { calls.push(url); return response({ available: false, url: null, reason: '缺少 tensorboard 依赖' }) }
  let current
  try {
    current = session(); current.state.expanded.value = true
    await flush()
    assert.equal(calls.length, 2)
    assert.equal(current.state.frameUrl.value, '')
    assert.match(current.state.message.value, /缺少 tensorboard/)
    assert.equal(current.state.frameLoaded.value, false)
  } finally { current?.scope.stop(); globalThis.fetch = original }
})

test('a stale unavailable status still allows a sync to recover the TensorBoard service', async () => {
  const original = globalThis.fetch
  globalThis.fetch = async url => response(url.endsWith('/status')
    ? { available: false, url: '/tensorboard/', reason: '之前的日志导出失败' }
    : { available: true, url: '/tensorboard/' })
  let current
  try {
    current = session(); current.state.expanded.value = true
    await flush()
    assert.match(current.state.frameUrl.value, /^\/tensorboard\/\?smoothing=0&runFilter=/)
    assert.equal(current.state.syncedRun.value, 'run/a')
    assert.equal(current.state.frameLoaded.value, false)
  } finally { current?.scope.stop(); globalThis.fetch = original }
})

test('switching run or disposing cancels old work and ignores late responses', async () => {
  const original = globalThis.fetch
  let resolveOld, oldSignal, current
  globalThis.fetch = async (url, options) => {
    if (url.includes('/old/')) { oldSignal = options.signal; return await new Promise(resolve => { resolveOld = resolve }) }
    return response({ available: true, url: '/tensorboard/' })
  }
  try {
    current = session('old'); current.state.expanded.value = true
    await flush()
    current.runId.value = 'new'
    await flush()
    assert.equal(oldSignal.aborted, true)
    assert.equal(current.state.syncedRun.value, 'new')
    resolveOld(response({ available: false, reason: '旧实验错误' }))
    await flush()
    assert.equal(current.state.syncedRun.value, 'new')
    assert.match(current.state.frameUrl.value, /^\/tensorboard\/\?smoothing=0&runFilter=/)
    current.state.expanded.value = false
    assert.equal(current.scheduled.size, 0)
    assert.equal(current.state.frameUrl.value, '')
  } finally { current?.scope.stop(); globalThis.fetch = original }
})

test('only the same-origin TensorBoard route may be embedded', () => {
  assert.equal(typeof module.tensorBoardUrl, 'function')
  assert.equal(module.tensorBoardUrl('/tensorboard/'), '/tensorboard/')
  for (const value of ['https://other.test/tensorboard/', '//other.test/', '/api/runs', '/tensorboard/../api', '/tensorboard/?run=made-up', null]) {
    assert.equal(module.tensorBoardUrl(value), '')
  }
})

test('iframe errors stay explicit during background log sync and a refresh retries loading', async () => {
  const original = globalThis.fetch
  globalThis.fetch = async () => response({ available: true, url: '/tensorboard/' })
  let current
  try {
    current = session(); current.state.expanded.value = true
    await flush()
    current.state.confirmFrame(false)
    assert.match(current.state.message.value, /加载失败/)
    const scheduled = [...current.scheduled.values()][0]
    await scheduled.fn()
    assert.match(current.state.message.value, /加载失败/)
    assert.equal(current.state.frameLoaded.value, false)
    await current.state.refresh()
    assert.equal(current.state.frameFailed.value, false)
    assert.match(current.state.message.value, /正在加载/)
  } finally { current?.scope.stop(); globalThis.fetch = original }
})

test('a late iframe load preserves the reason for a failed background synchronization', async () => {
  const original = globalThis.fetch
  globalThis.fetch = async () => response({ available: true, url: '/tensorboard/' })
  let current
  try {
    current = session()
    await flush()
    globalThis.fetch = async () => { throw new Error('日志写入失败') }
    await [...current.scheduled.values()].find(timer => timer.delay === 10000).fn()
    assert.equal(current.state.syncFailed.value, true)
    current.state.confirmFrame(true)
    assert.equal(current.state.frameLoaded.value, true)
    assert.match(current.state.message.value, /日志写入失败/)
  } finally { current?.scope.stop(); globalThis.fetch = original }
})
