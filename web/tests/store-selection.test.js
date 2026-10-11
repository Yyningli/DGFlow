import assert from 'node:assert/strict'
import test from 'node:test'
import { api } from '../src/lib/api.js'

const old = { run_id: 'old', created_at: '2026-10-08T01:00:00+00:00', status: 'completed' }
const middle = { run_id: 'middle', created_at: '2026-10-09T01:00:00+00:00', status: 'completed' }
const latest = { run_id: 'latest', created_at: '2026-10-10T01:00:00+00:00', status: 'completed' }
let instance = 0

async function withPoller(initial, action) {
  const previous = { window: globalThis.window, clearTimeout: globalThis.clearTimeout,
    status: api.status, listRuns: api.listRuns, getRun: api.getRun }
  const timers = new Map(), requested = []
  let timerId = 0
  const state = { active: '', runs: [], ...initial }
  globalThis.window = { setTimeout(fn, delay) { timers.set(++timerId, { fn, delay }); return timerId } }
  globalThis.clearTimeout = id => timers.delete(id)
  api.status = async () => ({ active_run_id: state.active })
  api.listRuns = async () => ({ runs: state.runs })
  api.getRun = async id => { requested.push(id); return state.runs.find(run => run.run_id === id) || { run_id: id, status: 'running' } }
  const { consoleStore: store } = await import(`../src/lib/store.js?selection-test=${++instance}`)
  const settle = async () => {
    for (let count = 0; count < 10 && ![...timers.values()].some(timer => timer.delay === 2000); count += 1) {
      await new Promise(resolve => setImmediate(resolve))
    }
    assert.equal(store.initialLoading.value, false)
  }
  const pollAgain = async () => {
    const [id, timer] = [...timers].find(([, candidate]) => candidate.delay === 2000)
    timers.delete(id)
    await timer.fn()
  }
  try {
    await action({ store, state, requested, settle, pollAgain })
  } finally {
    store.stop()
    globalThis.window = previous.window
    globalThis.clearTimeout = previous.clearTimeout
    api.status = previous.status
    api.listRuns = previous.listRuns
    api.getRun = previous.getRun
  }
}

test('initial selection uses the latest creation time rather than backend list order', async () => {
  await withPoller({ runs: [old, latest, middle] }, async ({ store, requested, settle }) => {
    store.start()
    await settle()
    assert.equal(store.selectedRunId.value, 'latest')
    assert.deepEqual(requested, ['latest'])
    assert.deepEqual(store.runs.value.map(run => run.run_id), ['old', 'latest', 'middle'])
  })
})

test('active experiment takes precedence even before it appears in the run list', async () => {
  await withPoller({ active: 'active-new', runs: [old, latest] }, async ({ store, requested, settle }) => {
    store.start()
    await settle()
    assert.equal(store.selectedRunId.value, 'active-new')
    assert.deepEqual(requested, ['active-new'])
  })
})

test('a user selection survives the initial active response and subsequent refreshes', async () => {
  await withPoller({ active: 'active-new', runs: [old, latest] }, async ({ store, state, requested, settle, pollAgain }) => {
    store.chooseRun('old')
    store.start()
    await settle()
    state.active = 'another-active'
    state.runs = [latest, old, { run_id: 'even-newer', created_at: '2026-10-11T01:00:00+00:00' }]
    await store.refreshRuns()
    await pollAgain()
    assert.equal(store.selectedRunId.value, 'old')
    assert.equal(store.currentRun.value.run_id, 'old')
    assert.deepEqual(requested, ['old', 'old'])
  })
})

test('background refresh preserves an already loaded default experiment', async () => {
  await withPoller({ runs: [old, latest] }, async ({ store, state, settle, pollAgain }) => {
    store.start()
    await settle()
    state.runs = [{ run_id: 'later', created_at: '2026-10-12T01:00:00+00:00' }, latest]
    await store.refreshRuns()
    await pollAgain()
    assert.equal(store.selectedRunId.value, 'latest')
    assert.equal(store.currentRun.value.run_id, 'latest')
  })
})

test('valid timestamps win over invalid or missing timestamps, including pre-epoch dates', async () => {
  await withPoller({ runs: [{ run_id: 'bad', created_at: 'not-a-date' }, { run_id: 'missing' },
    { run_id: 'numeric', created_at: 9999 }, { run_id: 'valid', created_at: '1960-01-01T00:00:00+00:00' }] },
  async ({ store, settle }) => {
    store.start()
    await settle()
    assert.equal(store.selectedRunId.value, 'valid')
  })
})

test('all invalid times use a stable first available experiment', async () => {
  await withPoller({ runs: [{ run_id: 'first', created_at: '' },
    { run_id: 'second', created_at: 'invalid' }] }, async ({ store, settle }) => {
    store.start()
    await settle()
    assert.equal(store.selectedRunId.value, 'first')
  })
})

test('equal creation times keep the first candidate without unstable reshuffling', async () => {
  await withPoller({ runs: [latest, { ...latest, run_id: 'same-time' }] }, async ({ store, settle }) => {
    store.start()
    await settle()
    assert.equal(store.selectedRunId.value, 'latest')
  })
})

test('an empty experiment list leaves selection empty and does not request a record', async () => {
  await withPoller({ runs: [] }, async ({ store, requested, settle }) => {
    store.start()
    await settle()
    assert.equal(store.selectedRunId.value, '')
    assert.equal(store.currentRun.value, null)
    assert.deepEqual(requested, [])
  })
})
