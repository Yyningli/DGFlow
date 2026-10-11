<script setup>
import { computed } from 'vue'
import { useTensorBoard, useTensorBoardHistory } from '../lib/tensorboard.js'
import { exportUrl } from '../lib/api.js'

const props = defineProps({
  runId: { type: String, default: '' },
  runs: { type: Array, default: () => [] },
})
const emit = defineEmits(['deploy'])
const selectedRun = computed(() => props.runId)
const { busy, frameUrl, frameLoaded, frameFailed, syncFailed, frameKey, message, confirmFrame, refresh } = useTensorBoard(selectedRun)
useTensorBoardHistory(computed(() => props.runs), selectedRun)

function loaded(event) {
  // An iframe load also fires for HTTP error pages; inspect the same-origin page.
  try {
    const document = event.target.contentDocument
    confirmFrame(Boolean(document?.querySelector('tb-web-app, tensor-board') || /TensorBoard/i.test(document?.title || '')))
  } catch { confirmFrame(false) }
}
</script>

<template>
  <section class="tensorboard-workspace" aria-label="TensorBoard 实验监控" :aria-busy="busy || (Boolean(frameUrl) && !frameLoaded && !frameFailed)">
    <iframe v-if="frameUrl" :key="frameKey" :src="frameUrl" title="TensorBoard 原生实验分析面板" referrerpolicy="same-origin" @load="loaded" @error="confirmFrame(false)"></iframe>
    <div v-if="!frameUrl || frameFailed" class="tb-overlay" role="status">
      <div class="tb-placeholder">
        <strong>{{ busy ? '正在准备 TensorBoard…' : 'TensorBoard 暂不可用' }}</strong>
        <p>{{ message }}</p>
        <div class="tb-recovery-actions">
          <button class="nb-btn primary" :disabled="busy" @click="refresh">重试加载</button>
          <a v-if="runId" class="nb-btn outline" :href="exportUrl(runId)" download>导出实验记录</a>
          <button v-else class="nb-btn outline" @click="emit('deploy')">部署实验</button>
          <a v-if="frameUrl" class="nb-btn outline" :href="frameUrl" target="_blank" rel="noopener noreferrer">新窗口打开</a>
        </div>
      </div>
    </div>
    <div v-else-if="syncFailed" class="tb-sync-warning" role="status">
      <span>{{ message }}</span><button :disabled="busy" @click="refresh">重试同步</button>
    </div>
  </section>
</template>

<style scoped>
.tensorboard-workspace { position: relative; width: 100%; height: 100%; min-width: 0; min-height: 0; overflow: hidden; background: #fff; }
.tensorboard-workspace iframe { display: block; width: 100%; height: 100%; border: 0; background: #fff; }
.tb-overlay { position: absolute; inset: 0; display: grid; place-items: center; padding: 24px; background: #fff; }
.tb-placeholder { width: min(100%, 560px); text-align: center; color: var(--nb-muted); font-size: 13px; }
.tb-placeholder strong { color: var(--nb-text); font-size: 18px; }
.tb-placeholder p { margin-top: 12px; overflow-wrap: anywhere; }
.tb-recovery-actions { display: flex; flex-wrap: wrap; justify-content: center; gap: 10px; margin-top: 20px; }
.tb-sync-warning { position: absolute; bottom: 12px; right: 12px; max-width: min(520px, calc(100% - 24px)); display: flex; align-items: center; gap: 12px; padding: 10px 14px; background: #fff7e7; border: 1px solid #e7c58a; border-radius: 5px; color: #97601c; box-shadow: var(--nb-shadow); font-size: 12px; }
.tb-sync-warning span { min-width: 0; overflow-wrap: anywhere; }
.tb-sync-warning button { border: 0; background: transparent; color: inherit; flex-shrink: 0; text-decoration: underline; }
</style>
