<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import NebulaHeader from './components/NebulaHeader.vue'
import NebulaFooter from './components/NebulaFooter.vue'
import PageBackdrop from './components/PageBackdrop.vue'
import LabIcon from './components/LabIcon.vue'
import Home from './views/Home.vue'
import Dashboard from './views/Dashboard.vue'
import Deploy from './views/Deploy.vue'
import Monitor from './views/Monitor.vue'
import Validation from './views/Validation.vue'
import Comparison from './views/Comparison.vue'
import { consoleStore as store } from './lib/store'
import { clockTime } from './lib/format'

const VIEWS = ['home', 'dashboard', 'deploy', 'monitor', 'validation', 'comparison']
const initial = window.location.hash.slice(1)
const page = ref(VIEWS.includes(initial) ? initial : 'home')

const activeRunId = computed(() => store.status.value?.active_run_id || '')
const lastUpdated = computed(() => (store.lastUpdated.value ? clockTime(store.lastUpdated.value) : ''))
const onlineNodes = computed(() => store.nodes.value.filter(n => n.status === 'online').length)

function navigate(id) {
  if (!VIEWS.includes(id)) return
  page.value = id
  window.location.hash = id
  window.scrollTo({ top: 0, behavior: 'smooth' })
}

function syncHash() {
  const id = window.location.hash.slice(1)
  if (VIEWS.includes(id)) page.value = id
}

onMounted(() => {
  window.addEventListener('hashchange', syncHash)
  store.start()
})
onBeforeUnmount(() => {
  window.removeEventListener('hashchange', syncHash)
  store.stop()
})
</script>

<template>
  <PageBackdrop />
  <div :class="['nb-app', { 'nb-app-monitor': page === 'monitor' }]">
    <NebulaHeader
      :page="page"
      :connected="store.connected.value"
      :active-run-id="activeRunId"
      :node-count="onlineNodes"
      @navigate="navigate"
    />

    <main class="nb-main">
      <div class="nb-container wide">
        <div v-if="page !== 'monitor' && !store.connected.value && !store.initialLoading.value" class="nb-notice warning" style="margin-top:24px">
          <LabIcon name="alert" :size="18" />
          <div>
            <strong>暂时无法连接实验引擎</strong>
            <p>{{ store.connectionError.value }} 界面每 2 秒自动重试；已有结果保留为上次读取状态。</p>
          </div>
        </div>

        <Home v-if="page === 'home'" @navigate="navigate" />
        <Dashboard v-else-if="page === 'dashboard'" @navigate="navigate" @open="navigate" />
        <Deploy v-else-if="page === 'deploy'" @deployed="navigate('monitor')" />
        <Monitor v-else-if="page === 'monitor'" @navigate="navigate" />
        <Validation v-else-if="page === 'validation'" @navigate="navigate" />
        <Comparison v-else-if="page === 'comparison'" @navigate="navigate" />
      </div>
    </main>

    <NebulaFooter
      v-if="page !== 'monitor'"
      :connected="store.connected.value"
      :version="store.status.value?.version"
      :last-updated="lastUpdated"
      @navigate="navigate"
    />
  </div>
</template>
