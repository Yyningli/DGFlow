<script setup>
import { computed } from 'vue'
import TensorBoardPanel from '../components/TensorBoardPanel.vue'
import { consoleStore as store } from '../lib/store.js'

const emit = defineEmits(['navigate'])
const runId = computed(() => store.currentRun.value?.run_id || store.selectedRunId.value || '')
</script>

<template>
  <TensorBoardPanel :run-id="runId" :runs="store.runs.value" @deploy="emit('navigate', 'deploy')" />
</template>
