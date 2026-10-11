<script setup>
import { computed, watch } from 'vue'
import EmptyState from './EmptyState.vue'
import { useNodeMonitor, monitorStatus } from '../lib/nodeMonitor.js'
import { bytes, date, fmt, percent, roleName, seconds } from '../lib/format.js'

const props = defineProps({ runId: { type: String, required: true } })
const emit = defineEmits(['inspect', 'snapshot'])
const { snapshot, busy, error, updatedAt, refresh } = useNodeMonitor(computed(() => props.runId))
const nodes = computed(() => snapshot.value?.nodes || [])
const machines = computed(() => snapshot.value?.machines || [])
const measuredPercent = value => typeof value === 'number' && Number.isFinite(value) ? `${fmt(value, 1)}%` : '—'
const measuredSeconds = value => typeof value === 'number' && Number.isFinite(value) && value > 0 && value < 1
  ? `${fmt(value * 1000, 2)} ms` : seconds(value)
watch(snapshot, value => emit('snapshot', value), { immediate: true })
</script>

<template>
  <section class="nb-card node-monitor" aria-label="节点与主机实测指标">
    <div class="nb-card-head">
      <div class="nb-card-head-left"><h2>节点监控</h2><span class="nb-badge neutral">{{ nodes.length }} 个节点</span></div>
      <div class="node-monitor-actions"><span class="nb-card-note">{{ updatedAt ? `更新于 ${date(updatedAt)}` : '等待监控快照' }}</span><button class="nb-btn outline small" :disabled="busy" @click="refresh">刷新</button></div>
    </div>
    <div v-if="error" class="nb-notice warning monitor-notice" role="status">{{ error }}<span v-if="snapshot">当前保留最近一次快照，指标更新时间见表格。</span></div>
    <div class="nb-card-body tight">
      <p class="nb-note">节点 CPU 为节点及已核验子进程的整机归一化利用率，RSS 为这些进程的内存合计。主机 GPU、内存和磁盘在下方独立展示，不分摊到节点。</p>
      <p v-if="snapshot?.legacy" class="nb-note">这是历史实验；未采集过的节点指标显示「未记录」，不会使用当前部署状态补全。</p>
    </div>
    <div v-if="nodes.length" class="nb-table-scroll">
      <table class="nb-table node-metrics-table">
        <thead><tr><th>节点 / 角色</th><th>主机 / 地址</th><th>所属边缘</th><th>CPU</th><th>进程 RSS</th><th>本地训练集指标</th><th>指标时间</th><th>采集状态</th><th>操作</th></tr></thead>
        <tbody>
          <tr v-for="node in nodes" :key="node.id">
            <td><strong>{{ node.id }}</strong><small>{{ roleName(node.role) }}</small></td>
            <td>{{ node.machine || '—' }}<small>{{ node.host || '—' }}</small></td>
            <td>{{ node.authority || '—' }}</td>
            <td class="mono">{{ measuredPercent(node.latest?.cpu_percent) }}</td>
            <td class="mono">{{ bytes(node.latest?.rss_bytes) }}</td>
            <td v-if="node.training?.step != null"><strong>{{ percent(node.training.accuracy) }}</strong><small>交叉熵 {{ fmt(node.training.loss, 4) }} · {{ node.training.samples ?? '—' }} 样本</small><small>第 {{ node.training.round_id ?? '—' }} 轮 · epoch {{ node.training.epoch ?? '—' }} · 累计 {{ node.training.step ?? '—' }}</small><small>训练 {{ measuredSeconds(node.training.elapsed_s) }} · 评估 {{ measuredSeconds(node.training.metrics_eval_s) }}</small></td>
            <td v-else class="nb-card-note">{{ node.role === 'client' ? '未记录' : '—' }}</td>
            <td>{{ node.last_seen ? date(node.last_seen) : '未记录' }}</td>
            <td><span :class="['nb-badge', node.status === 'online' ? 'success' : node.status === 'unavailable' ? 'warning' : 'neutral']">{{ monitorStatus(node.status) }}</span><small v-if="node.error" class="node-error">{{ node.error }}</small></td>
            <td><button class="nb-btn outline small" @click="emit('inspect', { kind: 'node', id: node.id })">查看指标</button></td>
          </tr>
        </tbody>
      </table>
    </div>
    <EmptyState v-else compact icon="nodes" :title="busy ? '正在读取节点快照' : '未记录节点快照'" text="只展示本次实验保存的节点与指标。" />
    <div class="machine-heading"><h3>主机资源</h3><span class="nb-card-note">每台机器一份 · 包含其他应用负载</span></div>
    <div v-if="machines.length" class="nb-table-scroll">
      <table class="nb-table machine-metrics-table">
        <thead><tr><th>主机 / 采集节点</th><th>整机 CPU</th><th>系统内存</th><th>GPU</th><th>显存</th><th>磁盘</th><th>指标时间 / 状态</th><th>操作</th></tr></thead>
        <tbody>
          <tr v-for="machine in machines" :key="machine.id">
            <td><strong>{{ machine.id }}</strong><small>{{ machine.source_node || '主控' }}</small></td>
            <td class="mono">{{ measuredPercent(machine.latest?.cpu?.system_utilization_percent) }}</td>
            <td>{{ bytes(machine.latest?.system?.memory_used_bytes) }}<small>总计 {{ bytes(machine.latest?.system?.memory_total_bytes) }}</small></td>
            <td class="mono">{{ measuredPercent(machine.latest?.gpu?.utilization_percent) }}</td>
            <td>{{ bytes(machine.latest?.gpu?.memory_used_bytes) }}<small>总计 {{ bytes(machine.latest?.gpu?.memory_total_bytes) }}</small></td>
            <td>{{ bytes(machine.latest?.system?.disk_used_bytes) }}<small>总计 {{ bytes(machine.latest?.system?.disk_total_bytes) }}</small><small>{{ machine.latest?.system?.disk_path || '—' }}</small></td>
            <td>{{ machine.last_seen ? date(machine.last_seen) : '未记录' }}<small>{{ monitorStatus(machine.status) }}</small></td>
            <td><button class="nb-btn outline small" @click="emit('inspect', { kind: 'machine', id: machine.id })">查看指标</button></td>
          </tr>
        </tbody>
      </table>
    </div>
    <EmptyState v-else compact icon="database" title="未记录主机资源" text="缺失或不支持的硬件读数显示「—」。" />
    <div class="nb-card-body tight"><p class="nb-note">监控通信量 {{ bytes(snapshot?.summary?.bytes_sent) }}，与协议通信量分别统计。资源曲线在 TensorBoard 中查看；节点表每 5 秒读取后端缓存。</p></div>
  </section>
</template>

<style scoped>
.node-monitor { margin-bottom: 20px; }
.node-monitor-actions { display: flex; align-items: center; gap: 12px; }
.node-metrics-table { min-width: 1100px; }
.machine-metrics-table { min-width: 940px; }
.node-monitor table td { vertical-align: middle; }
.node-monitor small { display: block; margin-top: 5px; color: var(--nb-muted); line-height: 1.5; font-size: 11px; }
.node-monitor .node-error { max-width: 260px; color: #a66525; white-space: normal; overflow-wrap: anywhere; }
.machine-heading { display: flex; flex-wrap: wrap; align-items: center; gap: 12px; padding: 20px 24px 15px; border-top: 1px solid var(--nb-line-soft); }
.machine-heading h3 { margin: 0; font-size: 15px; }
.monitor-notice { margin: 16px 24px 0; display: block; }
.monitor-notice span { display: block; margin-top: 5px; }
@media (max-width: 700px) { .node-monitor .nb-card-head { align-items: flex-start; flex-direction: column; gap: 12px; } .node-monitor-actions { width: 100%; justify-content: space-between; } }
</style>
