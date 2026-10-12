"""Best-effort, TensorFlow-free scalar events for live and archived experiments.

Files are append-only; exporters share the newest intact file under a short
OS lock that serializes deduplication across processes. No writer threads or open
event handles survive a call; the bounded cache is only an optimization.
"""
from __future__ import annotations

import importlib.util
import math
import os
import re
import struct
import sys
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from dgfl.crypto.gpu_profile import COUNTS as GPU_COUNTS
from dgfl.crypto.gpu_profile import KERNELS as GPU_KERNELS
from dgfl.crypto.gpu_profile import SECONDS as GPU_SECONDS
from dgfl.experiments.hardware import CPU_FIELDS, GPU_FIELDS

URL = '/tensorboard/'
_RUN_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z')
_ACTOR = re.compile(r'(?:coordinator|(?:client|authority|aggregator)[1-9][0-9]*)\Z')
_RESERVED = {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(10)), *(f'LPT{i}' for i in range(10))}
_COMBINE_SECONDS = ('combine_numerators_s', 'combine_dkg_constants_s', 'combine_proof_verification_s',
                    'combine_interpolation_s', 'combine_context_materials_s', 'combine_cloud_E_s',
                    'combine_total_s', 'combine_cpu_s')
_COMBINE_COUNTS = ('combine_commitment_cache_hits', 'combine_commitment_cache_misses',
                   'combine_dkg_transcript_reused')
_STAGE_FIELDS = ('dkg_s', 'client_stage_s', 'training_wall_sum_s', 'encryption_wall_sum_s', 'proof_wall_sum_s',
                 'validation_key_s', 'authorization_s', 'validation_s', 'proof_metrics_s', 'aggregate_key_s',
                 'partial_decryption_s', 'aggregate_verification_s', 'combine_and_confirmation_s',
                 'combine_metrics_s', 'aggregation_s', *_COMBINE_SECONDS,
                 *(f'{name}_max' for name in _COMBINE_SECONDS))
# A bounded dashboard: node IDs belong in run names, never in chart tags.
# Keep detailed measurements in the append-only event files and result export;
# the embedded DataProvider applies this same policy to old and new experiments.
KEY_SCALAR_TAGS = frozenset({
    'evaluation/accuracy', 'evaluation/cross_entropy',
    'training/accuracy', 'training/cross_entropy', 'training/epoch_s',
    'clients/accepted', 'clients/rejected', 'clients/collateral', 'clients/failures',
    'clients/attack_accepted', 'clients/honest_rejected',
    'protocol/stages/dkg_s', 'protocol/stages/client_stage_s', 'protocol/stages/validation_s',
    'protocol/stages/aggregate_key_s', 'protocol/stages/partial_decryption_s',
    'protocol/stages/combine_and_confirmation_s',
    'combine/combine_total_s', 'combine/combine_proof_verification_s', 'combine/combine_numerators_s',
    'proof/generation/proof_gen_s', 'proof/verification/proof_verify_total_s',
    'proof/verification/inner_product_total_s',
    'gpu_compute/totals/host_wall_ms', 'gpu_compute/totals/kernel_ms',
    'gpu_compute/totals/upload_ms', 'gpu_compute/totals/download_ms',
    'gpu_compute/totals/sync_ms', 'gpu_compute/totals/scheduler_wait_ms',
    'system/cpu_percent', 'system/memory_percent', 'system/gpu_utilization_percent',
    'system/gpu_memory_used_bytes', 'processes/cpu_percent', 'processes/rss_bytes',
    'communication/round_bytes', 'communication/task_total_bytes',
    'round/duration_s', 'task/duration_s', 'monitoring/metrics_eval_s',
})


def is_key_scalar(tag):
    """Exact membership prevents per-kernel/client diagnostic chart growth."""
    return tag in KEY_SCALAR_TAGS


_FIELD_LABELS = {
    'accuracy': ('准确率', '比例 0–1'), 'cross_entropy': ('交叉熵损失', '无量纲'),
    'samples': ('样本数量', '个'), 'duration_s': ('总墙钟耗时', '秒'),
    'controller_cpu_s': ('主控进程 CPU 时间', 'CPU 秒'), 'completed_rounds': ('已完成轮次', '轮'),
    'round_bytes': ('本轮协议通信量', '字节'),
    'cumulative_round_bytes': ('已完成轮次累计协议通信量', '字节'),
    'task_total_bytes': ('整个任务协议通信量', '字节'),
    'accepted': ('通过筛选客户端', '个'), 'rejected': ('拒绝客户端', '个'),
    'collateral': ('同组连带剔除客户端', '个'), 'failures': ('未成功返回客户端', '个'),
    'attack_accepted': ('被接纳的恶意客户端', '个'), 'attack_submitted': ('已提交恶意客户端', '个'),
    'honest_rejected': ('被拒绝的诚实客户端', '个'), 'honest_total': ('诚实客户端总数', '个'),
    'reference_norm_squared': ('全局量化模型平方范数', '量化坐标平方'),
    'dkg_s': ('分布式建钥阶段', '秒'), 'client_stage_s': ('客户端提交阶段墙钟', '秒'),
    'training_wall_sum_s': ('各客户端训练耗时之和', '秒'),
    'encryption_wall_sum_s': ('各客户端模型加密耗时之和', '秒'),
    'proof_wall_sum_s': ('各客户端证明生成耗时之和', '秒'),
    'validation_key_s': ('验证密钥派发', '秒'), 'authorization_s': ('验证与筛选授权', '秒'),
    'validation_s': ('验证阶段墙钟', '秒'), 'proof_metrics_s': ('证明诊断收集', '秒'),
    'aggregate_key_s': ('聚合解密密钥派发', '秒'), 'partial_decryption_s': ('云节点部分解密', '秒'),
    'aggregate_verification_s': ('部分解密材料验证', '秒'),
    'combine_and_confirmation_s': ('聚合合并与独立确认墙钟', '秒'),
    'combine_metrics_s': ('合并诊断收集', '秒'), 'aggregation_s': ('聚合阶段墙钟', '秒'),
    'combine_numerators_s': ('合并分子计算', '秒'), 'combine_dkg_constants_s': ('DKG 常数派生', '秒'),
    'combine_proof_verification_s': ('部分解密证明验证', '秒'),
    'combine_interpolation_s': ('门限插值', '秒'), 'combine_context_materials_s': ('上下文与材料核验', '秒'),
    'combine_cloud_E_s': ('云节点解密像核验', '秒'), 'combine_total_s': ('本进程合并总耗时', '秒'),
    'combine_cpu_s': ('本进程合并 CPU 时间', 'CPU 秒'),
    'combine_commitment_cache_hits': ('承诺缓存命中', '次'),
    'combine_commitment_cache_misses': ('承诺缓存未命中', '次'),
    'combine_dkg_transcript_reused': ('复用本地 DKG 转录', '0 或 1'),
    'proof_gen_s': ('证明生成', '秒'), 'parameters_load_s': ('可信参数加载', '秒'),
    'proof_verify_s': ('证明验证', '秒'), 'inner_product_s': ('内积计算', '秒'),
    'native_threads': ('原生计算线程', '个'),
    'cpu_percent': ('CPU 利用率', '%'), 'gpu_utilization_percent': ('GPU 利用率', '%'),
    'gpu_memory_used_bytes': ('已用 GPU 显存', '字节'), 'gpu_memory_total_bytes': ('GPU 显存总量', '字节'),
    'gpu_memory_percent': ('GPU 显存占用', '%'), 'rss_bytes': ('进程驻留内存', '字节'),
    'memory_used_bytes': ('已用内存', '字节'), 'memory_total_bytes': ('内存总量', '字节'),
    'memory_percent': ('内存占用', '%'), 'disk_used_bytes': ('文件系统已用空间', '字节'),
    'disk_total_bytes': ('文件系统总容量', '字节'), 'disk_percent': ('文件系统占用', '%'),
    'processor_performance_percent': ('CPU 性能计数器', '%'),
    'nominal_frequency_mhz': ('CPU 标称频率', 'MHz'), 'effective_frequency_mhz': ('CPU 有效平均频率', 'MHz'),
    'reported_frequency_mhz': ('CPU 上报频率', 'MHz'),
    'os_reported_current_frequency_mhz': ('系统上报 CPU 当前频率', 'MHz'), 'frequency_limit_mhz': ('CPU 频率上限', 'MHz'),
    'graphics_clock_mhz': ('GPU 图形频率', 'MHz'), 'sm_clock_mhz': ('GPU SM 频率', 'MHz'),
    'memory_clock_mhz': ('GPU 显存频率', 'MHz'), 'utilization_percent': ('GPU 利用率', '%'),
    'memory_utilization_percent': ('GPU 显存控制器利用率', '%'), 'power_w': ('GPU 功耗', '瓦'),
    'temperature_c': ('GPU 温度', '℃'), 'pstate': ('GPU 性能状态', 'P 状态编号'),
    'throttle_reason_mask': ('GPU 降频原因位掩码', '位掩码'),
    'local_epoch': ('本地 epoch 编号', 'epoch'), 'global_round': ('所属全局轮次', '轮'),
    'local_epochs': ('本轮本地训练 epoch 数', 'epoch'), 'epoch_s': ('本地 epoch 优化器训练', '秒'),
    'metrics_eval_s': ('监控评估额外耗时', '秒'), 'metrics_callback_s': ('监控回调额外耗时', '秒'),
    'proof_verify_total_s': ('本边缘各客户端证明验证耗时之和', '秒'),
    'inner_product_total_s': ('本边缘各客户端内积计算耗时之和', '秒'),
    'training_s': ('训练及原有量化攻击处理耗时', '秒'), 'clients_recorded': ('已记录训练开销客户端', '个'),
    'batches': ('计算批次', '批'), 'rows': ('计算行数', '行'), 'failed_batches': ('失败批次', '批'),
    'upload_bytes': ('主机上传设备数据', '字节'), 'download_bytes': ('设备回传主机数据', '字节'),
    'allocations': ('设备内存分配', '次'), 'allocation_reuses': ('设备内存复用', '次'), 'frees': ('设备内存释放', '次'),
    'failed_wall_ms': ('失败调用墙钟耗时', '毫秒'), 'host_wall_ms': ('成功调用主机墙钟耗时', '毫秒'),
    'lock_wait_ms': ('主机锁等待', '毫秒'), 'scheduler_wait_ms': ('调度等待', '毫秒'),
    'setup_ms': ('主机准备', '毫秒'), 'allocation_ms': ('设备内存分配耗时', '毫秒'),
    'upload_ms': ('上传设备耗时', '毫秒'), 'output_clear_ms': ('设备输出清零', '毫秒'),
    'launch_ms': ('内核发射', '毫秒'), 'sync_ms': ('主机等待设备完成', '毫秒'),
    'download_ms': ('设备回传耗时', '毫秒'), 'kernel_ms': ('CUDA 内核事件耗时', '毫秒'),
    'reserved_bytes': ('设备内存池当前保留', '字节'), 'peak_reserved_bytes': ('设备内存池峰值', '字节'),
    'budget_bytes': ('设备内存池预算', '字节'), 'slots': ('设备内存池槽位', '个'),
    'block_size': ('CUDA 线程块大小', '线程'), 'chunk_size': ('CUDA 分块行数', '行'),
    'scheduler_slots': ('设备调度槽位', '个'), 'multiprocessor_count': ('GPU SM 数量', '个'),
    'max_threads_per_multiprocessor': ('每 SM 线程上限', '线程'),
    'registers_per_multiprocessor': ('每 SM 寄存器数', '个'),
    'max_threads_per_block': ('内核线程块上限', '线程'), 'static_shared_bytes': ('内核静态共享内存', '字节'),
    'local_bytes_per_thread': ('每线程本地内存', '字节'), 'num_registers': ('每线程寄存器', '个'),
}
_GROUP_LABELS = {'evaluation': '全局测试集', 'round': '轮次执行', 'task': '整个任务',
                 'communication': '协议通信', 'clients': '客户端筛选', 'model': '全局模型',
                 'protocol': '协议', 'stages': '阶段耗时', 'proof': '证明', 'verification': '客户端证明核验',
                 'generation': '客户端证明生成', 'combine': '聚合合并', 'gpu_compute': 'GPU 密码计算',
                 'task_total': '整个任务累计', 'totals': '本进程合计', 'kernels': '逐内核',
                 'pool': '设备内存池', 'configuration': '计算配置', 'kernel_resources': '内核资源',
                 'device_resources': '设备资源', 'system': '主机', 'cpu': 'CPU', 'gpu': 'GPU',
                 'processes': '实验进程', 'training': '本地训练集', 'training_round': '本地训练轮次',
                 'monitoring': '监控额外开销'}


def scalar_label_catalog():
    """Detached labels for the native UI adapter; never a source of metrics."""
    return {'fields': {field: {'label': label, 'unit': unit} for field, (label, unit) in _FIELD_LABELS.items()},
            'groups': dict(_GROUP_LABELS), 'kernels': sorted(GPU_KERNELS)}


def metric_metadata(tag):
    """Human labels are metadata; stable ASCII tags remain compatible."""
    parts = tag.split('/')
    field = parts[-1]
    maximum = field.endswith('_max')
    label, unit = _FIELD_LABELS.get(field[:-4] if maximum else field, (field, '实测值'))
    if maximum: label += '（各边缘进程最大值）'
    prefix = ' / '.join(_GROUP_LABELS.get(part, part) for part in parts[:-1])
    axis = '资源采样序号；实际采样时间可切换 Wall time 查看'
    if parts[0] in ('evaluation', 'round', 'task', 'communication', 'clients', 'model', 'protocol',
                    'proof', 'combine', 'gpu_compute', 'training_round', 'monitoring'):
        axis = '全局轮次'
    elif parts[0] == 'training':
        axis = '累计本地 epoch，(全局轮次−1)×local_epochs+本地 epoch'
    notes = ['仅导出已记录的有限非负实测值；缺失读数不补零。']
    if parts[0] == 'training': notes.append('评估客户端实际训练分区；模型尚未攻击变换或量化，区别于全局测试集。')
    if parts[0] == 'evaluation':
        notes.append('根记录为全局测试集；nodes/clientN 为每轮最后本地 epoch 的客户端训练集指标。'
                     '横轴均为全局轮次，客户端模型尚未攻击变换或量化，两类数据集口径不同。')
    if parts[0] in ('protocol', 'combine'):
        notes.append('保留原始独立计时；父阶段、子阶段及并行进程耗时有重叠，不可直接相加成墙钟。')
    if field.endswith('wall_sum_s'):
        notes.append('各客户端调用时间之和，并行时可超过客户端阶段墙钟。')
    if field in ('proof_verify_total_s', 'inner_product_total_s'):
        notes.append('本边缘本轮已验证客户端调用耗时之和；仅全部计时齐全时记录，不代表验证阶段墙钟。')
    if parts[0] == 'gpu_compute':
        notes.append('CUDA kernel 与主机 sync/host wall 耗时重叠，不可相加；计数属于当前进程。')
    if 'task_total' in parts or parts[0] == 'task' or field == 'task_total_bytes':
        notes.append('仅任务终态最后已完成轮次记录一个整个任务累计值，包含启动及收尾。')
    if field == 'cumulative_round_bytes': notes.append('仅累计连续且有实测通信量的已完成轮次，不含轮次外启动通信。')
    if parts[0] in ('system', 'cpu', 'gpu'):
        notes.append('整台所在机器的读数，包含其他应用；磁盘为 runtime 所在文件系统。')
    if tag == 'processes/cpu_percent': notes.append('已按整机逻辑 CPU 核数归一化。')
    return f'{prefix} / {label}（{unit}）', f'单位：{unit}。横轴：{axis}。' + ''.join(notes)


def _number(value):
    try:
        return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None
    except OverflowError:
        return None


def _measured(metrics, tag, value, *, factor=1):
    value = _number(value)
    if value is not None:
        scaled = _number(value * factor)
        if scaled is not None and scaled <= float.fromhex('0x1.fffffep+127'):
            metrics[tag] = scaled


def _round_metrics(row):
    metrics = {}
    for field in ('duration_s', 'controller_cpu_s'):
        _measured(metrics, f'round/{field}', row.get(field))
    _measured(metrics, 'communication/round_bytes', row.get('bytes_sent'))
    for field in ('accepted', 'rejected', 'collateral'):
        members = row.get(f'{field}_clients')
        if isinstance(members, list):
            _measured(metrics, f'clients/{field}', len(members))
    failures = row.get('client_failures')
    if isinstance(failures, dict):
        _measured(metrics, 'clients/failures', len(failures))
    for field in ('attack_accepted', 'attack_submitted', 'honest_rejected', 'honest_total'):
        _measured(metrics, f'clients/{field}', row.get(field))
    _measured(metrics, 'model/reference_norm_squared', row.get('reference_norm_squared'))
    stages = row.get('stage_times')
    if isinstance(stages, dict):
        for field in _STAGE_FIELDS:
            _measured(metrics, f'protocol/stages/{field}', stages.get(field))
    overhead = row.get('monitoring_overhead')
    if isinstance(overhead, dict):
        for field in ('metrics_eval_s', 'metrics_callback_s', 'training_s', 'clients_recorded'):
            _measured(metrics, f'monitoring/{field}', overhead.get(field))
    elif 'metrics_eval_s' in row:
        _measured(metrics, 'monitoring/metrics_eval_s', row['metrics_eval_s'])
    return metrics


def _actor_scope(actor):
    if not isinstance(actor, str) or not _ACTOR.fullmatch(actor): return False
    return None if actor == 'coordinator' else ('nodes', actor)


def _gpu_metrics(profile, *, prefix='gpu_compute'):
    """Use the existing bounded GPU evidence schema, never arbitrary paths."""
    if not isinstance(profile, dict): return {}
    result = {}

    def counters(source, path):
        if not isinstance(source, dict): return
        for field in GPU_COUNTS:
            _measured(result, f'{path}/{field}', source.get(field))
        for field in GPU_SECONDS:
            _measured(result, f'{path}/{field.removesuffix("_seconds")}_ms', source.get(field), factor=1000)

    counters(profile.get('totals'), f'{prefix}/totals')
    kernels = profile.get('by_kernel')
    if isinstance(kernels, dict):
        for kernel in sorted(GPU_KERNELS & kernels.keys()):
            counters(kernels[kernel], f'{prefix}/kernels/{kernel}')
    fields = (
        ('pool', ('reserved_bytes', 'peak_reserved_bytes', 'budget_bytes', 'slots')),
        ('configuration', ('block_size', 'chunk_size', 'scheduler_slots')),
        ('device_resources', ('multiprocessor_count', 'max_threads_per_multiprocessor', 'registers_per_multiprocessor')),
    )
    for group, names in fields:
        source = profile.get(group)
        if isinstance(source, dict):
            for field in names:
                _measured(result, f'{prefix}/{group}/{field}', source.get(field))
    kernels = profile.get('kernel_resources')
    if isinstance(kernels, dict):
        for kernel in sorted(GPU_KERNELS & kernels.keys()):
            source = kernels[kernel]
            if isinstance(source, dict):
                for field in ('max_threads_per_block', 'static_shared_bytes', 'local_bytes_per_thread', 'num_registers'):
                    _measured(result, f'{prefix}/kernel_resources/{kernel}/{field}', source.get(field))
    return result


def _actor_metrics(row):
    """Per-process round evidence shares each node's existing TensorBoard run."""
    output = {}

    def actor_metrics(actor):
        scope = _actor_scope(actor)
        return output.setdefault(scope, {}) if scope is not False else {}

    generation = row.get('proof_generation_metrics')
    if isinstance(generation, dict):
        for actor, measured in generation.items():
            if not isinstance(measured, dict): continue
            target = actor_metrics(actor)
            for field in ('proof_gen_s', 'parameters_load_s'):
                _measured(target, f'proof/generation/{field}', measured.get(field))
    verification = row.get('proof_verification_metrics')
    if isinstance(verification, dict):
        for actor, measured in verification.items():
            if not isinstance(measured, dict) or not isinstance(measured.get('verifications'), dict): continue
            target = actor_metrics(actor)
            verified = []
            for client, timings in measured['verifications'].items():
                if not isinstance(client, str) or not re.fullmatch(r'client[1-9][0-9]*', client): continue
                verified.append(timings)
                if not isinstance(timings, dict): continue
                for field in ('proof_verify_s', 'parameters_load_s', 'inner_product_s', 'native_threads'):
                    _measured(target, f'proof/verification/{client}/{field}', timings.get(field))
            for field in ('proof_verify_s', 'inner_product_s'):
                values = [_number(timings.get(field)) if isinstance(timings, dict) else None
                          for timings in verified]
                if values and all(value is not None for value in values):
                    try:
                        total = math.fsum(values)
                    except OverflowError:
                        continue
                    _measured(target, f'proof/verification/{field.removesuffix("_s")}_total_s', total)
    combine = row.get('combine_metrics')
    if isinstance(combine, dict):
        for actor, measured in combine.items():
            if not isinstance(measured, dict) or not isinstance(measured.get('combine_timings'), dict): continue
            target = actor_metrics(actor)
            timings = measured['combine_timings']
            for field in (*_COMBINE_SECONDS, *_COMBINE_COUNTS):
                _measured(target, f'combine/{field}', timings.get(field))
            target.update(_gpu_metrics(timings.get('gpu_profile')))
    return {scope: metrics for scope, metrics in output.items() if metrics}


def _timestamp(value):
    measured = _number(value)
    if measured is not None and measured > 0:
        return float(measured)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
            if parsed.tzinfo is not None:
                return parsed.timestamp()
        except (ValueError, OverflowError):
            pass
    return time.time()


def _no_link(path):
    # is_symlink alone misses Windows junctions (also reparse points).
    if path.is_symlink() or (path.exists() and getattr(path.lstat(), 'st_file_attributes', 0) & 0x400):
        raise ValueError('TensorBoard 日志路径不允许符号链接或目录联接')


def _safe_open(path, flags):
    _no_link(path)
    return os.open(path, flags | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_BINARY', 0), 0o600)


@contextmanager
def _run_lock(directory):
    descriptor = _safe_open(directory / '.export.lock', os.O_RDWR | os.O_CREAT)
    with os.fdopen(descriptor, 'r+b', buffering=0) as stream:
        if os.fstat(stream.fileno()).st_size == 0:
            stream.write(b'\0')
        deadline = time.monotonic() + 1.
        while True:
            try:
                if sys.platform == 'win32':
                    import msvcrt
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise OSError('TensorBoard 日志正在由其他进程写入，请重试') from None
                time.sleep(.01)
        try:
            yield
        finally:
            if sys.platform == 'win32':
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _events(path, event_type, offset=0):
    """Read valid TFRecord frames without importing a compute runtime.

    A crashed session may leave a partial final frame. Preserve that file and
    resume into a new file; never append after an unreadable frame.
    """
    from tensorboard.compat.tensorflow_stub.pywrap_tensorflow import masked_crc32c
    with os.fdopen(_safe_open(path, os.O_RDONLY), 'rb') as stream:
        stream.seek(offset)
        while header := stream.read(12):
            if len(header) != 12 or masked_crc32c(header[:8]) != struct.unpack('<I', header[8:])[0]:
                raise EOFError('incomplete event record header')
            length = struct.unpack('<Q', header[:8])[0]
            if length > 16 * 1024 * 1024:
                raise EOFError('invalid event record length')
            data, footer = stream.read(length), stream.read(4)
            if len(data) != length or len(footer) != 4 or masked_crc32c(data) != struct.unpack('<I', footer)[0]:
                raise EOFError('incomplete event record payload')
            yield event_type.FromString(data)


def _event_snapshot(path):
    _no_link(path)
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _event_replaced(previous, current):
    # An external writer may rewrite the prefix and grow the same inode. Only
    # our own writes (whose final snapshot is saved below) are known appends.
    return current is None or current != previous


class TensorBoardLogs:
    def __init__(self, runtime, *, max_cached_runs=256):
        self.runtime = Path(runtime).absolute()
        self.logdir = self.runtime / 'tensorboard'
        self._lock = threading.RLock()
        self._sync_lock = threading.RLock()
        self._states = OrderedDict()
        self._replay_offsets = OrderedDict()
        self._replay_pending = {}
        self._max_pending_epochs = 4096
        self._max_cached_runs = max(1, int(max_cached_runs))
        self._last_error = None

    def status(self):
        try:
            available = importlib.util.find_spec('tensorboard') is not None
        except (ImportError, ValueError):
            available = False
        result = {'available': available, 'url': URL}
        if not available:
            result['reason'] = '未安装 TensorBoard，请安装项目依赖后重试。'
        if self._last_error:
            # Availability is a capability check, not a sticky failure latch:
            # the next export must be able to retry after a disk/lock failure.
            result['last_error'] = self._last_error
        return result

    def ensure_logdir(self):
        for parent in reversed((self.logdir, *self.logdir.parents)):
            _no_link(parent)
        self.logdir.mkdir(parents=True, exist_ok=True)
        return self.logdir

    def validate_tree(self):
        self.ensure_logdir()
        for directory, folders, files in os.walk(self.logdir, followlinks=False):
            for name in (*folders, *files):
                _no_link(Path(directory) / name)

    def _directory(self, run_id, scope=None):
        if (not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id) or run_id.endswith('.')
                or run_id.split('.')[0].upper() in _RESERVED):
            raise ValueError('无效的 TensorBoard 实验标识')
        self.ensure_logdir()
        directory = self.logdir / run_id
        _no_link(directory)
        directory.mkdir(exist_ok=True)
        if scope is not None:
            if not isinstance(scope, tuple) or len(scope) != 2 or scope[0] not in ('nodes', 'machines'):
                raise ValueError('无效的 TensorBoard 指标范围')
            for part in scope:
                if (not isinstance(part, str) or not _RUN_ID.fullmatch(part) or part.endswith('.')
                        or part.split('.')[0].upper() in _RESERVED):
                    raise ValueError('无效的 TensorBoard 指标标识')
                directory /= part
                _no_link(directory)
                directory.mkdir(exist_ok=True)
        return directory

    def _error(self, exc, run_id=None):
        if isinstance(exc, ImportError):
            reason = 'TensorBoard 依赖不可用，请安装项目依赖后重试。'
        else:
            reason = f'TensorBoard 导出失败：{type(exc).__name__}: {exc}'
            reason = reason.replace(str(self.runtime), '[runtime]')[:400]
        self._last_error = reason
        return {'available': False, 'url': URL, 'run_id': run_id, 'reason': reason}

    def _forget_replay(self, run_id):
        self._replay_offsets.pop(run_id, None)
        self._replay_pending.pop(run_id, None)

    def _check_replay_files(self, run_id):
        # A consumed journal may have no new batches to write. Validate the
        # files its cursor relied on before deciding to skip that journal.
        with self._lock:
            if run_id not in self._replay_offsets:
                return
            for key, state in self._states.items():
                if key == run_id:
                    scope = None
                elif isinstance(key, tuple) and key[0] == run_id:
                    scope = key[1:]
                else:
                    continue
                directory = self._directory(run_id, scope)
                if any(_event_replaced(previous, _event_snapshot(directory / name))
                       for name, previous in state['snapshots'].items()):
                    self._forget_replay(run_id)
                    return

    def _write(self, run_id, batches, *, scope=None):
        try:
            from tensorboard.compat.proto.event_pb2 import Event
            from tensorboard.compat.proto.summary_pb2 import Summary, SummaryMetadata
            from tensorboard.summary.writer.record_writer import RecordWriter
            with self._lock:
                directory = self._directory(run_id, scope)
                cache_key = run_id if scope is None else (run_id, *scope)
                with _run_lock(directory):
                    files = sorted(directory.glob('events.out.tfevents.*'))
                    snapshots = {path.name: _event_snapshot(path) for path in files}
                    state = self._states.pop(cache_key, None)
                    if state is not None and any(
                        _event_replaced(previous, snapshots.get(name))
                        for name, previous in state['snapshots'].items()
                    ):
                        # The cache describes persisted evidence, not attempted
                        # writes. Cleanup, truncation or replacement invalidates
                        # every derived key; retain and rescan the current files.
                        state = None
                        self._forget_replay(run_id)
                    if state is None:
                        state = {'seen': set(), 'metadata_tags': set(), 'snapshots': {}, 'damaged': set(), 'file': None}
                    self._states[cache_key] = state
                    while len(self._states) > self._max_cached_runs:
                        evicted, _ = self._states.popitem(last=False)
                        # No cursor may outlive the file identities needed to
                        # validate it. Replay is safe and preserves the bound.
                        self._forget_replay(evicted[0] if isinstance(evicted, tuple) else evicted)
                    for path in files:
                        size = snapshots[path.name][2]
                        previous = state['snapshots'].get(path.name)
                        if previous != snapshots[path.name]:
                            previous_size = previous[2] if previous else 0
                            offset = previous_size if size >= previous_size and path.name not in state['damaged'] else 0
                            try:
                                for event in _events(path, Event, offset):
                                    state['seen'].update((value.tag, event.step) for value in event.summary.value)
                                    state['metadata_tags'].update(value.tag for value in event.summary.value
                                                                  if value.HasField('metadata'))
                            except EOFError:
                                state['damaged'].add(path.name)
                            state['snapshots'][path.name] = snapshots[path.name]
                    # TensorBoard's live reader advances files monotonically.
                    # Never append to an older file after a different exporter
                    # has created a newer one, or live dashboards miss points.
                    state['file'] = files[-1] if files and files[-1].name not in state['damaged'] else None
                    pending = []
                    pending_keys = set()
                    pending_metadata = set()
                    for step, wall_time, metrics in batches:
                        values = []
                        for tag, value in metrics.items():
                            measured = {}
                            _measured(measured, tag, value)
                            if tag not in measured:
                                continue
                            key = (tag, step)
                            if key not in state['seen'] and key not in pending_keys:
                                scalar = Summary.Value(tag=tag, simple_value=measured[tag])
                                if tag not in state['metadata_tags'] and tag not in pending_metadata:
                                    label, description = metric_metadata(tag)
                                    scalar.metadata.CopyFrom(SummaryMetadata(
                                        display_name=label, summary_description=description,
                                        plugin_data=SummaryMetadata.PluginData(plugin_name='scalars')))
                                    pending_metadata.add(tag)
                                values.append(scalar)
                                pending_keys.add(key)
                        if values:
                            pending.append(Event(wall_time=wall_time, step=step, summary=Summary(value=values)))
                    if pending:
                        if state['file'] is None:
                            state['file'] = directory / f'events.out.tfevents.{time.time_ns():020d}.{os.getpid()}.{uuid.uuid4().hex}'
                        path = state['file']
                        try:
                            with os.fdopen(_safe_open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT), 'ab') as stream:
                                writer = RecordWriter(stream)
                                if stream.tell() == 0:
                                    writer.write(Event(wall_time=time.time(), file_version='brain.Event:2').SerializeToString())
                                for event in pending:
                                    writer.write(event.SerializeToString())
                                writer.flush()
                            state['seen'].update(pending_keys)
                            state['metadata_tags'].update(pending_metadata)
                            state['snapshots'][path.name] = _event_snapshot(path)
                        except Exception:
                            # Re-scan any successfully persisted prefix and rotate
                            # on retry; a torn frame must never hide later points.
                            self._states.pop(cache_key, None)
                            raise
                    self._last_error = None
                    return {'available': True, 'url': URL, 'run_id': run_id}
        except Exception as exc:
            return self._error(exc, run_id)

    @staticmethod
    def _sample_batch(row):
        step = row.get('sample_index')
        if type(step) is not int or step < 0:
            return None
        metrics = {}
        fields = (
            ('cpu', 'system_utilization_percent', 'system/cpu_percent'),
            ('gpu', 'utilization_percent', 'system/gpu_utilization_percent'),
            ('gpu', 'memory_used_bytes', 'system/gpu_memory_used_bytes'),
            ('gpu', 'memory_total_bytes', 'system/gpu_memory_total_bytes'),
            *((group, field, f'{group}/{field}') for group, names in (
                ('system', ('memory_used_bytes', 'memory_total_bytes', 'memory_percent',
                            'disk_used_bytes', 'disk_total_bytes', 'disk_percent')),
                ('processes', ('rss_bytes', 'cpu_percent')),
            ) for field in names),
        )
        for group, field, tag in fields:
            values = row.get(group) or {}
            value = _number(values.get(field)) if isinstance(values, dict) else None
            if value is not None:
                metrics[tag] = value
        for group, names in (('cpu', CPU_FIELDS), ('gpu', GPU_FIELDS)):
            values = row.get(group)
            if not isinstance(values, dict): continue
            aliases = {'system_utilization_percent'} if group == 'cpu' else {
                'utilization_percent', 'memory_used_bytes', 'memory_total_bytes'}
            for field in names:
                if field not in aliases:
                    _measured(metrics, f'{group}/{field}', values.get(field))
        gpu = row.get('gpu') or {}
        if isinstance(gpu, dict):
            used, total = _number(gpu.get('memory_used_bytes')), _number(gpu.get('memory_total_bytes'))
            if used is not None and total is not None and total > 0:
                metrics['system/gpu_memory_percent'] = used / total * 100
        return step, _timestamp(row.get('time')), metrics

    def write_sample(self, run_id, row):
        try:
            batch = self._sample_batch(row)
            return self._write(run_id, [batch] if batch else [])
        except Exception as exc:
            return self._error(exc, run_id)

    def _machine_batch(self, row):
        # Process totals refer to the experiment, not this physical host's
        # independent resource capacity, and stay out of the host run.
        return self._sample_batch({key: value for key, value in row.items() if key != 'processes'})

    def write_machine_sample(self, run_id, machine, row):
        try:
            batch = self._machine_batch(row)
            return self._write(run_id, [batch] if batch else [], scope=('machines', machine))
        except Exception as exc:
            return self._error(exc, run_id)

    def _node_batches(self, node, event):
        """Convert one observation without acquiring filesystem locks."""
        payload = event.get('payload') or {}
        metrics = {}
        if event.get('kind') == 'training':
            step = payload.get('round_id')
            for field in ('metrics_eval_s', 'metrics_callback_s', 'training_s'):
                _measured(metrics, f'training_round/{field}', payload.get(field))
            _measured(metrics, 'training_round/local_epochs', payload.get('epochs'))
        elif event.get('kind') == 'epoch':
            step = payload.get('step')
            fields = (('accuracy', 'training/accuracy'), ('loss', 'training/cross_entropy'),
                      ('samples', 'training/samples'), ('epoch', 'training/local_epoch'),
                      ('round_id', 'training/global_round'), ('elapsed_s', 'training/epoch_s'),
                      ('metrics_eval_s', 'training/metrics_eval_s'))
            for field, tag in fields:
                value = _number(payload.get(field))
                if value is not None and (field != 'accuracy' or value <= 1):
                    metrics[tag] = value
        elif event.get('kind') == 'resource':
            step = payload.get('sample_index')
            processes = payload.get('processes') or {}
            for field in ('cpu_percent', 'rss_bytes'):
                value = _number(processes.get(field))
                if value is not None:
                    metrics[f'processes/{field}'] = value
        else:
            raise ValueError('无效的节点指标事件')
        if type(step) is not int or step < 0:
            raise ValueError('无效的节点指标横轴')
        batches = {('nodes', node['id']): (step, _timestamp(event.get('time')), metrics)}
        machine = node.get('machine')
        if event.get('kind') == 'resource' and machine and any(key in payload for key in ('cpu', 'gpu', 'system')):
            batch = self._machine_batch({**payload, 'time': event.get('time')})
            if batch:
                batches[('machines', machine)] = batch
        return batches

    def write_node_event(self, run_id, node, event):
        try:
            result = {'available': True, 'url': URL, 'run_id': run_id}
            for scope, batch in self._node_batches(node, event).items():
                host_result = self._write(run_id, [batch], scope=scope)
                if not host_result['available']:
                    result = host_result
            return result
        except Exception as exc:
            return self._error(exc, run_id)

    @staticmethod
    def _client_round_batch(round_id, local_epochs, epochs, *, fallback_time=None):
        """Overlay only a recorded final epoch of an already completed round.

        Use the configured final epoch, not the last item received: events may
        be missing or unordered, and a partial epoch must never become the
        immutable comparison point. Per-epoch training tags remain unchanged.
        """
        if (type(round_id) is not int or round_id < 1 or type(local_epochs) is not int
                or local_epochs < 1 or not isinstance(epochs, list)):
            return None
        final = [epoch for epoch in epochs if isinstance(epoch, dict)
                 and type(epoch.get('round_id')) is int and epoch['round_id'] == round_id
                 and type(epoch.get('epoch')) is int and epoch['epoch'] == local_epochs
                 and type(epoch.get('step')) is int and epoch['step'] == round_id * local_epochs]
        if len(final) != 1:
            return None
        epoch = final[0]
        metrics = {}
        accuracy = _number(epoch.get('accuracy'))
        if accuracy is not None and accuracy <= 1:
            metrics['evaluation/accuracy'] = accuracy
        _measured(metrics, 'evaluation/cross_entropy', epoch.get('loss'))
        return (round_id, _timestamp(epoch.get('time') or fallback_time), metrics) if metrics else None

    def sync(self, record):
        # Serialize cursor/cache transactions, while live node observations can
        # still use the independent, short file-write lock between batches.
        with self._sync_lock:
            return self._sync(record)

    def _sync(self, record):
        run_id = record.get('run_id') if isinstance(record, dict) else None
        try:
            self._check_replay_files(run_id)
            batches = []
            scoped_batches = {}
            cumulative_bytes = 0
            cumulative_known = True
            expected_round = 1
            times = {event.get('round'): event.get('time') for event in record.get('events', [])
                     if isinstance(event, dict) and event.get('stage') == 'completed_round'}
            initial = record.get('initial_metrics') or {}
            rounds = [(0, initial), *((row.get('round'), row) for row in record.get('rounds', []))]
            for step, row in rounds:
                if type(step) is not int or step < 0:
                    continue
                metrics = {}
                accuracy, loss = _number(row.get('accuracy')), _number(row.get('loss'))
                if accuracy is not None and accuracy <= 1:
                    metrics['evaluation/accuracy'] = accuracy
                if loss is not None:
                    metrics['evaluation/cross_entropy'] = loss
                timestamp = row.get('time') or times.get(step) or record.get('created_at')
                if step > 0:
                    metrics.update(_round_metrics(row))
                    measured_bytes = _number(row.get('bytes_sent'))
                    if step != expected_round or measured_bytes is None:
                        cumulative_known = False
                    if cumulative_known:
                        cumulative_bytes += measured_bytes
                        _measured(metrics, 'communication/cumulative_round_bytes', cumulative_bytes)
                    expected_round = step + 1
                    for scope, values in _actor_metrics(row).items():
                        if scope is None:
                            metrics.update(values)
                        else:
                            scoped_batches.setdefault(scope, []).append((step, _timestamp(timestamp), values))
                batches.append((step, _timestamp(timestamp), metrics))
            evidence = record.get('evidence') or {}
            if record.get('status') in ('completed', 'failed', 'aborted'):
                completed = [row['round'] for row in record.get('rounds', [])
                             if isinstance(row, dict) and type(row.get('round')) is int and row['round'] >= 1]
                if completed:
                    summary = record.get('summary') or {}
                    metrics = {}
                    for field in ('elapsed_s', 'controller_cpu_s', 'completed_rounds'):
                        tag = 'duration_s' if field == 'elapsed_s' else field
                        _measured(metrics, f'task/{tag}', summary.get(field))
                    _measured(metrics, 'communication/task_total_bytes', summary.get('bytes_sent'))
                    compute = evidence.get('compute') or {}
                    metrics.update(_gpu_metrics(compute.get('coordinator_gpu_profile'), prefix='gpu_compute/task_total'))
                    finished = next((event.get('time') for event in reversed(record.get('events', []))
                                     if isinstance(event, dict) and event.get('stage') == 'finished'), None)
                    batches.append((max(completed), _timestamp(record.get('finished_at') or finished
                                                              or record.get('created_at')), metrics))
            resources = evidence.get('resources') or evidence.get('memory') or {}
            hardware = resources.get('hardware') or {}
            for row in hardware.get('samples', []):
                batch = self._sample_batch(row)
                if batch:
                    batches.append(batch)
            result = self._write(run_id, batches)

            def collect(node, event):
                converted = self._node_batches(node, event)
                for scope, batch in converted.items():
                    scoped_batches.setdefault(scope, []).append(batch)
                return tuple(converted)

            # Per-epoch records are small enough to retain with the round. They
            # also recover logs when live telemetry was unavailable. Group all
            # observations before writing: even a long history takes one file
            # lock/scan per node or machine, rather than one for every epoch.
            node_rows = {row['id']: row for row in evidence.get('node_snapshot', [])}
            local_epochs = (record.get('config') or {}).get('local_epochs')
            completed_rounds = {row['round'] for row in record.get('rounds', [])
                                if isinstance(row, dict) and type(row.get('round')) is int
                                and row['round'] >= 1}
            for row in record.get('rounds', []):
                for node_id, epochs in (row.get('client_training_metrics') or {}).items():
                    node = node_rows.get(node_id, {'id': node_id})
                    for epoch in epochs:
                        collect(node, {
                            'kind': 'epoch', 'time': epoch.get('time') or record.get('created_at'), 'payload': epoch})
                    if isinstance(node_id, str) and re.fullmatch(r'client[1-9][0-9]*', node_id):
                        batch = self._client_round_batch(row.get('round'), local_epochs, epochs,
                                                         fallback_time=record.get('created_at'))
                        if batch:
                            scoped_batches.setdefault(('nodes', node_id), []).append(batch)
            # Resource history stays outside result.json and the HTTP snapshot.
            from dgfl.experiments.monitoring import iter_monitor_events
            with self._lock:
                offsets = self._replay_offsets.pop(run_id, {})
                self._replay_offsets[run_id] = offsets
                pending_epochs = dict(self._replay_pending.get(run_id, {}))
                while len(self._replay_offsets)>8:
                    evicted, _ = self._replay_offsets.popitem(last=False)
                    self._replay_pending.pop(evicted, None)
            replay = []
            for row in iter_monitor_events(self.runtime, run_id, offsets):
                node = node_rows.get(row['node_id'])
                if node is not None:
                    replay.append((row['_file'], row['_offset'], collect(node, row['event'])))
                    event = row['event']
                    if event.get('kind') == 'epoch' and re.fullmatch(r'client[1-9][0-9]*', node['id']):
                        epoch = {**event.get('payload', {}), 'time': event.get('time')}
                        batch = self._client_round_batch(epoch.get('round_id'), local_epochs, [epoch],
                                                         fallback_time=record.get('created_at'))
                        if batch:
                            # Telemetry commonly arrives before aggregation has
                            # completed this round. Keep its final observation
                            # after advancing the resource/epoch replay cursor.
                            pending_epochs.setdefault((node['id'], batch[0]), batch)
            for (node_id, round_id), batch in pending_epochs.items():
                if round_id in completed_rounds:
                    scoped_batches.setdefault(('nodes', node_id), []).append(batch)
            local_machine = evidence.get('monitor_local_machine')
            if local_machine:
                for row in hardware.get('samples', []):
                    batch = self._machine_batch(row)
                    if batch:
                        scoped_batches.setdefault(('machines', local_machine), []).append(batch)
            written = {}
            for scope, grouped in scoped_batches.items():
                child = self._write(run_id, grouped, scope=scope)
                written[scope] = child['available']
                if not child['available']:
                    result = child
            failed_files = set()
            for name, offset, scopes in replay:
                # An event may require both a node and a machine run. Advance
                # only after both batches succeed; a partial write is safe to
                # replay because the exporter deduplicates each tag and step.
                if name in failed_files:
                    continue
                if not all(written.get(scope, False) for scope in scopes):
                    failed_files.add(name)
                    continue
                with self._lock:
                    offsets[name] = max(offsets.get(name, 0), offset)
            with self._lock:
                remaining = {key: batch for key, batch in pending_epochs.items()
                             if key[1] not in completed_rounds or not written.get(('nodes', key[0]), False)}
                if len(remaining) > self._max_pending_epochs:
                    # Bound the in-memory waiting queue without losing evidence:
                    # the next sync replays its persistent journals from scratch.
                    offsets.clear()
                    remaining.clear()
                if self._replay_offsets.get(run_id) is offsets:
                    self._replay_pending[run_id] = remaining
            return result
        except Exception as exc:
            return self._error(exc, run_id)

    def close(self, run_id):
        with self._sync_lock, self._lock:
            for key in list(self._states):
                if key == run_id or (isinstance(key, tuple) and key[0] == run_id):
                    self._states.pop(key, None)
            self._replay_offsets.pop(run_id, None)
            self._replay_pending.pop(run_id, None)

    def close_all(self):
        with self._sync_lock, self._lock:
            self._states.clear()
            self._replay_offsets.clear()
            self._replay_pending.clear()
