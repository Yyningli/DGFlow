"""Optional exact CUDA arithmetic for PUBLIC aggregate verification.

The driver and NVIDIA NVRTC are loaded only on demand. GPU selection is strict:
an unavailable device or failed kernel is an error, never a silent CPU fallback.
Canonical G1 decoding remains in the checked CPU library; the CUDA kernels
evaluate the complete public G1 and GT equations and exact GT subgroup tests.
"""
from __future__ import annotations

import copy
import csv
import ctypes as ct
import hashlib
import importlib.util
import os
import shutil
import subprocess
import threading
import time
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path

from dgfl import system_info


class GPUUnavailable(ValueError):
    """A requested CUDA device cannot execute the checked backend."""


_runtime = None
_runtime_lock = threading.RLock()
_probe = None
OPERATIONS = ['ec_batch', 'gt_exp_batch', 'gt_subgroup_batch']


def _nvrtc_path():
    try:
        spec = importlib.util.find_spec('nvidia.cuda_nvrtc')
    except (ImportError, ModuleNotFoundError, ValueError):
        spec = None
    if spec is None or not spec.submodule_search_locations:
        raise GPUUnavailable('GPU 模式需要安装项目 gpu 可选依赖（NVIDIA NVRTC）')
    folder = Path(next(iter(spec.submodule_search_locations)))
    pattern = 'nvrtc64_*.dll' if os.name == 'nt' else 'libnvrtc.so*'
    candidates = [p for directory in ('bin', 'lib') for p in (folder/directory).glob(pattern)
                  if '.alt.' not in p.name and 'builtins' not in p.name]
    if not candidates:
        raise GPUUnavailable('未找到 NVIDIA NVRTC 动态库')
    return sorted(candidates)[0]


def _library(path):
    try:
        return (ct.WinDLL if os.name == 'nt' else ct.CDLL)(str(path))
    except OSError as exc:
        raise GPUUnavailable('无法加载 CUDA 驱动或 NVIDIA NVRTC 动态库') from exc


def _bind(library, name, arguments):
    function = getattr(library, name)
    function.argtypes = arguments
    function.restype = ct.c_int
    return function


def detect_gpu_hardware():
    """Identify NVIDIA hardware without importing/installing CUDA Python packages.

    ``available`` here describes hardware only. Cryptographic availability must
    use compute_capabilities()/require_gpu(), including exact kernel self-tests.
    No context, compiler, network request or package installer is used here.
    """
    failure = ''
    try:
        driver = _library('nvcuda.dll' if os.name == 'nt' else 'libcuda.so.1')
        def call(name, arguments, *values):
            code = _bind(driver, name, arguments)(*values)
            if code:
                raise GPUUnavailable(f'{name} 返回 CUDA 状态码 {code}')
        call('cuInit', [ct.c_uint], 0)
        count = ct.c_int()
        call('cuDeviceGetCount', [ct.POINTER(ct.c_int)], ct.byref(count))
        if count.value < 1:
            raise GPUUnavailable('没有可用的 NVIDIA CUDA 显卡')
        device = ct.c_int()
        call('cuDeviceGet', [ct.POINTER(ct.c_int), ct.c_int], ct.byref(device), 0)
        name = ct.create_string_buffer(256)
        call('cuDeviceGetName', [ct.c_void_p, ct.c_int, ct.c_int], name, len(name), device)
        total, version = ct.c_size_t(), ct.c_int()
        call('cuDeviceTotalMem_v2', [ct.POINTER(ct.c_size_t), ct.c_int], ct.byref(total), device)
        call('cuDriverGetVersion', [ct.POINTER(ct.c_int)], ct.byref(version))
        return {'available': True, 'hardware_available': True, 'driver_available': True,
                'name': name.value.decode('utf8', 'replace'), 'device_index': 0,
                'memory_bytes': total.value, 'driver_version': version.value,
                'detection_source': 'cuda_driver', 'reason': ''}
    except (GPUUnavailable, OSError, AttributeError) as exc:
        failure = str(exc)
    executable = shutil.which('nvidia-smi')
    if executable is None and os.name == 'nt':
        program_files = Path(os.environ.get('PROGRAMFILES') or (
            os.environ.get('SYSTEMDRIVE', 'C:') + '/Program Files'))
        candidate = program_files/'NVIDIA Corporation'/'NVSMI'/'nvidia-smi.exe'
        if candidate.is_file():
            executable = str(candidate)
    if executable is not None:
        try:
            options = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
            answer = subprocess.run([executable, '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
                                    capture_output=True, text=True, timeout=3, check=True, **options)
            rows = list(csv.reader(answer.stdout.splitlines()))
            if rows and len(rows[0]) == 2 and rows[0][0].strip():
                return {'available': True, 'hardware_available': True, 'driver_available': False,
                        'name': rows[0][0].strip(), 'device_index': 0,
                        'memory_bytes': int(rows[0][1].strip()) * 1024**2,
                        'detection_source': 'nvidia-smi',
                        'reason': '已识别 NVIDIA 显卡，但 CUDA 驱动接口不可用：' + failure}
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return {'available': False, 'hardware_available': False, 'driver_available': False,
            'name': '', 'backend': 'cuda_nvrtc', 'detection_source': 'unavailable',
            'reason': failure or '未检测到 NVIDIA CUDA 显卡；CPU 计算可用'}


class CudaRuntime:
    def __init__(self, *, compile_kernels=True):
        self.lock = threading.RLock()
        self.driver = _library('nvcuda.dll' if os.name == 'nt' else 'libcuda.so.1')
        self._init = _bind(self.driver, 'cuInit', [ct.c_uint])
        self._count = _bind(self.driver, 'cuDeviceGetCount', [ct.POINTER(ct.c_int)])
        self._get = _bind(self.driver, 'cuDeviceGet', [ct.POINTER(ct.c_int), ct.c_int])
        self._name = _bind(self.driver, 'cuDeviceGetName', [ct.c_void_p, ct.c_int, ct.c_int])
        self._attribute = _bind(self.driver, 'cuDeviceGetAttribute', [ct.POINTER(ct.c_int), ct.c_int, ct.c_int])
        self._memory = _bind(self.driver, 'cuDeviceTotalMem_v2', [ct.POINTER(ct.c_size_t), ct.c_int])
        self._version = _bind(self.driver, 'cuDriverGetVersion', [ct.POINTER(ct.c_int)])
        self.check(self._init(0), 'CUDA 初始化')
        count = ct.c_int()
        self.check(self._count(ct.byref(count)), 'CUDA 设备枚举')
        if count.value < 1:
            raise GPUUnavailable('没有可用的 NVIDIA CUDA 显卡')
        self.device = ct.c_int()
        self.check(self._get(ct.byref(self.device), 0), 'CUDA 设备选择')
        name = ct.create_string_buffer(256)
        self.check(self._name(name, len(name), self.device), 'CUDA 设备名称')
        total = ct.c_size_t()
        self.check(self._memory(ct.byref(total), self.device), 'CUDA 显存查询')
        major, minor, version = ct.c_int(), ct.c_int(), ct.c_int()
        self.check(self._attribute(ct.byref(major), 75, self.device), 'CUDA 架构查询')
        self.check(self._attribute(ct.byref(minor), 76, self.device), 'CUDA 架构查询')
        self.check(self._version(ct.byref(version)), 'CUDA 驱动版本')
        self.nvrtc_path = _nvrtc_path()
        self.dll_handles=[]
        if os.name=='nt':
            self.dll_handles.append(os.add_dll_directory(str(self.nvrtc_path.parent)))
            # NVRTC opens its builtins by basename. Preload the matching wheel's
            # library rather than depending on the user's global PATH.
            self.dll_handles.extend(_library(path) for path in self.nvrtc_path.parent.glob('nvrtc-builtins*.dll'))
        self.info = {'available': True, 'name': name.value.decode('utf8', 'replace'),
                     'memory_bytes': total.value, 'device_index': 0,
                     'compute_capability': f'{major.value}.{minor.value}', 'driver_version': version.value,
                     'backend': 'cuda_nvrtc', 'accelerated_operations': list(OPERATIONS),
                     'reason': '', 'verified': False}
        self.info['device_resources'] = self._device_resources()
        self.context = ct.c_void_p()
        self.module = ct.c_void_p()
        self._context_retained = False
        self._module_loaded = False
        self.functions = {}
        self.kernel_resources = {}
        self.info['kernel_resources'] = self.kernel_resources
        self.stats = self._empty_stats()
        self.kernel_stats = {}
        self._pool = OrderedDict()
        self._pool_reserved_bytes = 0
        self._pool_peak_bytes = 0
        self._closed = False
        self._events = []
        self.verified = False
        try:
            self.configure(
                block_size=self._environment_integer('DGFLOW_CUDA_BLOCK_SIZE', 32),
                chunk_size=self._environment_integer('DGFLOW_CUDA_CHUNK_SIZE', 1024),
                pool_limit_bytes=self._environment_integer('DGFLOW_CUDA_POOL_BYTES', 128*1024*1024),
            scheduler_slots=(self._environment_integer('DGFLOW_CUDA_SCHEDULER_SLOTS', 1)
                                 if compile_kernels else 0))
            if compile_kernels:
                self._initialize_kernels(major.value, minor.value)
        except BaseException:
            self.close()
            raise

    def _initialize_kernels(self, major, minor):
        self._retain = _bind(self.driver, 'cuDevicePrimaryCtxRetain', [ct.POINTER(ct.c_void_p), ct.c_int])
        self._release = _bind(self.driver, 'cuDevicePrimaryCtxRelease_v2', [ct.c_int])
        self._set = _bind(self.driver, 'cuCtxSetCurrent', [ct.c_void_p])
        self._sync = _bind(self.driver, 'cuCtxSynchronize', [])
        self._alloc = _bind(self.driver, 'cuMemAlloc_v2', [ct.POINTER(ct.c_uint64), ct.c_size_t])
        self._free = _bind(self.driver, 'cuMemFree_v2', [ct.c_uint64])
        self._upload = _bind(self.driver, 'cuMemcpyHtoD_v2', [ct.c_uint64, ct.c_void_p, ct.c_size_t])
        self._download = _bind(self.driver, 'cuMemcpyDtoH_v2', [ct.c_void_p, ct.c_uint64, ct.c_size_t])
        self._clear = _bind(self.driver, 'cuMemsetD8_v2', [ct.c_uint64, ct.c_ubyte, ct.c_size_t])
        self._event_create = _bind(self.driver, 'cuEventCreate', [ct.POINTER(ct.c_void_p), ct.c_uint])
        self._event_destroy = _bind(self.driver, 'cuEventDestroy_v2', [ct.c_void_p])
        self._event_record = _bind(self.driver, 'cuEventRecord', [ct.c_void_p, ct.c_void_p])
        self._event_sync = _bind(self.driver, 'cuEventSynchronize', [ct.c_void_p])
        self._event_elapsed = _bind(self.driver, 'cuEventElapsedTime',
                                    [ct.POINTER(ct.c_float), ct.c_void_p, ct.c_void_p])
        self._load = _bind(self.driver, 'cuModuleLoadData', [ct.POINTER(ct.c_void_p), ct.c_void_p])
        self._unload = _bind(self.driver, 'cuModuleUnload', [ct.c_void_p])
        self._function = _bind(self.driver, 'cuModuleGetFunction',
                               [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_char_p])
        try:
            self._function_attribute = _bind(self.driver, 'cuFuncGetAttribute',
                                             [ct.POINTER(ct.c_int), ct.c_int, ct.c_void_p])
            self._function_attribute_error = None
        except (AttributeError, OSError) as exc:
            self._function_attribute = None
            self._function_attribute_error = f'{type(exc).__name__}: {exc}'
        self._launch = _bind(self.driver, 'cuLaunchKernel',
                            [ct.c_void_p, ct.c_uint, ct.c_uint, ct.c_uint, ct.c_uint, ct.c_uint,
                             ct.c_uint, ct.c_uint, ct.c_void_p, ct.POINTER(ct.c_void_p), ct.c_void_p])
        self.check(self._retain(ct.byref(self.context), self.device), 'CUDA 上下文创建')
        self._context_retained = True
        self.check(self._set(self.context), 'CUDA 上下文选择')
        cubin = self.compile(major, minor)
        buffer = ct.create_string_buffer(cubin)
        self.check(self._load(ct.byref(self.module), buffer), 'CUDA 密码内核加载')
        self._module_loaded = True
        for _ in range(2):
            event = ct.c_void_p()
            # CU_EVENT_BLOCKING_SYNC keeps the CPU available while waiting;
            # timing remains enabled. Events are reused under self.lock.
            self.check(self._event_create(ct.byref(event), 1), 'CUDA 计时事件创建')
            self._events.append(event)

    @staticmethod
    def check(code, action):
        if code:
            raise GPUUnavailable(f'{action}失败（CUDA/NVRTC 状态码 {code}）；GPU 任务已停止')

    @staticmethod
    def _empty_stats():
        return dict.fromkeys((
            'batches', 'rows', 'failed_batches', 'failed_wall_seconds',
            'host_wall_seconds', 'lock_wait_seconds', 'scheduler_wait_seconds',
            'setup_seconds', 'allocation_seconds', 'upload_seconds',
            'output_clear_seconds', 'launch_seconds', 'sync_seconds',
            'download_seconds', 'kernel_seconds', 'upload_bytes', 'download_bytes',
            'allocations', 'allocation_reuses', 'frees'), 0)

    @staticmethod
    def _environment_integer(name, default):
        try:
            return int(os.environ.get(name, default))
        except (TypeError, ValueError) as exc:
            raise GPUUnavailable(f'无效的 CUDA 配置 {name}') from exc

    def _device_resources(self):
        """Read optional occupancy diagnostics without changing crypto readiness."""
        result, errors = {}, {}
        fields = {'multiprocessor_count': 16, 'max_threads_per_multiprocessor': 39,
                  'registers_per_multiprocessor': 82}
        for name, attribute in fields.items():
            value = ct.c_int()
            try:
                status = self._attribute(ct.byref(value), attribute, self.device)
                if status:
                    errors[name] = f'CUDA status {status}'
                result[name] = value.value if not status else None
            except Exception as exc:
                result[name] = None
                errors[name] = f'{type(exc).__name__}: {exc}'
        return {**result, 'errors': errors}

    def _record_kernel_resources(self, name, function):
        """cuFuncGetAttribute reports actual JIT register/local-memory usage.

        An unavailable diagnostic is represented by None and an error; it can
        never replace, waive or prevent a cryptographic validation.
        """
        fields = {'max_threads_per_block': 0, 'static_shared_bytes': 1,
                  'local_bytes_per_thread': 3, 'num_registers': 4}
        result = {'function_name': name, 'errors': {}}
        for field, attribute in fields.items():
            value = ct.c_int()
            if self._function_attribute is None:
                result[field] = None
                result['errors'][field] = self._function_attribute_error
                continue
            try:
                status = self._function_attribute(ct.byref(value), attribute, function)
                if status:
                    result['errors'][field] = f'CUDA status {status}'
                result[field] = value.value if not status else None
            except Exception as exc:
                result[field] = None
                result['errors'][field] = f'{type(exc).__name__}: {exc}'
        self.kernel_resources[name] = result

    def configure(self, *, block_size=None, chunk_size=None,
                  pool_limit_bytes=None, scheduler_slots=None):
        """Tune launches and bounded reuse without changing the cryptographic ABI."""
        with self.lock:
            values = (
                ('block_size', block_size, 1, 1024),
                ('chunk_size', chunk_size, 1, 20_000),
                ('pool_limit_bytes', pool_limit_bytes, 0, 1024*1024*1024))
            updates = {}
            for name, value, minimum, maximum in values:
                if value is None:
                    continue
                if type(value) is not int or not minimum <= value <= maximum:
                    raise ValueError(f'{name} must be an integer between {minimum} and {maximum}')
                updates[name] = value
            slots = scheduler_slots if scheduler_slots is not None else getattr(self, 'scheduler_slots', 0)
            if type(slots) is not int or slots not in (0, 1, 2, 4):
                raise ValueError('scheduler_slots must be 0, 1, 2 or 4')
            changing_scheduler = slots != getattr(self, 'scheduler_slots', None)
            previous = getattr(self, '_scheduler', None)
            replacement = previous
            if changing_scheduler:
                replacement = None
                if slots:
                    from dgfl.crypto.cuda_scheduler import get_scheduler
                    replacement = get_scheduler(self.info['device_index'], slots)
            try:
                budget = updates.get('pool_limit_bytes', getattr(self, 'pool_limit_bytes', 128*1024*1024))
                if getattr(self, '_pool_reserved_bytes', 0) > budget:
                    self.check(self._set(self.context), 'CUDA 上下文选择')
                    self._release_pool()
            except BaseException:
                if changing_scheduler and replacement is not None:
                    replacement.close()
                raise
            for name, value in updates.items():
                setattr(self, name, value)
            if changing_scheduler:
                self._scheduler = replacement
                self.scheduler_slots = slots
                if previous is not None:
                    previous.close()

    def stats_snapshot(self):
        """Copy cumulative counters. Device elapsed overlaps host synchronization."""
        with self.lock:
            return {
                'schema_version': 1,
                'totals': dict(self.stats),
                'by_kernel': copy.deepcopy(self.kernel_stats),
                'kernel_resources': copy.deepcopy(self.kernel_resources),
                'device_resources': copy.deepcopy(self.info['device_resources']),
                'pool': {'reserved_bytes': self._pool_reserved_bytes,
                         'peak_reserved_bytes': self._pool_peak_bytes,
                         'budget_bytes': self.pool_limit_bytes, 'slots': len(self._pool)},
                'configuration': {'block_size': self.block_size, 'chunk_size': self.chunk_size,
                                  'scheduler_slots': self.scheduler_slots},
                'timing_notes': {
                    'host_wall_seconds': 'Complete successful execute calls, including waits and host preparation.',
                    'kernel_seconds': 'CUDA event elapsed; overlaps sync_seconds and may include GPU scheduling.',
                    'sync_seconds': 'Host blocking wait for the completion event; includes kernel execution.',
                    'upload_seconds': 'Host synchronous HtoD copy calls, excluding host packing.',
                    'download_seconds': 'Host synchronous DtoH copy calls, excluding host unpacking.',
                    'additive': 'Do not add kernel_seconds to host phase times; device and host overlap.'}}

    def stats_delta(self, before):
        """Difference cumulative counters while retaining current pool gauges."""
        after = self.stats_snapshot()
        old_totals = before.get('totals', {})
        after['totals'] = {key: value-old_totals.get(key, 0)
                           for key, value in after['totals'].items()}
        old_kernels = before.get('by_kernel', {})
        after['by_kernel'] = {
            name: {key: value-old_kernels.get(name, {}).get(key, 0) for key, value in counters.items()}
            for name, counters in after['by_kernel'].items()}
        return after

    def _release_pool(self):
        """Release retained memory, best effort even after a failed CUDA call."""
        for pointer, _capacity in reversed(self._pool.values()):
            self._free(pointer)
        self._pool.clear()
        self._pool_reserved_bytes = 0

    def close(self):
        with self.lock:
            if self._closed:
                return
            # Mark ownership consumed before calling the driver: a failing
            # cleanup must neither skip other resources nor release twice.
            self._closed = True
            self.verified = False
            self.info['verified'] = False
            failures = []

            def attempt(action, function, *arguments):
                try:
                    status = function(*arguments)
                    if status:
                        failures.append(f'{action}: CUDA status {status}')
                except Exception as exc:
                    failures.append(f'{action}: {type(exc).__name__}: {exc}')

            if self._context_retained:
                attempt('context selection', self._set, self.context)
                for pointer, _capacity in reversed(self._pool.values()):
                    attempt('buffer release', self._free, pointer)
                self._pool.clear()
                self._pool_reserved_bytes = 0
                events, self._events = self._events, []
                for event in events:
                    attempt('event destruction', self._event_destroy, event)
            if self._module_loaded:
                self._module_loaded = False
                attempt('module unload', self._unload, self.module)
                self.module = ct.c_void_p()
                self.functions.clear()
            if self._context_retained:
                self._context_retained = False
                # Release only this runtime's retain reference. Never reset a
                # primary context shared with another CUDA library.
                attempt('primary context release', self._release, self.device)
                self.context = ct.c_void_p()
            scheduler = getattr(self, '_scheduler', None)
            if scheduler is not None:
                attempt('scheduler close', scheduler.close)
            for handle in self.dll_handles:
                closer = getattr(handle, 'close', None)
                if callable(closer):
                    attempt('DLL directory close', closer)
            self.info['cleanup_errors'] = failures

    def _buffer(self, key, size, used_keys, temporary, counters):
        """Reuse one pointer per kernel ABI slot, evicting only idle slots."""
        required = max(1, size)
        cached = self._pool.get(key)
        if cached is not None and cached[1] >= required:
            self._pool.move_to_end(key)
            used_keys.add(key)
            counters['allocation_reuses'] += 1
            return cached[0]
        if cached is not None:
            self.check(self._free(cached[0]), 'GPU 旧显存释放')
            counters['frees'] += 1
            self._pool_reserved_bytes -= cached[1]
            del self._pool[key]
        capacity = 1 << (required-1).bit_length()
        if capacity <= self.pool_limit_bytes:
            for other, (pointer, available) in list(self._pool.items()):
                if self._pool_reserved_bytes+capacity <= self.pool_limit_bytes:
                    break
                if other in used_keys:
                    continue
                self.check(self._free(pointer), 'GPU 闲置显存释放')
                counters['frees'] += 1
                self._pool_reserved_bytes -= available
                del self._pool[other]
        retained = self._pool_reserved_bytes+capacity <= self.pool_limit_bytes
        pointer = ct.c_uint64()
        self.check(self._alloc(ct.byref(pointer), capacity if retained else required), 'GPU 显存分配')
        counters['allocations'] += 1
        if retained:
            self._pool[key] = (pointer, capacity)
            self._pool_reserved_bytes += capacity
            self._pool_peak_bytes = max(self._pool_peak_bytes, self._pool_reserved_bytes)
            used_keys.add(key)
        else:
            temporary.append(pointer)
        return pointer

    def compile(self, major, minor):
        folder = Path(__file__).with_name('cuda')
        names = ['field.cuh', 'g1.cuh', 'gt.cuh']
        try:
            contents = [folder.joinpath(name).read_bytes() for name in names]
        except OSError as exc:
            raise GPUUnavailable('GPU 密码内核源文件缺失，请安装包含 CUDA 内核的项目版本') from exc
        # Compile native machine code for the selected device. A newer NVRTC's
        # PTX requires a matching driver JIT even within the CUDA 12 family;
        # CUBIN permits CUDA minor-version compatibility without that JIT.
        options = [b'--std=c++14', f'--gpu-architecture=sm_{major}{minor}'.encode(),
                   b'--device-as-default-execution-space',b'--Ofast-compile=mid']
        digest = hashlib.sha256(b'\0'.join(contents+options)).hexdigest()
        self.info['kernel_source_sha256'] = hashlib.sha256(b'\0'.join(contents)).hexdigest()
        self.info['kernel_build_sha256'] = digest
        nvrtc = _library(self.nvrtc_path)
        version = _bind(nvrtc, 'nvrtcVersion', [ct.POINTER(ct.c_int), ct.POINTER(ct.c_int)])
        version_major, version_minor = ct.c_int(), ct.c_int()
        self.check(version(ct.byref(version_major), ct.byref(version_minor)), 'NVRTC 版本查询')
        self.info['nvrtc_version'] = f'{version_major.value}.{version_minor.value}'
        cache_key=f'{digest}-sm{major}{minor}-nvrtc{version_major.value}.{version_minor.value}'
        cache_dir=Path(__file__).resolve().parents[3]/'runtime'/'gpu-cache'
        cache_path=cache_dir/(cache_key+'.cubin')
        if cache_path.is_file() and 0<cache_path.stat().st_size<64*1024*1024:
            cached=cache_path.read_bytes()
            if cached and len(cached)<64*1024*1024:
                self.info['cubin_sha256']=hashlib.sha256(cached).hexdigest()
                return cached
        create = _bind(nvrtc, 'nvrtcCreateProgram', [ct.POINTER(ct.c_void_p), ct.c_char_p, ct.c_char_p,
                                                   ct.c_int, ct.POINTER(ct.c_char_p), ct.POINTER(ct.c_char_p)])
        compile_program = _bind(nvrtc, 'nvrtcCompileProgram', [ct.c_void_p, ct.c_int, ct.POINTER(ct.c_char_p)])
        log_size = _bind(nvrtc, 'nvrtcGetProgramLogSize', [ct.c_void_p, ct.POINTER(ct.c_size_t)])
        get_log = _bind(nvrtc, 'nvrtcGetProgramLog', [ct.c_void_p, ct.c_void_p])
        cubin_size = _bind(nvrtc, 'nvrtcGetCUBINSize', [ct.c_void_p, ct.POINTER(ct.c_size_t)])
        get_cubin = _bind(nvrtc, 'nvrtcGetCUBIN', [ct.c_void_p, ct.c_void_p])
        destroy = _bind(nvrtc, 'nvrtcDestroyProgram', [ct.POINTER(ct.c_void_p)])
        program = ct.c_void_p()
        source = b'\n'.join(b'#include "'+name.encode()+b'"' for name in names)
        headers = (ct.c_char_p*len(names))(*contents)
        include_names = (ct.c_char_p*len(names))(*(name.encode() for name in names))
        self.check(create(ct.byref(program), source, b'dgflow_crypto.cu', len(names), headers, include_names),
                   'CUDA 编译单元创建')
        try:
            result = compile_program(program, len(options), (ct.c_char_p*len(options))(*options))
            if result:
                length = ct.c_size_t()
                log_size(program, ct.byref(length))
                log = ct.create_string_buffer(max(1, length.value))
                get_log(program, log)
                raise GPUUnavailable('CUDA 密码内核编译失败：'+log.value.decode('utf8', 'replace')[-2500:])
            length = ct.c_size_t()
            self.check(cubin_size(program, ct.byref(length)), 'CUDA CUBIN 编译结果查询')
            if not 0<length.value<64*1024*1024:
                raise GPUUnavailable('CUDA CUBIN 编译结果大小无效；GPU 任务已停止')
            output = ct.create_string_buffer(length.value)
            self.check(get_cubin(program, output), 'CUDA CUBIN 编译结果读取')
            cubin=output.raw
            cache_dir.mkdir(parents=True,exist_ok=True)
            staged=cache_dir/(cache_key+f'.{os.getpid()}.tmp')
            staged.write_bytes(cubin)
            os.replace(staged,cache_path)
            self.info['cubin_sha256']=hashlib.sha256(cubin).hexdigest()
            return cubin
        finally:
            destroy(ct.byref(program))

    def execute(self, name, inputs, output_sizes, arguments, *, rows, block=None):
        """Execute checked kernels; reuse bounded buffers, never fall back to CPU."""
        if type(rows) is not int or not 1 <= rows <= 20_000:
            raise ValueError('GPU batch rows must be between 1 and 20000')
        block = self.block_size if block is None else block
        if type(block) is not int or not 1 <= block <= 1024:
            raise ValueError('GPU block size must be between 1 and 1024')
        if any(type(size) is not int or size < 1 for size in output_sizes):
            raise ValueError('GPU output sizes must be positive integers')
        started = time.perf_counter()
        counters = self._empty_stats()
        with self.lock:
            counters['lock_wait_seconds'] = time.perf_counter()-started
            temporary, holders, parameters, used_keys = [], [], [], set()
            if self._closed:
                raise GPUUnavailable('CUDA 运行时已关闭')
            scheduling = self._scheduler.acquire() if self._scheduler is not None else nullcontext(0.0)
            try:
                with scheduling as waiting:
                    counters['scheduler_wait_seconds'] = waiting
                    phase = time.perf_counter()
                    self.check(self._set(self.context), 'CUDA 上下文选择')
                    function = self.functions.get(name)
                    if function is None:
                        function = ct.c_void_p()
                        self.check(self._function(ct.byref(function), self.module, name.encode()), 'CUDA 内核查找')
                        self.functions[name] = function
                        self._record_kernel_resources(name, function)
                    counters['setup_seconds'] += time.perf_counter()-phase
                    for index, blob in enumerate(inputs):
                        raw = bytes(blob)
                        phase = time.perf_counter()
                        pointer = self._buffer((name, 'input', index), len(raw), used_keys, temporary, counters)
                        counters['allocation_seconds'] += time.perf_counter()-phase
                        host = ct.create_string_buffer(raw)
                        holders.append(host)
                        phase = time.perf_counter()
                        self.check(self._upload(pointer, host, len(raw)), 'GPU 输入传输')
                        counters['upload_seconds'] += time.perf_counter()-phase
                        counters['upload_bytes'] += len(raw)
                        parameters.append(pointer)
                    outputs = []
                    for index, size in enumerate(output_sizes):
                        phase = time.perf_counter()
                        pointer = self._buffer((name, 'output', index), size, used_keys, temporary, counters)
                        counters['allocation_seconds'] += time.perf_counter()-phase
                        # Invalid-input kernels may only write the rejection flag.
                        # Clearing prevents prior output bytes surviving reuse.
                        phase = time.perf_counter()
                        self.check(self._clear(pointer, 0, size), 'GPU 输出显存清零')
                        counters['output_clear_seconds'] += time.perf_counter()-phase
                        outputs.append((pointer, size))
                        parameters.append(pointer)
                    parameters.extend(ct.c_uint(value) for value in arguments)
                    addresses = (ct.c_void_p*len(parameters))(*(ct.cast(ct.byref(value), ct.c_void_p)
                                                               for value in parameters))
                    begin, end = self._events
                    phase = time.perf_counter()
                    self.check(self._event_record(begin, None), 'CUDA 开始计时')
                    self.check(self._launch(function, (rows+block-1)//block, 1, 1, block, 1, 1, 0,
                                            None, addresses, None), 'GPU 密码批量运算')
                    self.check(self._event_record(end, None), 'CUDA 结束计时')
                    counters['launch_seconds'] += time.perf_counter()-phase
                    phase = time.perf_counter()
                    self.check(self._event_sync(end), 'GPU 密码结果同步')
                    counters['sync_seconds'] += time.perf_counter()-phase
                    elapsed = ct.c_float()
                    self.check(self._event_elapsed(ct.byref(elapsed), begin, end), 'CUDA 内核计时读取')
                    counters['kernel_seconds'] += elapsed.value/1000.0
                    result = []
                    for pointer, size in outputs:
                        host = ct.create_string_buffer(size)
                        phase = time.perf_counter()
                        self.check(self._download(host, pointer, size), 'GPU 结果传输')
                        counters['download_seconds'] += time.perf_counter()-phase
                        counters['download_bytes'] += size
                        result.append(host.raw)
                    for pointer in reversed(temporary):
                        phase = time.perf_counter()
                        self.check(self._free(pointer), 'GPU 临时显存释放')
                        counters['allocation_seconds'] += time.perf_counter()-phase
                        counters['frees'] += 1
                    temporary.clear()
            except BaseException as exc:
                # Failed batches never count as successful work or return partial
                # output. Discard retained pointers after a CUDA failure.
                self._release_pool()
                for pointer in reversed(temporary):
                    self._free(pointer)
                failed_wall = time.perf_counter()-started
                self.stats['failed_batches'] += 1
                self.stats['failed_wall_seconds'] += failed_wall
                kernel = self.kernel_stats.setdefault(name, self._empty_stats())
                kernel['failed_batches'] += 1
                kernel['failed_wall_seconds'] += failed_wall
                self.verified = False
                self.info['verified'] = False
                self.info['reason'] = str(exc)
                raise
            counters['batches'] = 1
            counters['rows'] = rows
            counters['host_wall_seconds'] = time.perf_counter()-started
            kernel = self.kernel_stats.setdefault(name, self._empty_stats())
            for key, value in counters.items():
                self.stats[key] += value
                kernel[key] += value
            return result

    def execute_batched(self,name,inputs,input_strides,output_strides,arguments,*,rows,chunk=None,block=None):
        """Bound each launch; shared bases/challenges use a None input stride."""
        if type(rows) is not int or not 1<=rows<=20_000:
            raise ValueError('GPU batch rows must be between 1 and 20000')
        chunk = self.chunk_size if chunk is None else chunk
        if type(chunk) is not int or not 1 <= chunk <= 20_000:
            raise ValueError('GPU chunk size must be between 1 and 20000')
        if len(inputs)!=len(input_strides) or any(
                stride is not None and len(blob)!=rows*stride for blob,stride in zip(inputs,input_strides)):
            raise ValueError('GPU input row length mismatch')
        channels=[bytearray() for _ in output_strides]
        for start in range(0,rows,chunk):
            count=min(chunk,rows-start)
            blocks=[blob if stride is None else blob[start*stride:(start+count)*stride]
                    for blob,stride in zip(inputs,input_strides)]
            result=self.execute(name,blocks,[stride*count for stride in output_strides],
                                [count,*arguments],rows=count,block=block)
            for channel,blob in zip(channels,result):
                channel.extend(blob)
        return [bytes(channel) for channel in channels]


def compute_capabilities():
    global _probe
    cpu_name = system_info.cpu_name()
    if _runtime is not None:
        gpu = {**_runtime.info, 'hardware_available':True,
               'available':_runtime.verified, 'verified':_runtime.verified,
               'driver_available':True, 'compiler_available':True}
        if not _runtime.verified:
            gpu['reason']=gpu.get('reason') or 'CUDA 密码内核尚未通过自动自检；CPU 计算可用'
    elif _probe is not None:
        gpu = dict(_probe)
    else:
        hardware = detect_gpu_hardware()
        compiler_available = False
        reason = hardware['reason']
        if hardware['driver_available']:
            try:
                _nvrtc_path()
                compiler_available = True
                reason = '正在等待 CUDA 密码内核自动自检；CPU 计算可用'
            except GPUUnavailable as exc:
                reason = str(exc) + '；请在部署阶段安装 gpu 依赖，CPU 计算可用'
        gpu = {**hardware, 'available':False, 'backend':'cuda_nvrtc', 'verified':False,
               'compiler_available':compiler_available, 'accelerated_operations':[], 'reason':reason}
        _probe = dict(gpu)
    if _runtime is None and gpu.get('available'):
        gpu={**gpu,'hardware_available':True,'available':False,
             'reason':'CUDA 密码内核尚未通过自动自检；CPU 计算可用'}
    return {'cpu': {'available': True, 'name': cpu_name, 'logical_processors': os.cpu_count()}, 'gpu': gpu}


def runtime():
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = CudaRuntime()
        return _runtime


def _rows(values, size):
    from dgfl.crypto import backend as b
    if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= 660_000:
        raise ValueError('invalid GPU batch size')
    return b''.join(b._blob(value, size) for value in values)


def _flags(raw, count):
    if len(raw) != count*4:
        raise GPUUnavailable('GPU 核验结果长度不正确')
    return [int.from_bytes(raw[index:index+4], 'little') for index in range(0, len(raw), 4)]


def gt_validate(values):
    data = _rows(values, 576)
    flags, = runtime().execute_batched('dgfl_gt_validate',[data],[576],[4],[],rows=len(values))
    return _flags(flags, len(values))


def gt_pow_batch(values, scalars, *, check_inputs=True):
    from dgfl.crypto import backend as b
    if len(values) != len(scalars):
        raise ValueError('GPU GT batch length mismatch')
    exponents = [b.scalar_dump(value) for value in scalars]
    data = _rows(values, 576)
    result, flags = runtime().execute_batched('dgfl_gt_pow_batch',[data,_rows(exponents,32)],
                                            [576,32],[576,4],[int(check_inputs)],rows=len(values))
    if not all(value == 1 for value in _flags(flags, len(values))):
        raise ValueError('GPU GT input is noncanonical or outside the prime-order subgroup')
    return [result[index:index+576] for index in range(0, len(result), 576)]


def gt_product_powers_batch(rows, weights, *, check_inputs=True):
    from dgfl.crypto import backend as b
    # Recovery uses one D term in addition to at most 32 cloud E terms.
    if not rows or not weights or len(weights) > 33 or any(len(row) != len(weights) for row in rows):
        raise ValueError('GPU GT product batch length mismatch')
    data = _rows([value for row in rows for value in row], 576)
    exponents = _rows([b.scalar_dump(weight) for _row in rows for weight in weights], 32)
    result, flags = runtime().execute_batched('dgfl_gt_product_powers_batch',[data,exponents],
                                            [576*len(weights),32*len(weights)],[576,4],
                                            [len(weights),int(check_inputs)],rows=len(rows),
                                            chunk=max(64,1024//len(weights)))
    if not all(value == 1 for value in _flags(flags, len(rows))):
        raise ValueError('GPU GT product contains a noncanonical or invalid subgroup element')
    return [result[index:index+576] for index in range(0, len(result), 576)]


def g1_msm_batch(rows, scalars):
    from dgfl.crypto import backend as b
    if not rows or not scalars or len(scalars) > 4 or any(len(row) != len(scalars) for row in rows):
        raise ValueError('GPU G1 batch length mismatch')
    points = _rows([b.g1_load(value).to_xy_bytes_le() for row in rows for value in row], 96)
    exponents = _rows([b.scalar_dump(value) for _row in rows for value in scalars], 32)
    result, flags = runtime().execute_batched('dgflow_g1_msm',[points,exponents],
                                            [96*len(scalars),32*len(scalars)],[96,4],
                                            [len(scalars)],rows=len(rows))
    if not all(value == 1 for value in _flags(flags, len(rows))):
        raise ValueError('GPU G1 MSM input is invalid')
    points = [result[index:index+96] for index in range(0, len(result), 96)]
    return [b.G1.identity() if raw == bytes(96) else b.G1.from_xy_bytes_le(raw) for raw in points]


def require_gpu():
    """Run differential public vectors once in each participating process."""
    from dgfl.crypto import backend as b
    device = runtime()
    with device.lock:
        if not device.verified:
            base = b.gt_dump(b.GT_BASE)
            exponents = [0, 1, 7, b.ORDER-1]
            actual = gt_pow_batch([base]*len(exponents), exponents)
            expected = [b.gt_dump(b.gt_pow(b.GT_BASE, value)) for value in exponents]
            if actual != expected or gt_validate([base, bytes(576)]) != [1, 0]:
                raise GPUUnavailable('GPU GT 算术自检与 CPU 精确结果不一致')
            rows=[[b.G,b.H,b.G,b.H],[b.H,b.G,b.H,b.G],
                  [b.G,b.G,b.H,b.H],[b.G1.identity(),b.G,b.G1.identity(),b.H]]
            weights=[0,1,7,b.ORDER-1]
            points=g1_msm_batch([[b.g1_dump(point) for point in row] for row in rows],weights)
            expected=[b.g1_msm(row,weights) for row in rows]
            if points!=expected:
                raise GPUUnavailable('GPU 椭圆曲线算术自检与 CPU 精确结果不一致')
            coefficients = [b.g1_msm([b.G,b.H],[5,7]), b.g1_msm([b.G,b.H],[2,3])]
            commitment_a = b.g1_msm([b.G,b.H],[13,17])
            polynomial_xy = _rows([point.to_xy_bytes_le() for point in coefficients],96)
            responses = [13+19*11,17+19*16,14+19*11,17+19*16]
            flags, = device.execute_batched('dgflow_g1_verify_polynomial',
                [polynomial_xy*2,commitment_a.to_xy_bytes_le()*2,
                 _rows([b.scalar_dump(value) for value in responses],32),
                 b.G.to_xy_bytes_le()+b.H.to_xy_bytes_le(), b.scalar_dump(19)],
                [192,96,64,None,None],[4],[2,3],rows=2)
            if _flags(flags,2)!=[2,1]:
                raise GPUUnavailable('GPU 融合椭圆曲线验证自检与 CPU 精确结果不一致')
            device.verified = True
    return {**device.info, 'verified': True}


def _checked_g1_coordinates(values, workers=1):
    """Canonical, subgroup-checked public G1 bytes to bounded affine batches.

    New native extensions return the exact same 96-byte little-endian XY
    layout as the checked CPU wrapper. An older extension retains that wrapper;
    a malformed native result is an error, never a reason to skip checking.
    """
    from dgfl.crypto import backend as b
    b._batch_workers(workers)
    if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= 640_000:
        raise ValueError('invalid GPU G1 coordinate batch size')
    native = b._native_batch('checked_g1_coordinates_batch')
    chunks = []
    for start in range(0, len(values), 20_000):
        batch = [b._blob(value, 48) for value in values[start:start+20_000]]
        if native is not None:
            chunks.append(b._blob(native(batch, workers=workers), 96*len(batch)))
        else:
            chunks.append(b''.join(b.g1_load(value).to_xy_bytes_le() for value in batch))
    return b''.join(chunks)


class _Polynomial:
    __slots__ = ('_commitments', '_points', 'count', 'encoded', 'owner', 'terms')

    def __init__(self, owner, commitments, encoded):
        self.owner = owner
        self._commitments = commitments
        self._points = None
        self.count = len(commitments)
        self.terms = len(commitments[0])
        self.encoded = encoded

    @property
    def points(self):
        # Compatibility for arithmetic inspection/tests. Production kernels
        # consume already checked immutable bytes without Python point objects.
        if self._points is None:
            from dgfl.crypto import backend as b
            self._points = tuple(tuple(b.g1_load(value) for value in row) for row in self._commitments)
        return self._points


class GPUAggregateVerifier:
    """Checked CPU parsing plus exact, batched CUDA G1/GT equations."""

    def __init__(self, g, h, base, *, workers=1):
        from dgfl.crypto import backend as b
        self.workers = b._batch_workers(workers)
        require_gpu()
        self.g, self.h = b.g1_load(g), b.g1_load(h)
        self.bases_xy = self.g.to_xy_bytes_le()+self.h.to_xy_bytes_le()
        self.base = b._blob(base, 576)
        if self.g == b.G1.identity() or self.h == b.G1.identity() or gt_validate([self.base]) != [1]:
            raise ValueError('degenerate or invalid aggregate public bases')
        self.identity = object()
        table, flags = runtime().execute('dgfl_gt_prepare_public_table', [self.base],
                                        [64*16*576, 64*4], [64], rows=64)
        if _flags(flags, 64) != [1]*64:
            raise ValueError('GPU public GT table preparation failed')
        self._public_table = table

    def polynomial(self, commitments, constants):
        from dgfl.crypto import backend as b
        if (not isinstance(commitments, (list, tuple)) or not commitments or len(commitments) > 20_000
                or not isinstance(constants, (list, tuple)) or len(constants) != len(commitments)
                or not isinstance(commitments[0], (list, tuple)) or not 2 <= len(commitments[0]) <= 32
                or any(not isinstance(row, (list, tuple)) or len(row) != len(commitments[0]) for row in commitments)):
            raise ValueError('invalid GPU aggregate polynomial dimensions')
        raw = tuple(tuple(b._blob(value, 48) for value in row) for row in commitments)
        anchors = [b._blob(value, 48) for value in constants]
        if any(row[0] != constant for row, constant in zip(raw, anchors)):
            raise ValueError('aggregate key commitment violates trusted DKG')
        # Canonical byte equality binds the trusted anchor to coefficient zero.
        # That coefficient passes the same full subgroup decoder as every other
        # coefficient; no second decode or reduction substitutes for the check.
        encoded = _checked_g1_coordinates([value for row in raw for value in row], self.workers)
        return _Polynomial(self.identity, raw, encoded)

    def verify(self, polynomial, cloud_id, challenge, a, proof_b, e, responses):
        return self.verify_many([(polynomial, cloud_id, challenge, a, proof_b, e, responses)])[0]

    def _checked_job(self, job):
        from dgfl.crypto import backend as b
        if not isinstance(job, (tuple, list)) or len(job) != 7:
            raise ValueError('invalid GPU aggregate verification job')
        polynomial, cloud_id, challenge, a, proof_b, e, responses = job
        if not isinstance(polynomial, _Polynomial) or polynomial.owner is not self.identity:
            raise ValueError('aggregate polynomial/verifier binding mismatch')
        count = polynomial.count
        if (type(cloud_id) is not int or not 1 <= cloud_id <= 32
                or any(len(value) != count for value in (a, proof_b, e, responses))
                or any(len(row) != 2 for row in responses)):
            raise ValueError('aggregate proof dimension or cloud mismatch')
        challenge_raw = b._blob(challenge, 32)
        b.scalar_load(challenge_raw)
        pairs = [(b._blob(row[0], 32), b._blob(row[1], 32)) for row in responses]
        for zs, zr in pairs:
            b.scalar_load(zs)
            b.scalar_load(zr)
        return {'count': count, 'terms': polynomial.terms, 'polynomial': polynomial.encoded,
                'a': _checked_g1_coordinates(a, self.workers),
                'responses': b''.join(zs+zr for zs, zr in pairs),
                'zs': b''.join(zs for zs, _zr in pairs),
                'b': _rows(proof_b, 576), 'e': _rows(e, 576),
                'challenge': challenge_raw*count,
                'cloud': cloud_id.to_bytes(4, 'little')*count}

    def verify_many(self, jobs):
        """Batch independent original equations; return E only after all pass.

        Polynomials and A remain fully checked by the CPU subgroup decoder.
        Different cloud IDs/challenges are kept per coordinate. Launch buffers
        stay bounded even when several 20,000-coordinate statements are given.
        """
        if not isinstance(jobs, (tuple, list)) or not 1 <= len(jobs) <= 1024:
            raise ValueError('invalid GPU aggregate verification batch')
        entries = [self._checked_job(job) for job in jobs]
        device = runtime()
        # At 255 registers per GT thread, a 650-row launch offers less than one
        # warp per SM. A larger multi-proof launch lets more resident warps hide
        # arithmetic/local-memory latency. Preserve a configured larger budget.
        chunk = max(device.chunk_size, 4096)

        def check_group(segments, terms):
            strides = {'polynomial': 96*terms, 'a': 96, 'responses': 64,
                       'zs': 32, 'b': 576, 'e': 576, 'challenge': 32, 'cloud': 4}
            buffers = {name: b''.join(entry[name][start*stride:end*stride]
                                     for entry,start,end in segments)
                       for name,stride in strides.items()}
            count = sum(end-start for _entry,start,end in segments)
            g1_flags, = device.execute_batched('dgflow_g1_verify_polynomial_many',
                [buffers['polynomial'],buffers['a'],buffers['responses'],self.bases_xy,
                 buffers['challenge'],buffers['cloud']],
                [96*terms,96,64,None,32,4],[4],[terms],rows=count,chunk=chunk)
            if _flags(g1_flags,count) != [2]*count:
                raise ValueError('aggregate pairing-image proof failed')
            gt_flags, = device.execute_batched('dgfl_gt_verify_many',
                [buffers['b'],buffers['e'],buffers['zs'],self.base,self._public_table,buffers['challenge']],
                [576,576,32,None,None,32],[4],[],rows=count,chunk=chunk)
            if _flags(gt_flags,count) != [1]*count:
                raise ValueError('aggregate pairing-image proof failed')

        for terms in sorted({entry['terms'] for entry in entries}):
            segments, rows = [], 0
            for entry in (v for v in entries if v['terms'] == terms):
                start = 0
                while start < entry['count']:
                    end = min(entry['count'], start+20_000-rows)
                    segments.append((entry,start,end)); rows += end-start; start = end
                    if rows == 20_000:
                        check_group(segments,terms); segments,rows = [],0
            if segments:
                check_group(segments,terms)
        return [[entry['e'][i:i+576] for i in range(0,len(entry['e']),576)] for entry in entries]
