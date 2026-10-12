"""Authenticated node monitoring, isolated from stateful role execution.

Only public scalar diagnostics are journalled.  Driver and filesystem failures
are monitoring failures and must never change a training or proof result.
"""

import bisect
import json
import math
import os
import re
import stat
import threading
import time
from collections import deque
from contextlib import suppress
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import psutil

from dgfl.experiments.hardware import CPU_FIELDS, GPU_FIELDS, HardwareSampler

_IDENTIFIER = re.compile(r'[A-Za-z0-9_-]{1,100}')
_SYSTEM_FIELDS = ('memory_used_bytes', 'memory_total_bytes', 'memory_percent',
                  'disk_used_bytes', 'disk_total_bytes', 'disk_percent')


def _identifier(value):
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError('invalid telemetry task identifier')
    return value


def _number(value):
    try:
        return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else None
    except OverflowError:
        return None


def _scalars(value, fields):
    value = value if isinstance(value, dict) else {}
    return {field: _number(value.get(field)) for field in fields}


def _timestamp(value):
    if not isinstance(value, str) or len(value) > 64:
        raise ValueError('invalid telemetry timestamp')
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('telemetry timestamp requires a timezone')
    return value


def _safe_path(path, root):
    """Reject junctions as well as symlinks before opening monitoring files."""
    path, root = Path(path).absolute(), Path(root).absolute()
    if not path.is_relative_to(root):
        raise ValueError('telemetry path leaves runtime')
    current = root
    for component in (None, *path.relative_to(root).parts):
        if component is not None:
            current = current / component
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400):
            raise ValueError('telemetry paths must not contain links or junctions')
    return path


class _Journal:
    def __init__(self, path, runtime):
        self.path, self.runtime = path, runtime
        self.lock = threading.Lock()
        self.writer = threading.Lock()
        self.events = deque(maxlen=512)
        self.offsets = []
        self.epoch_steps = set()
        self.training_rounds = set()
        self.error = None
        self.last_sequence = 0
        self.last_sample = -1
        self.last_elapsed = 0.
        self.tail_needs_newline = False

    def load(self):
        try:
            path = _safe_path(self.path, self.runtime)
            if not path.exists():
                return
            with path.open('rb') as stream:
                while True:
                    offset = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    self.tail_needs_newline = not line.endswith(b'\n')
                    try:
                        event = json.loads(line)
                        sequence = event['sequence']
                        if (type(sequence) is not int or sequence <= self.last_sequence
                                or event['kind'] not in ('resource', 'epoch', 'training')
                                or not isinstance(event['payload'], dict)):
                            continue
                        timestamp = _timestamp(event['time'])
                        payload = _clean_payload(event['kind'], event['payload'])
                        if sequence != self.last_sequence + 1:
                            self.error = 'Telemetry journal has missing sequence numbers; later readings recovered.'
                        self._accept({'sequence': sequence, 'kind': event['kind'],
                                      'time': timestamp, 'payload': payload}, offset)
                    except (ValueError, KeyError, TypeError, UnicodeDecodeError):
                        self.error = 'Unreadable telemetry journal row was skipped.'
        except (OSError, ValueError) as exc:
            self.error = f'{type(exc).__name__}: telemetry journal unavailable'

    def _accept(self, event, offset=None):
        self.events.append(event)
        self.last_sequence = event['sequence']
        if offset is not None and (not self.offsets or event['sequence'] - self.offsets[-1][0] >= 128):
            self.offsets.append((event['sequence'], offset))
        if event['kind'] == 'epoch':
            self.epoch_steps.add(event['payload']['step'])
        elif event['kind'] == 'training':
            self.training_rounds.add(event['payload']['round_id'])
        else:
            self.last_sample = max(self.last_sample, event['payload']['sample_index'])
            self.last_elapsed = max(self.last_elapsed, event['payload']['elapsed_s'])

    def append(self, kind, payload):
        payload = _clean_payload(kind, payload)
        with self.writer:
            with self.lock:
                if kind == 'epoch' and payload['step'] in self.epoch_steps:
                    return False
                if kind == 'training' and payload['round_id'] in self.training_rounds:
                    return False
                if kind == 'resource':
                    # A bounded stop can leave a driver thread finishing while
                    # a new session starts. Allocate sample steps centrally.
                    payload['sample_index'] = self.last_sample + 1
                    payload['elapsed_s'] = max(payload['elapsed_s'], self.last_elapsed)
                event = {'sequence': self.last_sequence + 1, 'kind': kind,
                         'time': datetime.now(UTC).isoformat(timespec='milliseconds'),
                         'payload': payload}
            offset = None
            try:
                path = _safe_path(self.path, self.runtime)
                path.parent.mkdir(parents=True, exist_ok=True)
                _safe_path(path, self.runtime)
                raw = json.dumps(event, ensure_ascii=True, allow_nan=False, separators=(',', ':')).encode('utf8') + b'\n'
                with path.open('ab') as stream:
                    if self.tail_needs_newline:
                        stream.write(b'\n')
                    offset = stream.tell()
                    stream.write(raw)
                    stream.flush()
                self.tail_needs_newline = False
            except (OSError, ValueError) as exc:
                offset = None
                self.tail_needs_newline = True
                with self.lock:
                    self.error = f'{type(exc).__name__}: telemetry journal write failed'
                # Keep the measurement in the live cache even if persistence is
                # unavailable. Its sequence remains monotonic in this process.
            with self.lock:
                self._accept(event, offset)
            return True

    def page(self, after, limit):
        """Read the live bounded cache, or seek a sparse index for old cursors."""
        with self.lock:
            cached = deepcopy(list(self.events))
            last_sequence, error = self.last_sequence, self.error
            needs_disk = bool(cached and after < cached[0]['sequence'] - 1)
            index = bisect.bisect_right(self.offsets, after + 1, key=lambda item: item[0]) - 1
            offset = self.offsets[index][1] if index >= 0 else 0
        rows = []
        if needs_disk:
            try:
                with _safe_path(self.path, self.runtime).open('rb') as stream:
                    stream.seek(offset)
                    while len(rows) < limit:
                        line = stream.readline()
                        if not line:
                            break
                        try:
                            event = json.loads(line)
                            sequence = event['sequence']
                            if type(sequence) is not int or sequence <= (rows[-1]['sequence'] if rows else after):
                                continue
                            timestamp = _timestamp(event['time'])
                            if sequence != (rows[-1]['sequence'] if rows else after) + 1:
                                error = 'Telemetry journal has missing sequence numbers; later readings recovered.'
                            rows.append({'sequence': sequence, 'kind': event['kind'], 'time': timestamp,
                                         'payload': _clean_payload(event['kind'], event['payload'])})
                        except (ValueError, KeyError, TypeError, UnicodeDecodeError):
                            error = 'Unreadable telemetry journal row was skipped.'
            except (OSError, ValueError):
                error = 'Telemetry history unavailable; only cached readings can be returned.'
        for event in cached:
            if len(rows) >= limit:
                break
            if event['sequence'] > (rows[-1]['sequence'] if rows else after):
                rows.append(event)
        cursor = rows[-1]['sequence'] if rows else after
        result = {'events': rows, 'next_cursor': cursor, 'more': cursor < last_sequence}
        if error:
            result['error'] = error
        return result


def _clean_payload(kind, payload):
    if not isinstance(payload, dict):
        raise ValueError('invalid telemetry payload')
    if kind == 'training':
        if (type(payload.get('round_id')) is not int or payload['round_id'] < 1
                or type(payload.get('epochs')) is not int or payload['epochs'] < 0):
            raise ValueError('invalid training telemetry summary')
        return {'round_id': payload['round_id'], 'epochs': payload['epochs'],
                **_scalars(payload, ('metrics_eval_s', 'metrics_callback_s', 'training_s'))}
    if kind == 'epoch':
        for field in ('epoch', 'round_id', 'step', 'samples'):
            if type(payload.get(field)) is not int or payload[field] < (0 if field == 'samples' else 1):
                raise ValueError('invalid epoch telemetry')
        result = {field: payload[field] for field in ('epoch', 'round_id', 'step', 'samples')}
        result.update(_scalars(payload, ('accuracy', 'loss', 'elapsed_s', 'metrics_eval_s')))
        if result['accuracy'] is not None and result['accuracy'] > 1:
            result['accuracy'] = None
        return result
    if kind != 'resource' or type(payload.get('sample_index')) is not int or payload['sample_index'] < 0:
        raise ValueError('invalid resource telemetry')
    elapsed = _number(payload.get('elapsed_s'))
    if elapsed is None:
        raise ValueError('invalid resource telemetry elapsed time')
    result = {'sample_index': payload['sample_index'], 'elapsed_s': elapsed,
              'sampling_s': _number(payload.get('sampling_s')),
              'round_id': payload.get('round_id') if type(payload.get('round_id')) is int and payload['round_id'] >= 0 else 0,
              'stage': payload.get('stage', '')[:80] if isinstance(payload.get('stage', ''), str) else '',
              'processes': _scalars(payload.get('processes'), ('cpu_percent', 'rss_bytes'))}
    for device, fields in (('cpu', CPU_FIELDS), ('gpu', GPU_FIELDS), ('system', _SYSTEM_FIELDS)):
        if device in payload:
            result[device] = _scalars(payload[device], fields)
    if 'machine_sampling_s' in payload:
        result['machine_sampling_s'] = _number(payload['machine_sampling_s'])
    return result


class NodeTelemetry:
    """One resource sampler per node, with independently readable task journals."""

    def __init__(self, node_id, runtime):
        self.node_id = _identifier(node_id)
        self.runtime = Path(runtime)
        self.folder = self.runtime / 'nodes' / self.node_id / 'telemetry'
        self._lock = threading.RLock()
        self._journal_lock = threading.Lock()
        self._journals = {}
        self._contexts = {}
        self._active = None
        self._threads = set()
        self._closed = False

    def _journal(self, task):
        task = _identifier(task)
        with self._lock:
            existing = self._journals.get(task)
        if existing is not None:
            return existing
        # Cold history loading never holds the cache/context lock or business
        # lock. Concurrent readers share a single initialized journal.
        with self._journal_lock:
            with self._lock:
                existing = self._journals.get(task)
            if existing is None:
                existing = _Journal(self.folder / task / 'events.jsonl', self.runtime)
                existing.load()
                with self._lock:
                    self._journals[task] = existing
            return existing

    def start(self, payload):
        task = _identifier(payload.get('task_id'))
        interval = payload.get('interval_seconds', 5)
        if type(interval) not in (int, float) or not math.isfinite(interval) or not 0.1 <= interval <= 60:
            raise ValueError('telemetry interval must be between 0.1 and 60 seconds')
        if type(payload.get('collect_machine', False)) is not bool:
            raise ValueError('invalid telemetry machine collection flag')
        machine = payload.get('machine', '')
        if not isinstance(machine, str) or len(machine) > 200:
            raise ValueError('invalid telemetry machine name')
        journal = self._journal(task)
        with self._lock:
            if self._closed:
                raise ValueError('node telemetry is closed')
            if self._active is not None:
                if self._active['task_id'] != task:
                    raise ValueError('another telemetry task is active')
                return {'node_id': self.node_id, 'task_id': task, 'active': True}
            if self._threads:
                # stop() is bounded even if a driver is stuck. Keep at most
                # one sampler until its final probe and driver cleanup finish.
                raise ValueError('telemetry sampler cleanup is still pending')
            context = self._contexts.get(task, {'round_id': 0, 'stage': 'preparing'})
            state = {'task_id': task, 'machine': machine,
                     'collect_machine': payload.get('collect_machine', False),
                     'interval': float(interval), 'stop': threading.Event(),
                     'started': time.monotonic(), 'offset': journal.last_elapsed,
                     **context, 'previous': {}, 'previous_at': None,
                     'sample_index': journal.last_sample + 1, 'journal': journal}
            thread = threading.Thread(target=self._loop, args=(state,), daemon=True,
                                      name=f'dgflow-telemetry-{self.node_id}')
            state['thread'] = thread
            self._active = state
            self._threads.add(thread)
            thread.start()
        return {'node_id': self.node_id, 'task_id': task, 'active': True}

    def read(self, payload):
        task = _identifier(payload.get('task_id'))
        after, limit = payload.get('after', 0), payload.get('limit', 128)
        if type(after) is not int or after < 0 or type(limit) is not int or not 1 <= limit <= 256:
            raise ValueError('invalid telemetry cursor or limit')
        journal = self._journal(task)
        result = {'node_id': self.node_id, 'task_id': task, **journal.page(after, limit)}
        with self._lock:
            result['active'] = self._active is not None and self._active['task_id'] == task and not self._active['stop'].is_set()
        return result

    def stop(self, payload):
        task = _identifier(payload.get('task_id'))
        with self._lock:
            state = self._active
            if state is not None and state['task_id'] == task:
                state['stop'].set()
                self._active = None
            else:
                state = None
        if state is not None:
            state['thread'].join(timeout=1)
            if state['thread'].is_alive():
                with state['journal'].lock:
                    state['journal'].error = 'Telemetry sampler cleanup is still pending; bounded stop returned.'
        return {'node_id': self.node_id, 'task_id': task, 'active': False}

    def close(self):
        with self._lock:
            self._closed = True
            state = self._active
            if state is not None:
                state['stop'].set()
                self._active = None
            threads = list(self._threads)
        deadline = time.monotonic() + 1
        for thread in threads:
            thread.join(timeout=max(0., deadline - time.monotonic()))

    def set_context(self, task_id, round_id, stage):
        if not isinstance(task_id, str) or not _IDENTIFIER.fullmatch(task_id):
            return
        with self._lock:
            context = {'round_id': round_id if type(round_id) is int and round_id >= 0 else 0,
                       'stage': str(stage)[:80]}
            self._contexts[task_id] = context
            state = self._active
            if state is not None and state['task_id'] == task_id:
                state.update(context)

    def record_epoch(self, task_id, round_id, local_epochs, metrics):
        """Also accept pre-start epochs: training must not wait for monitoring."""
        try:
            if type(round_id) is not int or round_id < 1 or type(local_epochs) is not int or local_epochs < 1:
                raise ValueError('invalid epoch context')
            epoch = metrics.get('epoch')
            if type(epoch) is not int or not 1 <= epoch <= local_epochs:
                raise ValueError('invalid local epoch')
            row = {**metrics, 'round_id': round_id, 'step': (round_id - 1) * local_epochs + epoch}
            return self._journal(task_id).append('epoch', row)
        except Exception:
            # This optional callback must never escape into model training.
            return False

    def record_training_summary(self, task_id, round_id, local_epochs, result):
        """Publish observation overhead through telemetry rather than train RPC."""
        try:
            return self._journal(task_id).append('training', {
                'round_id': round_id, 'epochs': local_epochs,
                'metrics_eval_s': result.get('metrics_eval_s'),
                'metrics_callback_s': result.get('metrics_callback_s'),
                'training_s': result.get('training_s')})
        except Exception:
            return False

    def _processes(self, state):
        sampled_at = time.monotonic()
        complete = True
        processes = []
        try:
            root = psutil.Process(os.getpid())
            root_created = root.create_time()
            processes.append(root)
            for child in root.children(recursive=True):
                try:
                    if any(parent.pid == root.pid and parent.create_time() == root_created for parent in child.parents()):
                        processes.append(child)
                except psutil.NoSuchProcess:
                    continue
                except (OSError, psutil.Error):
                    complete = False
        except (OSError, psutil.Error):
            complete = False
        current, rss = {}, 0
        rss_complete = cpu_complete = complete
        for process in processes:
            try:
                created = process.create_time()
                cpu = process.cpu_times()
                current[(process.pid, created)] = cpu.user + cpu.system
            except psutil.NoSuchProcess:
                continue
            except (OSError, psutil.Error):
                cpu_complete = False
            try:
                rss += process.memory_info().rss
            except psutil.NoSuchProcess:
                continue
            except (OSError, psutil.Error):
                rss_complete = False
        try:
            cores = psutil.cpu_count(logical=True)
        except (OSError, psutil.Error):
            cores = None
        previous = state['previous']
        elapsed = sampled_at - state['previous_at'] if state['previous_at'] is not None else None
        cpu_percent = None
        if (cpu_complete and current and cores and elapsed is not None and elapsed > 0
                and all(key in previous and value >= previous[key] for key, value in current.items())):
            cpu_percent = min(100., max(0., sum(value - previous[key] for key, value in current.items()) / elapsed / cores * 100))
        state['previous'], state['previous_at'] = current, sampled_at
        return {'cpu_percent': cpu_percent, 'rss_bytes': rss if rss_complete and processes else None}

    def _system(self):
        result = dict.fromkeys(_SYSTEM_FIELDS)
        try:
            memory = psutil.virtual_memory()
            result.update(memory_used_bytes=memory.used, memory_total_bytes=memory.total, memory_percent=memory.percent)
        except (OSError, psutil.Error):
            pass
        try:
            path = self.runtime.absolute()
            while not path.exists() and path.parent != path:
                path = path.parent
            disk = psutil.disk_usage(str(path))
            result.update(disk_used_bytes=disk.used, disk_total_bytes=disk.total, disk_percent=disk.percent)
        except (OSError, psutil.Error):
            pass
        return result

    def _sample(self, state, hardware):
        started = time.perf_counter()
        row = {'sample_index': state['sample_index'],
               'elapsed_s': state['offset'] + time.monotonic() - state['started'],
               'processes': self._processes(state)}
        with self._lock:
            row.update(round_id=state['round_id'], stage=state['stage'])
        if state['collect_machine']:
            machine_started = time.perf_counter()
            row.update(hardware.sample() if hardware is not None else {'cpu': {}, 'gpu': {}})
            row['system'] = self._system()
            row['machine_sampling_s'] = time.perf_counter() - machine_started
        row['sampling_s'] = time.perf_counter() - started
        state['journal'].append('resource', row)
        state['sample_index'] += 1

    def _loop(self, state):
        hardware = None
        try:
            if state['collect_machine']:
                hardware = HardwareSampler()
            while True:
                try:
                    self._sample(state, hardware)
                except Exception as exc:
                    with state['journal'].lock:
                        state['journal'].error = f'{type(exc).__name__}: telemetry sampling failed'
                if state['stop'].wait(state['interval']):
                    break
            # Final reading captures the end of a short run as well as long
            # runs. It remains on this daemon if a device driver is unresponsive.
            with suppress(Exception):
                self._sample(state, hardware)
        except Exception as exc:
            with state['journal'].lock:
                state['journal'].error = f'{type(exc).__name__}: telemetry sampler unavailable'
        finally:
            if hardware is not None:
                with suppress(Exception):
                    hardware.close()
            with self._lock:
                self._threads.discard(threading.current_thread())
                if self._active is state:
                    self._active = None
