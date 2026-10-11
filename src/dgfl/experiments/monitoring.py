"""Bounded, best-effort observability over the existing authenticated node RPC.

Business RPC counters and execution pools are deliberately independent. HTTP
readers only receive detached summaries, never a sensor query or raw history.
"""
from __future__ import annotations

import json
import math
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from dgfl.experiments.hardware import CPU_FIELDS, GPU_FIELDS
from dgfl.experiments.tensorboard import _no_link, _safe_open
from dgfl.transport.client import RPCClient

_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z')
_NODE = re.compile(r'(client|authority|aggregator)[1-9][0-9]*\Z')
_RESERVED = {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(10)), *(f'LPT{i}' for i in range(10))}
SYSTEM_FIELDS = ('memory_used_bytes', 'memory_total_bytes', 'memory_percent',
                 'disk_used_bytes', 'disk_total_bytes', 'disk_percent')


def utc():
    return datetime.now(UTC).isoformat(timespec='milliseconds')


def component(value):
    if (not isinstance(value, str) or not _ID.fullmatch(value) or value.endswith('.')
            or value.split('.')[0].upper() in _RESERVED):
        raise ValueError('invalid monitoring identifier')
    return value


def number(value):
    try:
        return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None
    except OverflowError:
        return None


def safe_directory(runtime, run_id, *, create=False):
    directory = Path(runtime).resolve() / 'monitoring' / component(run_id)
    for parent in reversed((directory, *directory.parents)):
        _no_link(parent)
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def frozen_nodes(cluster, config):
    """Freeze the experiment's identities, rather than consult a later deployment."""
    counts = {role: config.get(f'{role}_count', 0) for role in ('client', 'authority', 'aggregator')}
    mapping = cluster.get('client_authorities', {})
    rows = []
    for node_id, entry in cluster.get('nodes', {}).items():
        match = _NODE.fullmatch(node_id)
        if not match:
            continue
        role = match[1]
        if counts[role] and int(node_id[len(role):]) > counts[role]:
            continue
        rows.append({'id': node_id, 'role': role,
                     'machine': entry.get('machine', 'local' if cluster.get('deployment') == 'single_host' else None),
                     'host': entry.get('url'), 'authority': mapping.get(node_id)})
    return rows


def monitor_snapshot(record, live=None):
    evidence = record.get('evidence') or {}
    cached = live if live is not None else evidence.get('node_monitoring')
    if isinstance(cached, dict):
        result = deepcopy(cached)
        result.update(run_id=record['run_id'], status=record.get('status'))
        if record.get('status') not in ('queued', 'running', 'stopping'):
            for row in [*result.get('nodes', []), *result.get('machines', [])]:
                if row.get('status') in ('online', 'starting'):
                    row['status'] = 'recorded' if row.get('last_seen') else 'missing'
        return result
    nodes = deepcopy(evidence.get('node_snapshot') or [])
    if not nodes:
        config = record.get('config') or {}
        members = evidence.get('members') or [f'client{i}' for i in range(1, (config.get('client_count') or 0) + 1)]
        names = [*members, *(f'authority{i}' for i in range(1, (config.get('authority_count') or 0) + 1)),
                 *(f'aggregator{i}' for i in range(1, (config.get('aggregator_count') or 0) + 1))]
        nodes = [{'id': name, 'role': _NODE.fullmatch(name)[1], 'machine': None, 'host': None, 'authority': None}
                 for name in names if isinstance(name, str) and _NODE.fullmatch(name)]
    for row in nodes:
        row.update(status='missing', last_seen=None, latest={}, training={}, error=None)
    return {'run_id': record['run_id'], 'status': record.get('status'), 'interval_seconds': 5,
            'nodes': nodes, 'machines': [], 'summary': {'bytes_sent': 0}, 'legacy': True}


def clean_event(event, local_epochs):
    """Accept only public numeric diagnostics, even from an authenticated peer."""
    if not isinstance(event, dict) or type(event.get('sequence')) is not int or event['sequence'] < 1:
        raise ValueError('invalid telemetry sequence')
    timestamp = event.get('time')
    if not isinstance(timestamp, str) or len(timestamp) > 64:
        raise ValueError('invalid telemetry timestamp')
    parsed = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('telemetry timestamp requires a timezone')
    raw = event.get('payload')
    if not isinstance(raw, dict):
        raise ValueError('invalid telemetry payload')
    kind = event.get('kind')
    if kind == 'epoch':
        epoch, round_id = raw.get('epoch'), raw.get('round_id')
        if (type(epoch) is not int or not 1 <= epoch <= local_epochs
                or type(round_id) is not int or round_id < 1):
            raise ValueError('invalid training epoch')
        step = (round_id - 1) * local_epochs + epoch
        if raw.get('step') != step:
            raise ValueError('epoch step differs from task')
        payload = {key: number(raw.get(key)) for key in
                   ('accuracy', 'loss', 'samples', 'elapsed_s', 'metrics_eval_s')}
        if payload['accuracy'] is not None and payload['accuracy'] > 1:
            raise ValueError('invalid training accuracy')
        payload.update(epoch=epoch, round_id=round_id, step=step)
    elif kind == 'training':
        round_id, epochs = raw.get('round_id'), raw.get('epochs')
        if (type(round_id) is not int or round_id < 1
                or type(epochs) is not int or epochs != local_epochs):
            raise ValueError('invalid training summary')
        payload = {key: number(raw.get(key)) for key in
                   ('metrics_eval_s', 'metrics_callback_s', 'training_s')}
        payload.update(round_id=round_id, epochs=epochs)
    elif kind == 'resource':
        if type(raw.get('sample_index')) is not int or raw['sample_index'] < 0:
            raise ValueError('invalid resource index')
        payload = {'sample_index': raw['sample_index'], 'elapsed_s': number(raw.get('elapsed_s')),
                   'round_id': raw.get('round_id') if type(raw.get('round_id')) is int else None,
                   'stage': str(raw.get('stage', ''))[:64], 'sampling_s': number(raw.get('sampling_s'))}
        if 'machine_sampling_s' in raw:
            payload['machine_sampling_s'] = number(raw['machine_sampling_s'])
        for group, fields in (('processes', ('cpu_percent', 'rss_bytes')),
                              ('cpu', CPU_FIELDS), ('gpu', GPU_FIELDS), ('system', SYSTEM_FIELDS)):
            values = raw.get(group)
            if isinstance(values, dict):
                payload[group] = {key: number(values.get(key)) for key in fields}
    else:
        raise ValueError('unknown telemetry event')
    return {'sequence': event['sequence'], 'kind': kind, 'time': timestamp, 'payload': payload}


def iter_monitor_events(runtime, run_id, offsets=None):
    """Stream persisted evidence; a partial final line never hides other files."""
    import os
    directory = safe_directory(runtime, run_id)
    if not directory.exists():
        return
    for path in sorted(directory.glob('*.jsonl')):
        _no_link(path)
        with os.fdopen(_safe_open(path, os.O_RDONLY), 'rb') as stream:
            size = path.stat().st_size
            offset = (offsets or {}).get(path.name, 0)
            stream.seek(offset if 0 <= offset <= size else 0)
            remaining = size - stream.tell()
            while remaining > 0:
                line = stream.readline(min(remaining, 16385))
                remaining -= len(line)
                if not line:
                    break
                if len(line) > 16384 or not line.endswith(b'\n'):
                    break
                try:
                    row = json.loads(line)
                    if (not isinstance(row, dict) or not _NODE.fullmatch(row.get('node_id', ''))
                            or not isinstance(row.get('event'), dict)):
                        continue
                    yield {**row, '_file': path.name, '_offset': stream.tell()}
                except (ValueError, TypeError):
                    continue


class RunNodeMonitor:
    def __init__(self, runtime, run_id, nodes, tensorboard, *, local_machine='local', local_epochs=1,
                 rpc_factory=RPCClient, interval_seconds=5, eligible_ids=None):
        self.runtime = Path(runtime)
        self.run_id = component(run_id)
        self.nodes = deepcopy(nodes)
        self.tensorboard = tensorboard
        self.local_machine = local_machine
        self.local_epochs = local_epochs
        self.interval_seconds = interval_seconds
        self._rpc_factory = rpc_factory
        self._rpc = None
        self._pool = None
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._cycle_lock = threading.Lock()
        self._active = set()
        self._cursors = dict.fromkeys((row['id'] for row in nodes), 0)
        self._session = uuid.uuid4().hex
        self._eligible = set(eligible_ids) if eligible_ids is not None else {row['id'] for row in nodes}
        self._sources = {}
        self._rows = {}
        for node in self.nodes:
            machine = node.get('machine')
            if machine is not None and machine != local_machine and node['id'] in self._eligible:
                self._sources.setdefault(machine, node['id'])
            self._rows[node['id']] = {**node, 'status': 'starting', 'last_seen': None,
                                     'latest': {}, 'training': {}, 'error': None}
            if node['id'] not in self._eligible:
                self._rows[node['id']].update(status='missing', error='节点未提供逐节点遥测能力')
        self._machines = {}
        self._summary = {'bytes_sent': 0, 'events_received': 0, 'collection_failures': 0,
                         'events_missing': 0, 'metrics_eval_s': 0., 'last_error': None,
                         'training_rounds': {}, 'log_failures': 0,
                         'node_sampling_s': None, 'node_samples_timed': 0,
                         'local_machine_sampling_s': None, 'local_machine_samples_timed': 0}
        self._epoch_seen = set()
        self._training_seen = set()
        self._local_seen = set()

    def _failure(self, node_id, error):
        detail = f'{type(error).__name__}: {error}'.replace(str(self.runtime.parent), '[workspace]')[:300]
        with self._lock:
            self._summary['collection_failures'] += 1
            self._summary['last_error'] = detail
            if node_id in self._rows:
                self._rows[node_id].update(status='unavailable', error=detail)
                machine = self._rows[node_id].get('machine')
                if self._sources.get(machine) == node_id and machine in self._machines:
                    self._machines[machine].update(status='unavailable', error=detail)

    def _append(self, node_id, event):
        import os
        directory = safe_directory(self.runtime, self.run_id, create=True)
        path = directory / f'{component(node_id)}.{self._session}.jsonl'
        raw = (json.dumps({'node_id': node_id, 'event': event}, allow_nan=False,
                          separators=(',', ':')) + '\n').encode('utf8')
        try:
            with os.fdopen(_safe_open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT), 'ab') as stream:
                stream.write(raw)
                stream.flush()
        except OSError:
            # Preserve any torn tail. A retry writes a separate segment so an
            # incomplete line cannot absorb subsequent intact measurements.
            self._session = uuid.uuid4().hex
            raise

    def observe_training(self, node_id, epochs):
        with self._lock:
            row = self._rows.get(node_id)
            if row is None:
                return
            for epoch in epochs:
                key = (node_id, epoch['step'])
                if key not in self._epoch_seen:
                    self._summary['metrics_eval_s'] += epoch.get('metrics_eval_s') or 0.
                    self._epoch_seen.add(key)
                if epoch['step'] >= row.get('training', {}).get('step', -1):
                    row['training'] = deepcopy(epoch)

    def _training_summary(self, node_id, payload):
        key = (node_id, payload['round_id'])
        if key in self._training_seen:
            return
        self._training_seen.add(key)
        totals = self._summary['training_rounds'].setdefault(str(payload['round_id']), {
            'clients_recorded': 0, 'metrics_eval_s': 0., 'metrics_callback_s': 0., 'training_s': 0.})
        totals['clients_recorded'] += 1
        for field in ('metrics_eval_s', 'metrics_callback_s', 'training_s'):
            # Missing observations remain explicitly incomplete rather than
            # silently becoming a fabricated zero duration.
            value = payload.get(field)
            totals[field] = totals[field] + value if value is not None and totals[field] is not None else None

    def _accept(self, node, reply):
        node_id = node['id']
        if (not isinstance(reply, dict) or reply.get('node_id') != node_id
                or reply.get('task_id') != self.run_id or not isinstance(reply.get('events'), list)
                or len(reply['events']) > 128):
            raise ValueError('node telemetry does not match requested task')
        with self._lock:
            cursor = self._cursors[node_id]
        for raw in reply['events']:
            event = clean_event(raw, self.local_epochs)
            if event['sequence'] <= cursor:
                continue
            if event['kind'] == 'resource' and self._sources.get(node.get('machine')) != node_id:
                for group in ('cpu', 'gpu', 'system', 'machine_sampling_s'):
                    event['payload'].pop(group, None)
            self._append(node_id, event)
            payload = event['payload']
            with self._lock:
                self._summary['events_missing'] += event['sequence']-cursor-1
                row = self._rows[node_id]
                row.update(status='online', last_seen=event['time'], error=None)
                if event['kind'] == 'epoch':
                    self.observe_training(node_id, [{**payload, 'time': event['time']}])
                elif event['kind'] == 'training':
                    self._training_summary(node_id, payload)
                else:
                    row['latest'] = deepcopy(payload.get('processes', {}))
                    if payload.get('sampling_s') is not None:
                        self._summary['node_sampling_s'] = (self._summary['node_sampling_s'] or 0.) + payload['sampling_s']
                        self._summary['node_samples_timed'] += 1
                    machine = node.get('machine')
                    if self._sources.get(machine) == node_id:
                        self._machines[machine] = {'id': machine, 'source_node': node_id, 'last_seen': event['time'],
                                                   'status': 'online', 'latest': deepcopy(payload)}
                cursor = event['sequence']
                self._cursors[node_id] = cursor
                self._summary['events_received'] += 1
            try:
                exported = self.tensorboard.write_node_event(self.run_id, node, event)
                if isinstance(exported, dict) and exported.get('available') is False:
                    with self._lock:
                        self._summary['log_failures'] += 1
                        self._summary['last_error'] = exported.get('reason')
            except Exception as exc:
                with self._lock:
                    self._summary['log_failures'] += 1
                    self._summary['last_error'] = f'TensorBoard: {type(exc).__name__}'
        with self._lock:
            self._rows[node_id]['status'] = 'starting' if reply.get('active') is False else 'online'
            self._rows[node_id]['error'] = reply.get('error')
        return bool(reply.get('more'))

    def _one(self, node, *, stopping=False):
        node_id = node['id']
        if node_id not in self._eligible or (self._stop.is_set() and not stopping):
            return
        try:
            if node_id not in self._active:
                if stopping:
                    return
                self._rpc.call(node_id, 'telemetry_start', {
                    'task_id': self.run_id, 'machine': node.get('machine') or 'unknown',
                    'collect_machine': self._sources.get(node.get('machine')) == node_id,
                    'interval_seconds': self.interval_seconds}, retries=0, timeout=2)
                self._active.add(node_id)
            if stopping:
                self._rpc.call(node_id, 'telemetry_stop', {'task_id': self.run_id}, retries=0, timeout=2)
            # A finite page budget bounds a reconnect while preserving the cursor.
            for _ in range(8 if stopping else 2):
                with self._lock:
                    cursor = self._cursors[node_id]
                reply = self._rpc.call(node_id, 'telemetry_read', {
                    'task_id': self.run_id, 'after': cursor, 'limit': 128}, retries=0, timeout=2)
                more = self._accept(node, reply)
                if reply.get('active') is False:
                    # A restarted node retains its durable journal, but loses
                    # its sampler. Start it again next cycle after draining.
                    self._active.discard(node_id)
                if not more:
                    break
        except Exception as exc:
            self._failure(node_id, exc)

    def collect(self, *, stopping=False):
        if not self._cycle_lock.acquire(blocking=False):
            return
        try:
            if self._pool is not None:
                list(self._pool.map(lambda node: self._one(node, stopping=stopping), self.nodes))
            with self._lock:
                self._summary['bytes_sent'] = getattr(self._rpc, 'bytes_sent', 0)
        finally:
            self._cycle_lock.release()

    def observe_local(self, sample):
        """Reuse the controller's one physical-host probe, without another sensor."""
        with self._lock:
            duration = number(sample.get('sample_duration_ms'))
            key = sample.get('sample_index')
            if key is not None and key not in self._local_seen and duration is not None:
                self._local_seen.add(key)
                self._summary['local_machine_sampling_s'] = (self._summary['local_machine_sampling_s'] or 0.) + duration / 1000
                self._summary['local_machine_samples_timed'] += 1
            self._machines[self.local_machine] = {
                'id': self.local_machine, 'source_node': 'coordinator',
                'last_seen': sample.get('time'), 'status': 'online', 'latest': deepcopy(sample)}
        self.tensorboard.write_machine_sample(self.run_id, self.local_machine, sample)

    def start(self):
        def loop():
            try:
                endpoints = {node['id']: {'url': node['host']} for node in self.nodes}
                self._rpc = self._rpc_factory(self.runtime, endpoints)
                self._pool = ThreadPoolExecutor(max_workers=min(8, max(1, len(self.nodes))),
                                                thread_name_prefix='dgflow-telemetry-rpc')
                while not self._stop.is_set():
                    self.collect()
                    if self._stop.wait(self.interval_seconds):
                        break
                self.collect(stopping=True)
            except Exception as exc:
                self._failure(None, exc)
            finally:
                if self._pool is not None:
                    self._pool.shutdown(wait=True, cancel_futures=True)
                if self._rpc is not None:
                    self._rpc.close()
        self._thread = threading.Thread(target=loop, daemon=True, name='dgflow-node-monitor')
        self._thread.start()

    def snapshot(self):
        with self._lock:
            return {'run_id': self.run_id, 'interval_seconds': self.interval_seconds,
                    'nodes': deepcopy(list(self._rows.values())), 'machines': deepcopy(list(self._machines.values())),
                    'summary': deepcopy(self._summary), 'legacy': False}

    def finish(self):
        self._stop.set()
        if self._thread is not None:
            # Normal shutdown includes one final read. A slow/unavailable LAN
            # must not hold up an experiment indefinitely.
            self._thread.join(timeout=6)
        with self._lock:
            self._summary['cleanup_complete'] = not (self._thread and self._thread.is_alive())
        return self.snapshot()
