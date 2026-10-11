"""The monitoring plane must remain independent of training and deployment."""
import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from dgfl.experiments.monitoring import (
    RunNodeMonitor,
    clean_event,
    frozen_nodes,
    iter_monitor_events,
    monitor_snapshot,
)
from dgfl.experiments.tensorboard import TensorBoardLogs
from dgfl.services.control import create_control_app

TIME = '2026-10-10T01:00:00+00:00'


def nodes():
    return [{'id': name, 'role': role, 'machine': machine, 'host': f'https://{machine}:1234', 'authority': None}
            for name, role, machine in [('client1', 'client', 'local'), ('authority1', 'authority', 'B'),
                                        ('aggregator1', 'aggregator', 'B')]]


def resource(sequence=1, sample_index=0, machine=False):
    payload = {'sample_index': sample_index, 'elapsed_s': sample_index * 5, 'round_id': 0,
               'stage': 'prepare', 'processes': {'cpu_percent': None, 'rss_bytes': 1024}}
    if machine:
        payload.update(cpu={'system_utilization_percent': 35}, gpu={'utilization_percent': 0},
                       system={'memory_used_bytes': 4096})
    return {'sequence': sequence, 'kind': 'resource', 'time': TIME, 'payload': payload}


def epoch(sequence=2, epoch_id=1, round_id=1):
    return {'sequence': sequence, 'kind': 'epoch', 'time': TIME, 'payload': {
        'epoch': epoch_id, 'round_id': round_id, 'step': (round_id-1)*2+epoch_id,
        'accuracy': .75, 'loss': .5, 'samples': 10, 'elapsed_s': .02, 'metrics_eval_s': .003}}


class FakeRPC:
    def __init__(self, *args):
        self.bytes_sent = 0
        self.calls = []
        self.sources = {}
        self.events = {}
        self.closed = False
        self.read_done = threading.Event()
        self.fail = set()

    def call(self, node, action, payload, **kwargs):
        self.calls.append((node, action, copy.deepcopy(payload), kwargs))
        self.bytes_sent += 10
        if node in self.fail:
            raise TimeoutError('node unavailable')
        if action == 'telemetry_start':
            self.sources[node] = payload['collect_machine']
            self.events.setdefault(node, [resource(machine=payload['collect_machine'])])
            return {'active': True}
        if action == 'telemetry_stop':
            return {'active': False}
        available = [event for event in self.events[node] if event['sequence'] > payload['after']]
        events = available[:payload['limit']]
        self.read_done.set()
        return {'node_id': node, 'task_id': payload['task_id'], 'events': copy.deepcopy(events),
                'next_cursor': events[-1]['sequence'] if events else payload['after'],
                'more': len(available) > len(events), 'active': True}

    def close(self):
        self.closed = True


@pytest.fixture
def collector(tmp_path):
    logs = TensorBoardLogs(tmp_path)
    monitor = RunNodeMonitor(tmp_path, 'run-a', nodes(), logs, local_epochs=2)
    monitor._rpc = FakeRPC()
    monitor._pool = ThreadPoolExecutor(max_workers=3)
    yield monitor
    monitor._pool.shutdown(wait=True)
    logs.close_all()


def scalars(path):
    reader = EventAccumulator(str(path), purge_orphaned_data=False).Reload()
    return {tag: reader.Scalars(tag) for tag in reader.Tags()['scalars']}


def test_local_and_lan_sources_are_unique_and_protocol_counter_is_independent(collector):
    business_counter = 12345
    collector.collect()
    snapshot = collector.snapshot()
    assert collector._rpc.sources == {'client1': False, 'authority1': True, 'aggregator1': False}
    assert [row['id'] for row in snapshot['machines']] == ['B']
    assert all(row['latest']['rss_bytes'] == 1024 for row in snapshot['nodes'])
    assert all(row['latest']['cpu_percent'] is None for row in snapshot['nodes'])
    assert snapshot['summary']['bytes_sent'] == 60
    assert business_counter == 12345
    assert all(call[3] == {'retries': 0, 'timeout': 2} for call in collector._rpc.calls)
    snapshot['nodes'].clear()
    assert len(collector.snapshot()['nodes']) == 3


def test_incremental_retries_and_recovery_never_duplicate_events(collector):
    collector._rpc.fail.add('client1')
    collector.collect()
    assert collector.snapshot()['nodes'][0]['status'] == 'unavailable'
    collector._rpc.fail.clear()
    collector.collect()
    collector._rpc.events['client1'].append(epoch())
    collector.collect()
    collector.collect()
    persisted = list(iter_monitor_events(collector.runtime, collector.run_id))
    assert len(persisted) == 4
    assert collector.snapshot()['nodes'][0]['training']['step'] == 1
    assert collector.snapshot()['summary']['metrics_eval_s'] == .003
    collector.observe_training('client1', [{**epoch()['payload'], 'time': TIME}])
    assert collector.snapshot()['summary']['metrics_eval_s'] == .003
    reader = scalars(collector.tensorboard.logdir / 'run-a' / 'nodes' / 'client1')
    assert reader['training/accuracy'][0].value == .75
    assert 'processes/cpu_percent' not in reader
    assert 'system/gpu_utilization_percent' not in reader


def test_unavailable_machine_source_preserves_last_reading_but_marks_it_unavailable(collector):
    collector.collect()
    before = collector.snapshot()['machines'][0]
    collector._rpc.fail.add('authority1')
    collector.collect()
    stale = collector.snapshot()['machines'][0]
    assert stale['status'] == 'unavailable'
    assert stale['last_seen'] == before['last_seen'] and stale['latest'] == before['latest']


def test_live_node_and_machine_events_are_deduplicated_after_restart(collector):
    collector.collect()
    collector._rpc.events['client1'].append(epoch())
    collector.collect()
    record = {'run_id': 'run-a', 'created_at': TIME, 'evidence': {'node_snapshot': nodes()},
              'rounds': [{'round': 1, 'accuracy': .8, 'loss': .4,
                          'client_training_metrics': {'client1': [{**epoch()['payload'], 'time': TIME}]}}]}
    restarted = TensorBoardLogs(collector.runtime)
    for _ in range(2):
        assert restarted.sync(record)['available']
    path = restarted.logdir / 'run-a'
    assert len(scalars(path / 'nodes' / 'client1')['training/accuracy']) == 1
    assert len(scalars(path / 'machines' / 'B')['system/cpu_percent']) == 1
    assert scalars(path)['evaluation/accuracy'][0].step == 1
    assert restarted._replay_offsets['run-a']


def test_protocol_result_fallback_populates_node_table_when_sampling_fails(collector):
    collector.observe_training('client1', [{**epoch()['payload'], 'time': TIME}])
    row = collector.snapshot()['nodes'][0]
    assert row['training']['accuracy'] == .75
    assert row['latest'] == {}


def test_local_host_uses_existing_sample_and_never_calls_another_driver(collector):
    sample = {**resource(machine=True)['payload'], 'time': TIME}
    collector.observe_local(sample)
    host = collector.snapshot()['machines'][0]
    assert host['id'] == 'local' and host['source_node'] == 'coordinator'
    assert 'processes/rss_bytes' not in scalars(collector.tensorboard.logdir / 'run-a' / 'machines' / 'local')


def test_only_machine_representative_can_publish_host_metrics_and_sampling_is_counted_once(collector):
    injected = resource(machine=True)
    injected['payload'].update(sampling_s=.025, machine_sampling_s=.02)
    collector._rpc.events['client1'] = [injected]
    collector.collect()
    collector.collect()
    sample = {**resource(machine=True)['payload'], 'sample_duration_ms': 40, 'time': TIME}
    collector.observe_local(sample)
    collector.observe_local(sample)
    summary = collector.snapshot()['summary']
    assert summary['node_sampling_s'] == .025 and summary['node_samples_timed'] == 1
    assert summary['local_machine_sampling_s'] == .04 and summary['local_machine_samples_timed'] == 1
    raw = next(row['event']['payload'] for row in iter_monitor_events(collector.runtime, collector.run_id)
               if row['node_id'] == 'client1')
    assert all(field not in raw for field in ('cpu', 'gpu', 'system', 'machine_sampling_s'))


def test_training_summary_is_allowlisted_deduplicated_and_separate_from_epoch(collector):
    collector.collect()
    payload = {'round_id': 1, 'epochs': 2, 'metrics_eval_s': .03, 'metrics_callback_s': .02,
               'training_s': .1, 'private_weights': [123]}
    collector._rpc.events['client1'].extend([epoch(), {'sequence': 3, 'kind': 'training',
                                                      'time': TIME, 'payload': payload}])
    collector.collect()
    collector.collect()
    summary = collector.snapshot()['summary']
    assert summary['training_rounds']['1'] == {'clients_recorded': 1, 'metrics_eval_s': .03,
                                              'metrics_callback_s': .02, 'training_s': .1}
    assert summary['metrics_eval_s'] == .003
    assert summary['log_failures'] == 0
    assert 'private_weights' not in json.dumps(list(iter_monitor_events(collector.runtime, collector.run_id)))


def test_simulated_three_machine_collection_has_one_host_source_per_machine(tmp_path):
    rows = [{'id': node, 'role': role, 'machine': machine, 'host': f'https://{machine}:1234'}
            for node, role, machine in [('authority1', 'authority', 'A'), ('authority2', 'authority', 'B'),
                                        ('aggregator1', 'aggregator', 'B'), ('client1', 'client', 'C'),
                                        ('client2', 'client', 'C')]]
    monitor = RunNodeMonitor(tmp_path, 'three-machines', rows, TensorBoardLogs(tmp_path), local_machine='A')
    monitor._rpc = FakeRPC()
    monitor._pool = ThreadPoolExecutor(max_workers=5)
    try:
        monitor.collect()
        monitor.observe_local({**resource(machine=True)['payload'], 'time': TIME})
        assert monitor._rpc.sources == {'authority1': False, 'authority2': True, 'aggregator1': False,
                                        'client1': True, 'client2': False}
        assert {row['id'] for row in monitor.snapshot()['machines']} == {'A', 'B', 'C'}
    finally:
        monitor._pool.shutdown(wait=True)
        monitor.tensorboard.close_all()


@pytest.mark.parametrize('bad', [
    {'sequence': True, 'kind': 'resource', 'time': TIME, 'payload': {}},
    {'sequence': 1, 'kind': 'epoch', 'time': TIME, 'payload': {'epoch': 1, 'round_id': 1, 'step': 4}},
    {'sequence': 1, 'kind': 'epoch', 'time': 'no-date', 'payload': {}},
    {'sequence': 1, 'kind': 'private-model', 'time': TIME, 'payload': {'weights': [1, 2]}},
])
def test_invalid_peer_diagnostics_are_rejected_without_advancing_cursor(collector, bad):
    collector._rpc.events['client1'] = [bad]
    collector.collect()
    assert collector._cursors['client1'] == 0
    assert collector.snapshot()['nodes'][0]['status'] == 'unavailable'


def test_numeric_allowlist_drops_nonfinite_and_private_fields():
    raw = resource()
    raw['payload']['processes'].update(cpu_percent=float('nan'), rss_bytes=10**400, secret='not a metric')
    raw['payload']['raw_model'] = [1, 2, 3]
    clean = clean_event(raw, 2)
    assert clean['payload']['processes'] == {'cpu_percent': None, 'rss_bytes': None}
    assert 'raw_model' not in clean['payload']


def test_thread_stop_drains_and_closes_its_own_transport(tmp_path):
    rpc = FakeRPC()
    collector = RunNodeMonitor(tmp_path, 'threaded', nodes(), TensorBoardLogs(tmp_path),
                               rpc_factory=lambda *args: rpc)
    collector.start()
    assert rpc.read_done.wait(5)
    result = collector.finish()
    assert result['summary']['cleanup_complete'] is True
    assert rpc.closed
    started = {node for node, action, _, _ in rpc.calls if action == 'telemetry_start'}
    stopped = {node for node, action, _, _ in rpc.calls if action == 'telemetry_stop'}
    assert started and stopped == started


def test_restarted_node_resumes_its_sampler_without_losing_cursor(collector):
    collector.collect()
    original = collector._rpc.call
    restarted = {'client1'}
    def call(node, action, payload, **kwargs):
        result = original(node, action, payload, **kwargs)
        if action == 'telemetry_read' and node in restarted:
            result['active'] = False
            restarted.remove(node)
        return result
    collector._rpc.call = call
    collector.collect()
    assert 'client1' not in collector._active
    assert collector.snapshot()['nodes'][0]['status'] == 'starting'
    collector._rpc.events['client1'].append(resource(sequence=2, sample_index=1))
    collector.collect()
    assert 'client1' in collector._active
    assert collector._cursors['client1'] == 2
    starts = [node for node, action, _, _ in collector._rpc.calls if action == 'telemetry_start']
    assert starts.count('client1') == 2
    assert len(list(iter_monitor_events(collector.runtime, collector.run_id))) == 4


def test_history_snapshot_never_uses_current_deployment():
    rows = frozen_nodes({'deployment': 'single_host', 'nodes': {
        'client1': {'url': 'https://old:1', 'machine': 'local'},
        'client2': {'url': 'https://old:2', 'machine': 'local'}},
        'client_authorities': {'client1': 'authority1'}}, {'client_count': 1})
    assert len(rows) == 1 and rows[0]['authority'] == 'authority1'
    record = {'run_id': 'old', 'status': 'completed', 'evidence': {'node_snapshot': rows}}
    snapshot = monitor_snapshot(record)
    assert snapshot['nodes'][0]['host'] == 'https://old:1'
    assert snapshot['nodes'][0]['status'] == 'missing'
    old = monitor_snapshot({'run_id': 'legacy', 'status': 'completed',
                            'config': {'client_count': 2, 'authority_count': None, 'aggregator_count': None}})
    assert old['legacy'] is True and len(old['nodes']) == 2
    assert all(row['host'] is None and row['latest'] == {} for row in old['nodes'])


def test_control_monitor_endpoint_reads_detached_cache_without_role_queries(tmp_path):
    app = create_control_app(tmp_path, auto_prepare_compute=False)
    manager = app.state.manager
    manager.records['old'] = {'run_id': 'old', 'status': 'completed', 'config': {'client_count': 2},
                              'evidence': {'node_snapshot': nodes(),
                                           'resources': {'hardware': {'samples': [{'sample_index': 1}]}}}}
    with TestClient(app) as client:
        response = client.get('/api/runs/old/monitor')
        assert response.status_code == 200
        body = response.json()
        assert body['run_id'] == 'old' and len(body['nodes']) == 3
        assert body['summary']['bytes_sent'] == 0
        assert client.get('/api/runs/missing/monitor').status_code == 404
        detail = client.get('/api/runs/old').json()
        assert 'samples' not in detail['evidence']['resources']['hardware']
        exported = client.get('/api/runs/old/export').json()
        assert exported['evidence']['resources']['hardware']['samples'] == [{'sample_index': 1}]
    assert not (tmp_path / 'monitoring').exists()


def test_raw_journal_torn_tail_does_not_hide_other_segments(tmp_path):
    directory = tmp_path / 'monitoring' / 'run-a'
    directory.mkdir(parents=True)
    (directory / 'client1.old.jsonl').write_bytes(b'{"incomplete"')
    row = {'node_id': 'client1', 'event': resource()}
    (directory / 'client1.new.jsonl').write_text(json.dumps(row) + '\n', encoding='utf8')
    assert len(list(iter_monitor_events(tmp_path, 'run-a'))) == 1


def test_replay_does_not_advance_past_a_failed_export(collector, monkeypatch):
    collector.collect()
    collector._rpc.events['client1'].append(resource(sequence=2, sample_index=1))
    collector.collect()
    logs = TensorBoardLogs(collector.runtime)
    original = logs._write
    failed = []
    def export(run_id, batches, *, scope=None):
        if scope==('nodes','client1') and not failed:
            failed.append(True)
            return {'available': False, 'reason': 'transient failure'}
        return original(run_id,batches,scope=scope)
    monkeypatch.setattr(logs,'_write',export)
    record = {'run_id':'run-a','evidence':{'node_snapshot':nodes()}}
    assert logs.sync(record)['available'] is False
    assert logs.sync(record)['available'] is True
    assert len(scalars(logs.logdir/'run-a'/'nodes'/'client1')['processes/rss_bytes'])==2


@pytest.mark.parametrize('scope', [('nodes', '../outside'), ('machines', 'CON'), ('invalid', 'local')])
def test_tensorboard_scope_cannot_escape_runtime(tmp_path, scope):
    logs = TensorBoardLogs(tmp_path)
    assert logs._write('safe', [(1, 1., {'metric': 1.})], scope=scope)['available'] is False
