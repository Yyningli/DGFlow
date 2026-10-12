"""Infrastructure failures release admission without claiming experiment success."""
import json
import threading

import numpy as np
import pytest

from dgfl.experiments import runner


class PendingThread:
    def __init__(self, **kwargs):
        pass

    def start(self):
        pass


class StubMonitor:
    def __init__(self, *args):
        self.finished = False

    def start(self):
        pass

    def finish(self):
        self.finished = True
        return {'scope': 'test-double'}


class FakeRPC:
    bytes_sent = 17

    def __init__(self, *args):
        self.closed = False

    def call(self, node, action, payload=None, **kwargs):
        if action == 'health':
            return {'node_id': node, 'capabilities': {'compute': {'gpu': {'available': True}}}}
        if action == 'prepare_compute':
            return {}
        if action == 'prepare_data':
            return {'samples': 1, 'partition_hash': 'test-double'}
        if action == 'train':
            return {'plain_model': payload['reference'], 'training_s': 0,
                    'encrypt_s': 0, 'proof_s': 0}
        raise AssertionError(f'unexpected RPC action: {action}')

    def close(self):
        self.closed = True


def test_tensorboard_failure_does_not_fail_training(experiment, monkeypatch):
    manager, record, _ = experiment

    class BrokenLogs:
        def sync(self, value):
            raise OSError('event storage unavailable')

        def write_sample(self, run_id, row):
            raise OSError('event storage unavailable')

        def close(self, run_id):
            raise OSError('event close unavailable')

    manager.tensorboard = BrokenLogs()
    manager.records[record['run_id']] = record
    manager._run(record)
    assert record['status'] == 'completed'
    assert record['summary']['completed_rounds'] == 1
    assert record['evidence']['tensorboard']['available'] is False
    assert record['run_id'] not in manager._monitors


def test_tensorboard_receives_committed_metrics_and_sampling_callback(experiment, monkeypatch):
    manager, record, _ = experiment
    exported, sampled, closed = [], [], []

    class RecordingLogs:
        def sync(self, value):
            exported.append(json.loads(json.dumps(value)))
            return {'available': True, 'url': '/tensorboard/'}

        def write_sample(self, run_id, row):
            sampled.append((run_id, row))

        def close(self, run_id):
            closed.append(run_id)

    class SamplingMonitor(StubMonitor):
        def start(self):
            self.on_sample({'sample_index': 1})

    monkeypatch.setattr(runner, 'LocalProcessMonitor', SamplingMonitor)
    manager.tensorboard = RecordingLogs()
    manager._run(record)
    assert record['status'] == 'completed'
    assert any(value.get('initial_metrics') and not value['rounds'] for value in exported)
    assert any(len(value['rounds']) == 1 for value in exported)
    assert all(any(event['stage'] == 'completed_round' and event['round'] == 1
                   for event in value['events']) for value in exported if value['rounds'])
    assert sampled == [(record['run_id'], {'sample_index': 1})]
    assert closed == [record['run_id']]


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    nodes = [*[f'client{i}' for i in range(1, 7)],
             *runner.AUTHORITIES, 'aggregator1', 'aggregator2', 'aggregator3']
    cluster = {'deployment': 'single_host', 'nodes': {
        node: {'url': f'https://127.0.0.1:{9000 + index}'} for index, node in enumerate(nodes)}}
    (runtime / 'cluster.json').write_text(json.dumps(cluster), encoding='utf8')
    monkeypatch.setattr(runner, 'implementation_evidence', lambda: {})
    monkeypatch.setattr(runner, 'LocalProcessMonitor', StubMonitor)
    monkeypatch.setattr(runner, 'load_mnist', lambda *a, **k: (None, None, None, None))
    monkeypatch.setattr(runner, 'initial_model', lambda seed, features=64: np.zeros(features * 10 + 10))
    monkeypatch.setattr(runner, 'evaluate', lambda *a, **k: {'accuracy': .5})
    rpcs = []

    def rpc_factory(*args):
        rpc = FakeRPC(*args)
        rpcs.append(rpc)
        return rpc

    monkeypatch.setattr(runner, 'RPCClient', rpc_factory)
    record = {'run_id': 'lifecycle', 'status': 'queued', 'current_round': 0,
              'rounds': [], 'events': [], 'summary': {}, 'error': None, 'evidence': {},
              'config': {'mode': 'plain', 'rounds': 1, 'seed': 42, 'attack': 'none',
                         'malicious_clients': 0, 'non_iid': False, 'offline_aggregators': 0,
                         'train_limit': 6, 'test_limit': 1, 'local_epochs': 1,
                         'backend': 'numpy', 'grid': 8}}
    manager = runner.RunManager(runtime)
    return manager, record, rpcs


def test_initial_save_failure_never_reserves_admission(experiment, monkeypatch):
    manager, record, _ = experiment
    save = manager._save

    def broken_save(*args):
        raise OSError('storage unavailable')

    monkeypatch.setattr(manager, '_save', broken_save)
    with pytest.raises(ValueError, match='初始记录无法保存.*storage unavailable'):
        manager.start(record['config'])
    assert manager.active is None
    assert manager.records == {}
    monkeypatch.setattr(manager, '_save', save)
    monkeypatch.setattr(runner.threading, 'Thread', PendingThread)
    assert manager.start(record['config'])['status'] == 'queued'


@pytest.mark.parametrize('rollback_save_fails', [False, True])
def test_thread_start_failure_is_terminal_and_releases_admission(experiment, monkeypatch, rollback_save_fails):
    manager, record, _ = experiment
    save = manager._save
    calls = 0

    class BrokenThread(PendingThread):
        def start(self):
            raise RuntimeError('thread unavailable')

    def save_once(value):
        nonlocal calls
        calls += 1
        if rollback_save_fails and calls > 1:
            raise OSError('rollback storage unavailable')
        save(value)

    monkeypatch.setattr(manager, '_save', save_once)
    monkeypatch.setattr(runner.threading, 'Thread', BrokenThread)
    with pytest.raises(ValueError, match='线程无法启动.*thread unavailable'):
        manager.start(record['config'])
    assert manager.active is None
    final = next(iter(manager.records.values()))
    assert final['status'] == 'failed'
    assert final['summary']['completed_rounds'] == 0
    if rollback_save_fails:
        assert final['evidence']['persistence']['saved'] is False
        assert '失败记录无法保存' in final['error']
    else:
        persisted = json.loads((manager.runtime / 'results' / final['run_id'] / 'result.json').read_text('utf8'))
        assert persisted['status'] == 'failed'
    monkeypatch.setattr(manager, '_save', save)
    monkeypatch.setattr(runner.threading, 'Thread', PendingThread)
    assert manager.start(record['config'])['status'] == 'queued'


@pytest.mark.parametrize('stage', ['construct', 'start', 'finish'])
def test_monitor_lifecycle_failure_is_diagnostic_and_preserves_training(experiment, monkeypatch, stage):
    manager, record, rpcs = experiment
    monitors = []

    class BrokenMonitor(StubMonitor):
        def __init__(self, *args):
            if stage == 'construct':
                raise ValueError('monitor construct unavailable')
            super().__init__(*args)
            monitors.append(self)

        def start(self):
            if stage == 'start':
                raise ValueError('monitor start unavailable')

        def finish(self):
            super().finish()
            if stage == 'finish':
                raise OSError('monitor finish unavailable')
            return {'scope': 'test-double'}

    monkeypatch.setattr(runner, 'LocalProcessMonitor', BrokenMonitor)
    manager.records[record['run_id']] = record
    manager.active = record['run_id']
    manager._run(record)
    final = manager.snapshot(record['run_id'])
    assert manager.active is None
    assert final['status'] == 'completed'
    assert final['error'] is None
    assert final['summary']['completed_rounds'] == 1
    assert f'monitor {stage} unavailable' in final['evidence']['monitoring']['error']
    assert final['summary']['elapsed_s'] > 0
    assert final['events'][-1]['stage'] == 'finished'
    assert all(rpc.closed for rpc in rpcs)
    assert all(monitor.finished for monitor in monitors)
    if stage != 'finish':
        assert final['evidence']['monitoring']['stage'] == stage
    else:
        assert final['evidence']['memory']['available'] is False
    persisted = json.loads((manager.runtime / 'results' / record['run_id'] / 'result.json').read_text('utf8'))
    assert persisted['status'] == 'completed'


@pytest.mark.parametrize('stage', ['executor_close', 'gpu_profile', 'rpc_close'])
def test_cleanup_failure_does_not_skip_other_cleanup_or_claim_success(experiment, monkeypatch, stage):
    manager, record, rpcs = experiment
    executor_type = runner.RunExecutor
    monitors = []

    class Monitor(StubMonitor):
        def __init__(self, *args):
            super().__init__(*args)
            monitors.append(self)

    class Executor(executor_type):
        def close(self):
            super().close()
            if stage == 'executor_close':
                raise OSError('executor close unavailable')

    monkeypatch.setattr(runner, 'LocalProcessMonitor', Monitor)
    monkeypatch.setattr(runner, 'RunExecutor', Executor)
    if stage == 'gpu_profile':
        from dgfl.crypto import gpu

        class Device:
            lock = threading.Lock()

            def stats_snapshot(self):
                return {}

            def stats_delta(self, *args):
                raise OSError('gpu profile unavailable')

        monkeypatch.setattr(gpu, 'require_gpu', lambda: {'accelerated_operations': []})
        monkeypatch.setattr(gpu, 'runtime', Device)
        record['config']['compute_device'] = 'gpu'
        record['evidence']['compute'] = {}
    if stage == 'rpc_close':
        def broken_close(self):
            self.closed = True
            raise OSError('rpc close unavailable')

        monkeypatch.setattr(FakeRPC, 'close', broken_close)
    manager.records[record['run_id']] = record
    manager.active = record['run_id']
    manager._run(record)
    final = manager.snapshot(record['run_id'])
    assert manager.active is None
    assert final['status'] == 'failed'
    assert final['summary']['completed_rounds'] == 1
    assert final['evidence']['cleanup_errors'][0]['stage'] == stage
    assert all(rpc.closed for rpc in rpcs)
    assert monitors[0].finished


def test_final_save_failure_stays_failed_in_memory_and_releases_admission(experiment, monkeypatch):
    manager, record, rpcs = experiment
    save = manager._save

    def terminal_save_failure(value):
        if value['status'] == 'completed':
            raise OSError('final storage unavailable')
        save(value)

    monkeypatch.setattr(manager, '_save', terminal_save_failure)
    manager.records[record['run_id']] = record
    manager.active = record['run_id']
    manager._run(record)
    final = manager.snapshot(record['run_id'])
    assert final['status'] == 'failed'
    assert '最终记录无法保存' in final['error']
    assert 'final storage unavailable' in final['error']
    assert final['summary']['completed_rounds'] == 1
    assert final['evidence']['persistence']['saved'] is False
    assert final['events'][-1]['message'] == final['error']
    assert manager.active is None
    assert all(rpc.closed for rpc in rpcs)
    monkeypatch.setattr(runner.threading, 'Thread', PendingThread)
    assert manager.start(record['config'])['status'] == 'queued'


@pytest.mark.parametrize('stage', ['snapshot', 'final_sample'])
def test_node_monitor_failure_preserves_training_and_always_stops_sampler(experiment, monkeypatch, stage):
    manager, record, _ = experiment
    node_monitors = []
    call = FakeRPC.call

    def telemetry_health(self, node, action, payload=None, **kwargs):
        reply = call(self, node, action, payload, **kwargs)
        if action == 'health':
            reply['capabilities']['node_telemetry'] = True
        return reply

    class NodeMonitor:
        def __init__(self, *args, **kwargs):
            self.finished = False
            node_monitors.append(self)

        def start(self):
            pass

        def snapshot(self):
            if stage == 'snapshot':
                raise OSError('node snapshot unavailable')
            return {'nodes': [], 'machines': [], 'summary': {}}

        def observe_training(self, *args):
            pass

        def observe_local(self, sample):
            raise OSError('final sample log unavailable')

        def finish(self):
            self.finished = True
            return self.snapshot()

    class Monitor(StubMonitor):
        def finish(self):
            self.finished = True
            return {'hardware': {'samples': [{'sample_index': 1}]}}

    monkeypatch.setattr(FakeRPC, 'call', telemetry_health)
    monkeypatch.setattr(runner, 'RunNodeMonitor', NodeMonitor)
    monkeypatch.setattr(runner, 'LocalProcessMonitor', Monitor)
    manager._run(record)
    assert record['status'] == 'completed'
    assert record['error'] is None
    assert record['summary']['completed_rounds'] == 1
    assert node_monitors[0].finished
    assert 'unavailable' in record['evidence']['node_monitoring_error']
    persisted = json.loads((manager.runtime / 'results' / record['run_id'] / 'result.json').read_text('utf8'))
    assert persisted['status'] == 'completed'
    assert manager.active is None


def test_monitor_snapshot_failure_does_not_hide_experiment_or_saved_evidence(experiment):
    manager, record, _ = experiment
    manager.records[record['run_id']] = record
    previous = {'nodes': [], 'machines': [], 'summary': {'events_received': 4}}
    record['evidence']['node_monitoring'] = previous

    class BrokenMonitor:
        def snapshot(self):
            raise OSError('node snapshot unavailable')

    manager._node_monitors[record['run_id']] = BrokenMonitor()
    detail = manager.snapshot(record['run_id'])
    assert detail['evidence']['node_monitoring'] == previous
    assert 'node snapshot unavailable' in detail['evidence']['node_monitoring_error']
    assert manager.monitor_snapshot(record['run_id'])['summary']['events_received'] == 4
    detail['evidence']['node_monitoring']['summary']['events_received'] = 0
    assert record['evidence']['node_monitoring'] == previous
