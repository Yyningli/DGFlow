"""Node journals are durable and readable without entering the role lock."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import psutil
import pytest
from fastapi.testclient import TestClient

from dgfl.services import node
from dgfl.services.telemetry import NodeTelemetry
from dgfl.transport.security import Identity, create_cluster


def epoch(epoch=1, **extra):
    return {'epoch': epoch, 'accuracy': .75, 'loss': .42, 'samples': 12,
            'elapsed_s': .3, 'metrics_eval_s': .01, **extra}


def wait_event(telemetry, task='task'):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        rows = telemetry.read({'task_id': task})['events']
        if any(row['kind'] == 'resource' for row in rows):
            return rows
        threading.Event().wait(.01)
    pytest.fail('resource sampler did not produce a reading')


def test_prestart_epochs_are_durable_deduplicated_and_allowlisted(tmp_path):
    telemetry = NodeTelemetry('client1', tmp_path)
    assert telemetry.record_epoch('task', 2, 3, epoch(2, model=[1, 2], x=[[42]], secret='hidden'))
    assert not telemetry.record_epoch('task', 2, 3, epoch(2))
    result = telemetry.read({'task_id': 'task'})
    assert result['active'] is False
    assert result['next_cursor'] == 1
    event = result['events'][0]
    assert event['payload']['step'] == 5
    assert event['payload']['round_id'] == 2
    assert set(event['payload']) == {'epoch', 'round_id', 'step', 'samples', 'accuracy', 'loss', 'elapsed_s', 'metrics_eval_s'}
    assert 'secret' not in json.dumps(result)
    telemetry.close()
    reopened = NodeTelemetry('client1', tmp_path)
    assert reopened.read({'task_id': 'task'})['events'] == result['events']
    assert not reopened.record_epoch('task', 2, 3, epoch(2))
    reopened.close()


def test_cursor_pagination_acknowledges_only_returned_rows(tmp_path):
    telemetry = NodeTelemetry('client1', tmp_path)
    for i in range(1, 5):
        assert telemetry.record_epoch('task', 1, 4, epoch(i))
    first = telemetry.read({'task_id': 'task', 'after': 0, 'limit': 2})
    assert first['next_cursor'] == 2 and first['more'] is True
    second = telemetry.read({'task_id': 'task', 'after': 2, 'limit': 2})
    assert [row['sequence'] for row in second['events']] == [3, 4]
    assert second['next_cursor'] == 4 and second['more'] is False
    assert telemetry.read({'task_id': 'task', 'after': 4})['next_cursor'] == 4
    first['events'][0]['payload']['accuracy'] = 0
    assert telemetry.read({'task_id': 'task'})['events'][0]['payload']['accuracy'] == .75
    for request in ({'after': -1}, {'limit': 257}, {'limit': True}):
        with pytest.raises(ValueError):
            telemetry.read({'task_id': 'task', **request})
    telemetry.close()


def test_long_journal_keeps_bounded_cache_and_old_cursor_can_page_after_restart(tmp_path):
    telemetry = NodeTelemetry('client1', tmp_path)
    for i in range(1, 701):
        assert telemetry.record_epoch('task', 1, 700, epoch(i))
    assert len(telemetry._journals['task'].events) == 512
    assert len(telemetry._journals['task'].offsets) < 10
    first = telemetry.read({'task_id': 'task', 'limit': 128})
    assert [row['sequence'] for row in first['events']] == list(range(1, 129))
    assert first['more'] is True
    telemetry.close()
    restarted = NodeTelemetry('client1', tmp_path)
    second = restarted.read({'task_id': 'task', 'after': 128, 'limit': 128})
    assert [row['sequence'] for row in second['events']] == list(range(129, 257))
    assert second['next_cursor'] == 256 and second['more'] is True
    assert len(restarted._journals['task'].events) == 512
    tail = restarted.read({'task_id': 'task', 'after': 699})
    assert [row['sequence'] for row in tail['events']] == [700]
    assert tail['more'] is False
    restarted.close()


def test_sampler_start_end_and_restart_continue_indices_without_machine_data(tmp_path):
    telemetry = NodeTelemetry('client1', tmp_path)
    telemetry.record_epoch('task', 1, 2, epoch())
    telemetry.start({'task_id': 'task', 'machine': 'A', 'collect_machine': False})
    assert telemetry.start({'task_id': 'task'})['active'] is True
    with pytest.raises(ValueError, match='another'):
        telemetry.start({'task_id': 'other'})
    rows = wait_event(telemetry)
    assert rows[0]['kind'] == 'epoch'
    first = next(row['payload'] for row in rows if row['kind'] == 'resource')
    assert first['sample_index'] == 0
    assert first['processes']['cpu_percent'] is None
    assert first['processes']['rss_bytes'] > 0
    assert first['sampling_s'] >= 0
    assert 'machine_sampling_s' not in first
    assert not {'cpu', 'gpu', 'system'} & first.keys()
    telemetry.set_context('task', 2, 'train')
    telemetry.stop({'task_id': 'task'})
    result = telemetry.read({'task_id': 'task'})
    resources = [row['payload'] for row in result['events'] if row['kind'] == 'resource']
    assert len(resources) == 2 and resources[-1]['round_id'] == 2
    assert resources[-1]['stage'] == 'train' and result['active'] is False
    telemetry.close()
    reopened = NodeTelemetry('client1', tmp_path)
    reopened.start({'task_id': 'task', 'collect_machine': False})
    deadline = time.monotonic() + 3
    while len(reopened.read({'task_id': 'task'})['events']) <= len(result['events']) and time.monotonic() < deadline:
        threading.Event().wait(.01)
    reopened.stop({'task_id': 'task'})
    resources = [row['payload'] for row in reopened.read({'task_id': 'task'})['events'] if row['kind'] == 'resource']
    assert [row['sample_index'] for row in resources] == [0, 1, 2, 3]
    assert resources[2]['elapsed_s'] >= resources[1]['elapsed_s']
    reopened.close()


def test_machine_unavailable_values_stay_null_and_not_zero(tmp_path, monkeypatch):
    from dgfl.services import telemetry as module

    closed = []

    class MissingHardware:
        def sample(self):
            return {'cpu': {'system_utilization_percent': None}, 'gpu': {'utilization_percent': None}}

        def close(self):
            closed.append(True)

    monkeypatch.setattr(module, 'HardwareSampler', MissingHardware)
    monkeypatch.setattr(module.psutil, 'virtual_memory', lambda: (_ for _ in ()).throw(OSError('unavailable')))
    monkeypatch.setattr(module.psutil, 'disk_usage', lambda _: (_ for _ in ()).throw(OSError('unavailable')))
    telemetry = NodeTelemetry('authority1', tmp_path)
    telemetry.start({'task_id': 'task', 'collect_machine': True, 'machine': 'B'})
    payload = wait_event(telemetry)[0]['payload']
    assert payload['cpu']['system_utilization_percent'] is None
    assert payload['gpu']['utilization_percent'] is None
    assert all(value is None for value in payload['system'].values())
    telemetry.close()
    assert closed == [True]


def test_sampling_duration_counts_process_and_host_probes_but_not_journal(tmp_path, monkeypatch):
    from dgfl.services import telemetry as module

    clock = [10.]
    monkeypatch.setattr(module.time, 'perf_counter', lambda: clock[0])
    telemetry = NodeTelemetry('authority1', tmp_path)
    journal = telemetry._journal('task')
    append = journal.append

    def processes(state):
        clock[0] += .25
        return {'rss_bytes': 100, 'cpu_percent': None}

    class Hardware:
        def sample(self):
            clock[0] += 1.
            return {'cpu': {}, 'gpu': {}}

    def system():
        clock[0] += .5
        return {}

    def slow_journal(kind, payload):
        clock[0] += 5.
        return append(kind, payload)

    monkeypatch.setattr(telemetry, '_processes', processes)
    monkeypatch.setattr(telemetry, '_system', system)
    monkeypatch.setattr(journal, 'append', slow_journal)
    state = {'sample_index': 0, 'offset': 0., 'started': module.time.monotonic(),
             'round_id': 1, 'stage': 'train', 'collect_machine': True, 'journal': journal}
    telemetry._sample(state, Hardware())
    payload = telemetry.read({'task_id': 'task'})['events'][0]['payload']
    assert payload['sampling_s'] == 1.75
    assert payload['machine_sampling_s'] == 1.5
    telemetry.close()
    restarted = NodeTelemetry('authority1', tmp_path)
    assert restarted.read({'task_id': 'task'})['events'][0]['payload'] == payload
    restarted.close()


def test_old_journal_duration_is_missing_instead_of_zero(tmp_path):
    telemetry = NodeTelemetry('client1', tmp_path)
    journal = telemetry._journal('task')
    assert journal.append('resource', {'sample_index': 0, 'elapsed_s': 1.,
                                      'round_id': 1, 'stage': 'train', 'processes': {}})
    telemetry.close()
    path = tmp_path / 'nodes' / 'client1' / 'telemetry' / 'task' / 'events.jsonl'
    row = json.loads(path.read_text('utf8'))
    row['payload'].pop('sampling_s')
    path.write_text(json.dumps(row) + '\n', encoding='utf8')
    restarted = NodeTelemetry('client1', tmp_path)
    payload = restarted.read({'task_id': 'task'})['events'][0]['payload']
    assert payload['sampling_s'] is None
    assert 'machine_sampling_s' not in payload
    restarted.close()


def test_disk_failure_is_diagnostic_and_does_not_escape_epoch_callback(tmp_path, monkeypatch):
    from pathlib import Path

    original = Path.open

    def denied(path, *args, **kwargs):
        if path.name == 'events.jsonl':
            raise PermissionError('denied')
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', denied)
    telemetry = NodeTelemetry('client1', tmp_path)
    assert telemetry.record_epoch('task', 1, 2, epoch())
    result = telemetry.read({'task_id': 'task'})
    assert result['events'][0]['payload']['accuracy'] == .75
    assert 'write failed' in result['error']
    assert not telemetry.record_epoch('../unsafe', 1, 2, epoch())
    assert not telemetry.record_epoch('task', 1, 2, epoch(3))
    telemetry.close()


def test_partial_crash_tail_is_skipped_and_next_event_remains_readable(tmp_path):
    telemetry = NodeTelemetry('client1', tmp_path)
    assert telemetry.record_epoch('task', 1, 2, epoch())
    telemetry.close()
    path = tmp_path / 'nodes' / 'client1' / 'telemetry' / 'task' / 'events.jsonl'
    with path.open('ab') as stream:
        stream.write(b'{"sequence":2,"kind":')
    restarted = NodeTelemetry('client1', tmp_path)
    assert restarted.record_epoch('task', 1, 2, epoch(2))
    restarted.close()
    final = NodeTelemetry('client1', tmp_path)
    result = final.read({'task_id': 'task'})
    assert [row['sequence'] for row in result['events']] == [1, 2]
    assert result['next_cursor'] == 2
    assert 'skipped' in result['error']
    final.close()


@pytest.mark.parametrize('payload', [None, [], 1])
def test_old_cursor_skips_nonmapping_payload_in_damaged_journal(tmp_path, payload):
    path = tmp_path / 'nodes' / 'client1' / 'telemetry' / 'task' / 'events.jsonl'
    path.parent.mkdir(parents=True)
    with path.open('w', encoding='utf8') as stream:
        for sequence in range(1, 515):
            row = {'sequence': sequence, 'kind': 'epoch', 'time': '2026-10-11T00:00:00+00:00',
                   'payload': {**epoch(sequence), 'round_id': 1, 'step': sequence}}
            if sequence == 2:
                row['payload'] = payload
            stream.write(json.dumps(row) + '\n')
    telemetry = NodeTelemetry('client1', tmp_path)
    try:
        result = telemetry.read({'task_id': 'task', 'limit': 4})
        assert [row['sequence'] for row in result['events']] == [1, 3, 4, 5]
        assert result['next_cursor'] == 5 and result['more'] is True
        assert 'error' in result
    finally:
        telemetry.close()


@pytest.mark.parametrize('timestamp', ['corrupt', '2026-10-11T00:00:00', 'x' * 65])
@pytest.mark.parametrize('event_count', [3, 515])
def test_damaged_timestamp_cannot_block_later_telemetry_after_restart(tmp_path, timestamp, event_count):
    path = tmp_path / 'nodes' / 'client1' / 'telemetry' / 'task' / 'events.jsonl'
    path.parent.mkdir(parents=True)
    with path.open('w', encoding='utf8') as stream:
        for sequence in range(1, event_count + 1):
            row = {'sequence': sequence, 'kind': 'epoch',
                   'time': timestamp if sequence == 2 else '2026-10-11T00:00:00+00:00',
                   'payload': {**epoch(sequence), 'round_id': 1, 'step': sequence}}
            stream.write(json.dumps(row) + '\n')
    telemetry = NodeTelemetry('client1', tmp_path)
    try:
        result = telemetry.read({'task_id': 'task', 'limit': 2})
        assert [row['sequence'] for row in result['events']] == [1, 3]
        assert result['next_cursor'] == 3
        assert 'error' in result
    finally:
        telemetry.close()


def test_links_and_junction_attributes_are_rejected(tmp_path, monkeypatch):
    import stat
    from pathlib import Path

    target = tmp_path / 'nodes' / 'client1' / 'telemetry'
    target.mkdir(parents=True)
    original = Path.lstat

    def junction(path):
        value = original(path)
        if path == target:
            return SimpleNamespace(st_mode=value.st_mode, st_file_attributes=getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400))
        return value

    monkeypatch.setattr(Path, 'lstat', junction)
    telemetry = NodeTelemetry('client1', tmp_path)
    assert telemetry.record_epoch('task', 1, 1, epoch())
    assert 'write failed' in telemetry.read({'task_id': 'task'})['error']
    assert not (target / 'task' / 'events.jsonl').exists()
    telemetry.close()


def test_resource_cpu_is_logical_core_normalized_and_unknown_children_excluded(tmp_path, monkeypatch):
    from dgfl.services import telemetry as module

    cpu_time = [2.]

    class OwnProcess:
        pid = 100

        def create_time(self):
            return 1.

        def children(self, recursive):
            return [UnrelatedProcess()]

        def cpu_times(self):
            return SimpleNamespace(user=cpu_time[0], system=0.)

        def memory_info(self):
            return SimpleNamespace(rss=100)

    class UnrelatedProcess(OwnProcess):
        pid = 101

        def parents(self):
            return []

    monkeypatch.setattr(module.psutil, 'Process', lambda _=None: OwnProcess())
    monkeypatch.setattr(module.psutil, 'cpu_count', lambda logical: 4)
    times = iter((10., 11.))
    monkeypatch.setattr(module.time, 'monotonic', lambda: next(times))
    telemetry = NodeTelemetry('client1', tmp_path)
    state = {'previous': {}, 'previous_at': None}
    assert telemetry._processes(state) == {'cpu_percent': None, 'rss_bytes': 100}
    cpu_time[0] = 4.
    assert telemetry._processes(state) == {'cpu_percent': 50., 'rss_bytes': 100}


def test_resource_access_denied_remains_null(tmp_path, monkeypatch):
    from dgfl.services import telemetry as module

    monkeypatch.setattr(module.psutil, 'Process', lambda _: (_ for _ in ()).throw(psutil.AccessDenied()))
    telemetry = NodeTelemetry('client1', tmp_path)
    assert telemetry._processes({'previous': {}, 'previous_at': None}) == {'cpu_percent': None, 'rss_bytes': None}
    telemetry.close()


def test_rpc_requires_coordinator_and_telemetry_does_not_enter_business_lock(tmp_path, monkeypatch):
    from dgfl.services import roles

    create_cluster(tmp_path / 'keys', dict.fromkeys(('coordinator', 'client1', 'authority1'), '127.0.0.1'))
    coordinator = Identity(tmp_path / 'keys', 'coordinator')
    other = Identity(tmp_path / 'keys', 'authority1')
    entered, release = threading.Event(), threading.Event()

    class BlockedWorker:
        def __init__(self, *args, **kwargs):
            pass

        def execute(self, action, payload):
            entered.set()
            assert release.wait(5)
            return {'done': True}

        def close(self):
            pass

    monkeypatch.setattr(roles, 'RoleWorker', BlockedWorker)
    app = node.create_node_app('client1', tmp_path)

    def envelope(identity, action, payload, context):
        return identity.seal('client1', 'rpc', {'action': action, 'payload': payload}, context)

    with TestClient(app) as client, ThreadPoolExecutor(max_workers=2) as pool:
        denied = client.post('/rpc', content=envelope(other, 'telemetry_read', {'task_id': 'task'}, 'denied'))
        assert denied.status_code == 403
        future = pool.submit(client.post, '/rpc', content=envelope(coordinator, 'train', {}, 'business'))
        assert entered.wait(3)
        try:
            for action, payload in (('telemetry_start', {'task_id': 'task'}),
                                    ('telemetry_read', {'task_id': 'task'}),
                                    ('telemetry_stop', {'task_id': 'task'})):
                response = pool.submit(client.post, '/rpc', content=envelope(coordinator, action, payload, action)).result(timeout=2)
                assert response.status_code == 200
                result = coordinator.open(response.content, 'rpc-response', action, 'client1')['payload']
                assert result['task_id'] == 'task'
        finally:
            release.set()
        assert future.result(timeout=3).status_code == 200
    assert len(list((tmp_path / 'nodes' / 'client1' / 'rpc-cache').glob('*.bin'))) == 1
    assert app.state.telemetry._closed is True


def test_stop_is_bounded_when_hardware_driver_is_stuck(tmp_path, monkeypatch):
    from dgfl.services import telemetry as module

    entered, release, closed = threading.Event(), threading.Event(), threading.Event()

    class StuckHardware:
        def sample(self):
            entered.set()
            assert release.wait(5)
            return {'cpu': {}, 'gpu': {}}

        def close(self):
            closed.set()

    monkeypatch.setattr(module, 'HardwareSampler', StuckHardware)
    telemetry = NodeTelemetry('authority1', tmp_path)
    telemetry.start({'task_id': 'task', 'collect_machine': True})
    assert entered.wait(2)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(telemetry.stop, {'task_id': 'task'}).result(timeout=2)
        assert result['active'] is False
        assert 'cleanup' in telemetry.read({'task_id': 'task'})['error']
    finally:
        release.set()
        assert closed.wait(3)
        telemetry.close()


@pytest.mark.parametrize('next_task', ['task', 'next-task'])
def test_pending_sampler_cleanup_prevents_overlapping_restart(tmp_path, monkeypatch, next_task):
    from dgfl.services import telemetry as module

    entered, release = threading.Event(), threading.Event()

    class StuckHardware:
        def sample(self):
            entered.set()
            assert release.wait(5)
            return {'cpu': {}, 'gpu': {}}

        def close(self):
            pass

    monkeypatch.setattr(module, 'HardwareSampler', StuckHardware)
    telemetry = NodeTelemetry('authority1', tmp_path)
    try:
        telemetry.start({'task_id': 'task', 'collect_machine': True})
        assert entered.wait(2)
        sampler = telemetry._active['thread']
        assert telemetry.stop({'task_id': 'task'})['active'] is False
        with pytest.raises(ValueError, match='cleanup'):
            telemetry.start({'task_id': next_task, 'collect_machine': True})
        # The pending optional probe never prevents training diagnostics.
        assert telemetry.record_epoch(next_task, 1, 1, epoch())
        release.set()
        sampler.join(timeout=3)
        assert not sampler.is_alive()
        assert telemetry.start({'task_id': next_task})['active'] is True
        assert any(row['kind'] == 'resource' for row in wait_event(telemetry, next_task))
        telemetry.stop({'task_id': next_task})
    finally:
        release.set()
        telemetry.close()
