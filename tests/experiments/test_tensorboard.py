"""TensorBoard's own reader verifies event compatibility and persistence."""
import importlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pytest


def logs_class():
    try:
        module = importlib.import_module('dgfl.experiments.tensorboard')
    except ModuleNotFoundError:
        module = None
    assert module is not None, 'TensorBoard event export is not implemented'
    return module.TensorBoardLogs


def read_scalars(path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    reader = EventAccumulator(str(path), size_guidance={'scalars': 0}, purge_orphaned_data=False).Reload()
    return {tag: reader.Scalars(tag) for tag in reader.Tags()['scalars']}


def record():
    return {
        'run_id': 'run-example', 'created_at': '2026-10-08T01:00:00+00:00',
        'initial_metrics': {'accuracy': .125, 'loss': 2.5},
        'rounds': [{'round': 1, 'accuracy': .75, 'loss': .875}],
        'events': [{'stage': 'completed_round', 'round': 1, 'time': '2026-10-08T01:01:00+00:00'}],
        'evidence': {'resources': {'hardware': {'samples': [sample()]}}},
    }


def sample():
    return {
        'sample_index': 7, 'round': 1, 'time': '2026-10-08T01:00:45+00:00',
        'cpu': {'system_utilization_percent': 25.},
        'gpu': {'utilization_percent': 80., 'memory_used_bytes': 2048, 'memory_total_bytes': 8192,
                'memory_utilization_percent': 99.},
        'system': {'memory_used_bytes': 4096, 'memory_total_bytes': 16384, 'memory_percent': 25.,
                   'disk_used_bytes': 16384, 'disk_total_bytes': 65536, 'disk_percent': 25.},
        'processes': {'rss_bytes': 1024, 'cpu_percent': 125.},
    }


def test_exports_round_steps_hardware_steps_and_real_wall_times(tmp_path):
    logs = logs_class()(tmp_path)
    assert logs.sync(record())['available']
    scalars = read_scalars(logs.logdir / 'run-example')
    assert [(item.step, item.value) for item in scalars['evaluation/accuracy']] == [(0, .125), (1, .75)]
    assert [(item.step, item.value) for item in scalars['evaluation/cross_entropy']] == [(0, 2.5), (1, .875)]
    assert scalars['evaluation/accuracy'][1].wall_time == datetime.fromisoformat('2026-10-08T01:01:00+00:00').timestamp()
    expected = {'system/cpu_percent': 25., 'system/gpu_utilization_percent': 80.,
                'system/gpu_memory_used_bytes': 2048., 'system/gpu_memory_percent': 25.,
                'system/memory_used_bytes': 4096., 'system/memory_percent': 25.,
                'system/disk_used_bytes': 16384., 'system/disk_percent': 25.,
                'processes/rss_bytes': 1024., 'processes/cpu_percent': 125.}
    for tag, value in expected.items():
        assert [(item.step, item.value) for item in scalars[tag]] == [(7, value)]
        assert scalars[tag][0].wall_time == datetime.fromisoformat(sample()['time']).timestamp()
    logs.close_all()


def test_repeat_sync_and_restart_do_not_duplicate_existing_points(tmp_path):
    logs = logs_class()(tmp_path)
    data = record()
    logs.sync(data)
    logs.write_sample(data['run_id'], sample())
    logs.sync(data)
    logs.close_all()
    restarted = logs_class()(tmp_path)
    data['rounds'].append({'round': 2, 'accuracy': .875, 'loss': .5})
    assert restarted.sync(data)['available']
    scalars = read_scalars(restarted.logdir / data['run_id'])
    assert [point.step for point in scalars['evaluation/accuracy']] == [0, 1, 2]
    assert len(scalars['system/cpu_percent']) == 1
    restarted.close_all()


def test_missing_and_invalid_metrics_do_not_become_zero(tmp_path):
    logs = logs_class()(tmp_path)
    assert logs.sync({'run_id': 'empty', 'initial_metrics': {'accuracy': True, 'loss': float('nan')},
                      'rounds': [{'round': 1, 'accuracy': 75, 'loss': None}]})['available']
    assert logs.write_sample('empty', {'sample_index': 0, 'cpu': {}, 'gpu': {},
                                       'system': {'memory_used_bytes': None},
                                       'processes': {'cpu_percent': float('inf')}})['available']
    assert read_scalars(logs.logdir / 'empty') == {}


def test_independent_instances_serialize_shared_exports(tmp_path):
    instances = [logs_class()(tmp_path) for _ in range(4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda logs: logs.sync(record()), instances))
    assert all(result['available'] for result in results)
    scalars = read_scalars(instances[0].logdir / 'run-example')
    assert len(scalars['evaluation/accuracy']) == 2
    for logs in instances:
        logs.close_all()


def test_live_reader_observes_alternating_exporters_and_restarts(tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    first, second = logs_class()(tmp_path), logs_class()(tmp_path)
    def write(logs, step):
        row = sample()
        row['sample_index'] = step
        assert logs.write_sample('live', row)['available']
    write(first, 1)
    write(second, 2)
    reader = EventAccumulator(str(first.logdir / 'live'), purge_orphaned_data=False).Reload()
    assert [point.step for point in reader.Scalars('system/cpu_percent')] == [1, 2]
    write(first, 3)
    reader.Reload()
    assert [point.step for point in reader.Scalars('system/cpu_percent')] == [1, 2, 3]
    first.close_all()
    write(logs_class()(tmp_path), 4)
    write(second, 5)
    reader.Reload()
    assert [point.step for point in reader.Scalars('system/cpu_percent')] == [1, 2, 3, 4, 5]


def test_torn_event_tail_is_preserved_and_new_events_remain_readable(tmp_path):
    logs = logs_class()(tmp_path)
    logs.write_sample('interrupted', sample())
    path = next((logs.logdir / 'interrupted').glob('events.out.tfevents.*'))
    with path.open('ab') as stream:
        stream.write(b'partial')
    previous = path.read_bytes()
    updated = sample()
    updated['sample_index'] = 8
    assert logs.write_sample('interrupted', updated)['available']
    assert path.read_bytes() == previous
    assert [point.step for point in read_scalars(path.parent)['system/cpu_percent']] == [7, 8]


@pytest.mark.parametrize('change', ['delete', 'truncate', 'replace'])
def test_cached_export_recovers_points_after_event_file_is_removed_or_replaced(tmp_path, change):
    logs = logs_class()(tmp_path)
    data = record()
    assert logs.sync(data)['available']
    path = next((logs.logdir / data['run_id']).glob('events.out.tfevents.*'))
    if change == 'delete':
        path.unlink()
    elif change == 'truncate':
        path.write_bytes(b'')
    else:
        # Replace the inode with a different valid stream under the same name.
        other = logs_class()(tmp_path / 'replacement')
        assert other.sync({'run_id': 'other', 'initial_metrics': {'accuracy': .5}})['available']
        replacement = next((other.logdir / 'other').glob('events.out.tfevents.*'))
        replacement.replace(path)
    assert logs.sync(data)['available']
    scalars = read_scalars(path.parent)
    expected_accuracy = [(0, .5), (1, .75)] if change == 'replace' else [(0, .125), (1, .75)]
    assert [(item.step, item.value) for item in scalars['evaluation/accuracy']] == expected_accuracy
    assert [(item.step, item.value) for item in scalars['evaluation/cross_entropy']] == [(0, 2.5), (1, .875)]
    assert [(item.step, item.value) for item in scalars['system/cpu_percent']] == [(7, 25.)]
    assert logs.sync(data)['available']
    assert len(read_scalars(path.parent)['evaluation/accuracy']) == 2


@pytest.mark.parametrize('rewritten_steps, expected', [
    ([2], [(1, .75), (2, .75)]),
    ([2, 3], [(1, .75), (2, .75), (3, .75)]),
])
def test_cached_export_rescans_in_place_event_rewrite(tmp_path, rewritten_steps, expected):
    import os

    from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.summary.writer.record_writer import RecordWriter

    logs = logs_class()(tmp_path)
    data = {'run_id': 'rewritten', 'rounds': [{'round': 1, 'accuracy': .75}]}
    assert logs.sync(data)['available']
    path = next((logs.logdir / data['run_id']).glob('events.out.tfevents.*'))
    before = path.stat()
    events = [Event.FromString(raw) for raw in RawEventFileLoader(str(path)).Load()]
    with path.open('wb') as stream:
        writer = RecordWriter(stream)
        for event in events:
            if event.summary.value:
                for step in rewritten_steps:
                    event.step = step
                    writer.write(event.SerializeToString())
            else:
                writer.write(event.SerializeToString())
        writer.flush()
    assert path.stat().st_ino == before.st_ino
    if len(rewritten_steps) == 1:
        assert path.stat().st_size == before.st_size
    else:
        assert path.stat().st_size > before.st_size
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
    assert logs.sync(data)['available']
    points = read_scalars(path.parent)['evaluation/accuracy']
    assert sorted((item.step, item.value) for item in points) == expected


def test_missing_dependency_has_a_friendly_status_and_never_raises(tmp_path, monkeypatch):
    import builtins
    real_import = builtins.__import__
    def missing(name, *args, **kwargs):
        if name.startswith('tensorboard'):
            raise ModuleNotFoundError('tensorboard is not installed')
        return real_import(name, *args, **kwargs)
    logs = logs_class()(tmp_path)
    monkeypatch.setattr(builtins, '__import__', missing)
    result = logs.sync(record())
    assert result['available'] is False
    assert 'TensorBoard' in result['reason']
    assert not logs.logdir.exists()


@pytest.mark.parametrize('run_id', ['../outside', '..', '/absolute', 'bad/name', 'bad\\name', 'C:escape', 'CON'])
def test_rejects_unsafe_run_directory_names(tmp_path, run_id):
    logs = logs_class()(tmp_path)
    result = logs.sync({'run_id': run_id, 'initial_metrics': {'accuracy': .5}})
    assert result['available'] is False
    assert result['reason']


def test_symlink_log_directory_is_rejected(tmp_path):
    outside = tmp_path / 'outside'
    outside.mkdir()
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    try:
        (runtime / 'tensorboard').symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip('symlink creation unavailable')
    logs = logs_class()(runtime)
    assert logs.sync(record())['available'] is False
    assert not list(outside.iterdir())


def test_io_failure_is_diagnostic_and_does_not_abort_experiment(tmp_path):
    (tmp_path / 'tensorboard').write_text('blocked', encoding='utf8')
    result = logs_class()(tmp_path).sync(record())
    assert result['available'] is False
    assert result['reason']


def test_legacy_memory_evidence_exports_hardware_samples(tmp_path):
    logs = logs_class()(tmp_path)
    data = record()
    data['evidence']['memory'] = data['evidence'].pop('resources')
    assert logs.sync(data)['available']
    assert read_scalars(logs.logdir / data['run_id'])['system/cpu_percent'][0].value == 25.


def test_status_allows_retry_after_a_transient_export_error(tmp_path):
    blocker = tmp_path / 'tensorboard'
    blocker.write_text('blocked', encoding='utf8')
    logs = logs_class()(tmp_path)
    assert logs.sync(record())['available'] is False
    blocker.unlink()
    assert logs.status()['available'] is True
    assert logs.sync(record())['available'] is True
