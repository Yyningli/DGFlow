"""Round overlays recover actual client epochs from independent telemetry."""
import json
from collections import Counter
from copy import deepcopy

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from dgfl.experiments.tensorboard import TensorBoardLogs

TIME = '2026-10-10T01:00:00+00:00'


def epoch(round_id, index, *, accuracy=.75, loss=.5):
    return {'round_id': round_id, 'epoch': index, 'step': (round_id - 1) * 2 + index,
            'accuracy': accuracy, 'loss': loss, 'samples': 12, 'elapsed_s': .02,
            'metrics_eval_s': .003}


def record(*completed, run_id='journal-overlay'):
    return {'run_id': run_id, 'created_at': TIME, 'status': 'running',
            'config': {'local_epochs': 2},
            'rounds': [{'round': index, 'accuracy': .5, 'loss': 1., 'client_training_metrics': {}}
                       for index in completed],
            'evidence': {'node_snapshot': [{'id': 'client1', 'machine': 'local'},
                                          {'id': 'client2', 'machine': 'local'}]}}


def append(runtime, items, *, node_id='client1', run_id='journal-overlay'):
    directory = runtime / 'monitoring' / run_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{node_id}.segment.jsonl'
    rows = [{'node_id': node_id, 'event': {'kind': 'epoch', 'time': TIME,
                                         'sequence': index, 'payload': payload}}
            for index, payload in enumerate(items, 1)]
    with path.open('a', encoding='utf8') as stream:
        for row in rows:
            stream.write(json.dumps(row) + '\n')
    return path


def scalars(logs, *, run_id='journal-overlay', node_id='client1'):
    path = logs.logdir / run_id / 'nodes' / node_id
    if not path.exists():
        return {}
    reader = EventAccumulator(str(path), size_guidance={'scalars': 0},
                              purge_orphaned_data=False).Reload()
    return {tag: [(point.step, point.value) for point in reader.Scalars(tag)]
            for tag in reader.Tags()['scalars']}


def test_early_epoch_cursor_can_advance_before_completed_round_without_losing_overlay(tmp_path):
    logs = TensorBoardLogs(tmp_path)
    path = append(tmp_path, [epoch(1, 1, accuracy=.25), epoch(1, 2, accuracy=.75)])
    assert logs.sync(record())['available']
    assert logs._replay_offsets['journal-overlay'][path.name] == path.stat().st_size
    measured = scalars(logs)
    assert measured['training/accuracy'] == [(1, .25), (2, .75)]
    assert 'evaluation/accuracy' not in measured
    assert ('client1', 1) in logs._replay_pending['journal-overlay']

    completed = record(1)
    original = deepcopy(completed)
    assert logs.sync(completed)['available']
    assert scalars(logs)['evaluation/accuracy'] == [(1, .75)]
    assert scalars(logs)['evaluation/cross_entropy'] == [(1, .5)]
    assert logs._replay_pending['journal-overlay'] == {}
    assert logs.sync(completed)['available']
    assert len(scalars(logs)['evaluation/accuracy']) == 1
    assert completed == original


def test_late_epoch_after_completed_round_is_exported_only_when_final_epoch_arrives(tmp_path):
    logs = TensorBoardLogs(tmp_path)
    completed = record(1)
    assert logs.sync(completed)['available']
    append(tmp_path, [epoch(1, 1)])
    assert logs.sync(completed)['available']
    assert 'evaluation/accuracy' not in scalars(logs)
    append(tmp_path, [epoch(1, 2, accuracy=.5, loss=1.25)])
    assert logs.sync(completed)['available']
    assert scalars(logs)['evaluation/accuracy'] == [(1, .5)]
    assert scalars(logs)['evaluation/cross_entropy'] == [(1, 1.25)]


def test_partial_overlay_write_retries_both_metrics_without_duplicate_persisted_prefix(tmp_path, monkeypatch):
    logs = TensorBoardLogs(tmp_path)
    path = append(tmp_path, [epoch(1, 1), epoch(1, 2)])
    assert logs.sync(record())['available']
    original_write = logs._write
    failed = []

    def write(run_id, batches, *, scope=None):
        if scope == ('nodes', 'client1') and not failed:
            failed.append(True)
            step, wall_time, metrics = next(batch for batch in batches if 'evaluation/accuracy' in batch[2])
            assert original_write(run_id, [(step, wall_time, {'evaluation/accuracy': metrics['evaluation/accuracy']})],
                                  scope=scope)['available']
            return {'available': False, 'reason': 'interrupted after one metric persisted'}
        return original_write(run_id, batches, scope=scope)

    monkeypatch.setattr(logs, '_write', write)
    assert logs.sync(record(1))['available'] is False
    assert ('client1', 1) in logs._replay_pending['journal-overlay']
    assert logs._replay_offsets['journal-overlay'][path.name] == path.stat().st_size
    assert logs.sync(record(1))['available']
    measured = scalars(logs)
    assert measured['evaluation/accuracy'] == [(1, .75)]
    assert measured['evaluation/cross_entropy'] == [(1, .5)]
    assert measured['training/accuracy'] == [(1, .75), (2, .75)]


def test_failed_first_journal_batch_preserves_cursor_and_recovers_final_epoch(tmp_path, monkeypatch):
    logs = TensorBoardLogs(tmp_path)
    path = append(tmp_path, [epoch(1, 1), epoch(1, 2)])
    original_write = logs._write
    failed = []

    def write(run_id, batches, *, scope=None):
        if scope == ('nodes', 'client1') and not failed:
            failed.append(True)
            assert original_write(run_id, batches[:1], scope=scope)['available']
            return {'available': False, 'reason': 'interrupted node batch'}
        return original_write(run_id, batches, scope=scope)

    monkeypatch.setattr(logs, '_write', write)
    assert logs.sync(record(1))['available'] is False
    assert path.name not in logs._replay_offsets['journal-overlay']
    assert logs.sync(record(1))['available']
    assert logs._replay_offsets['journal-overlay'][path.name] == path.stat().st_size
    assert scalars(logs)['evaluation/accuracy'] == [(1, .75)]
    assert len(scalars(logs)['training/accuracy']) == 2


@pytest.mark.parametrize('cleanup', ['restart', 'close', 'close_all', 'evict'])
def test_persistent_journal_recovers_pending_history_after_cache_cleanup(tmp_path, cleanup):
    logs = TensorBoardLogs(tmp_path)
    append(tmp_path, [epoch(index, local_epoch, accuracy=index / 4)
                      for index in (1, 2, 3) for local_epoch in (1, 2)])
    assert logs.sync(record())['available']
    if cleanup == 'restart':
        logs = TensorBoardLogs(tmp_path)
    elif cleanup == 'close':
        logs.close('journal-overlay')
    elif cleanup == 'close_all':
        logs.close_all()
    else:
        for index in range(8):
            assert logs.sync(record(run_id=f'other-{index}'))['available']
    assert 'journal-overlay' not in logs._replay_pending
    assert 'journal-overlay' not in logs._replay_offsets
    assert logs.sync(record(1, 2, 3))['available']
    assert scalars(logs)['evaluation/accuracy'] == [(1, .25), (2, .5), (3, .75)]
    assert [step for step, _ in scalars(logs)['training/accuracy']] == [1, 2, 3, 4, 5, 6]


def test_configured_last_epoch_uses_round_axis_and_never_publishes_unfinished_round(tmp_path):
    logs = TensorBoardLogs(tmp_path)
    append(tmp_path, [epoch(index, local_epoch, accuracy=index / 4)
                      for index in (1, 2, 3) for local_epoch in (1, 2)])
    data = record(1, 2)
    data['status'] = 'failed'
    assert logs.sync(data)['available']
    measured = scalars(logs)
    assert measured['evaluation/accuracy'] == [(1, .25), (2, .5)]
    assert [step for step, _ in measured['training/accuracy']] == [1, 2, 3, 4, 5, 6]
    assert ('client1', 3) in logs._replay_pending['journal-overlay']


@pytest.mark.parametrize('field,value', [('round_id', True), ('epoch', 1), ('epoch', True),
                                        ('step', 3), ('step', 2.), ('accuracy', None)])
def test_journal_overlay_uses_existing_strict_final_epoch_validation(tmp_path, field, value):
    logs = TensorBoardLogs(tmp_path)
    item = epoch(1, 2)
    item[field] = value
    if field == 'accuracy':
        item['loss'] = None
    append(tmp_path, [item])
    # The regular epoch exporter also rejects a non-integer step; otherwise
    # malformed final-epoch metadata must not suppress valid training records.
    assert logs.sync(record(1))['available'] is (type(item['step']) is int)
    assert 'evaluation/accuracy' not in scalars(logs)
    assert 'evaluation/cross_entropy' not in scalars(logs)


def test_summary_and_journal_duplicate_final_epochs_are_batched_once_per_scope(tmp_path, monkeypatch):
    logs = TensorBoardLogs(tmp_path)
    items = [epoch(1, 1), epoch(1, 2)]
    append(tmp_path, items)
    data = record(1)
    data['rounds'][0]['client_training_metrics'] = {'client1': [{**item, 'time': TIME} for item in items]}
    calls = []
    original_write = logs._write

    def write(run_id, batches, *, scope=None):
        calls.append(scope)
        return original_write(run_id, batches, scope=scope)

    monkeypatch.setattr(logs, '_write', write)
    assert logs.sync(data)['available']
    assert Counter(calls) == Counter({None: 1, ('nodes', 'client1'): 1})
    assert scalars(logs)['evaluation/accuracy'] == [(1, .75)]
    assert len(scalars(logs)['training/accuracy']) == 2


def test_pending_cache_limit_restarts_replay_without_losing_persisted_evidence(tmp_path):
    logs = TensorBoardLogs(tmp_path)
    logs._max_pending_epochs = 1
    path = append(tmp_path, [epoch(1, 2), epoch(2, 2)])
    assert logs.sync(record())['available']
    assert logs._replay_pending['journal-overlay'] == {}
    assert logs._replay_offsets['journal-overlay'] == {}
    assert logs.sync(record(1, 2))['available']
    assert scalars(logs)['evaluation/accuracy'] == [(1, .75), (2, .75)]
    assert logs._replay_offsets['journal-overlay'][path.name] == path.stat().st_size
