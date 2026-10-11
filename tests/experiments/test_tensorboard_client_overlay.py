"""Compare global tests and final local training epochs on the round axis."""
from copy import deepcopy

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from dgfl.experiments.tensorboard import KEY_SCALAR_TAGS, TensorBoardLogs

TIME = '2026-10-10T01:00:00+00:00'


def epoch(round_id, local_epoch, *, accuracy=.75, loss=.5):
    return {'time': TIME, 'epoch': local_epoch, 'round_id': round_id,
            'step': (round_id - 1) * 2 + local_epoch, 'accuracy': accuracy, 'loss': loss,
            'samples': 12, 'elapsed_s': .02, 'metrics_eval_s': .003}


def record():
    return {'run_id': 'client-overlay', 'created_at': TIME, 'config': {'local_epochs': 2},
            'initial_metrics': {'accuracy': .1, 'loss': 2.5},
            'rounds': [{'round': index, 'accuracy': index * .2, 'loss': 2. - index * .2,
                        'client_training_metrics': {
                            'client1': [epoch(index, 2, accuracy=.8, loss=.25),
                                        epoch(index, 1, accuracy=.5, loss=1.5)],
                            'client2': [epoch(index, 1), epoch(index, 2, accuracy=.6, loss=.75)]}}
                       for index in (1, 2)],
            'evidence': {'node_snapshot': [{'id': 'client1'}, {'id': 'client2'}]}}


def scalars(path):
    events = EventAccumulator(str(path), size_guidance={'scalars': 0}).Reload()
    return {tag: [(point.step, point.value) for point in events.Scalars(tag)]
            for tag in events.Tags()['scalars']}


def test_same_charts_show_final_client_epochs_on_global_round_axis_without_changing_originals(tmp_path):
    logs = TensorBoardLogs(tmp_path)
    data = record()
    original = deepcopy(data)
    assert logs.sync(data)['available']
    assert logs.sync(data)['available']
    root = scalars(logs.logdir / data['run_id'])
    client1 = scalars(logs.logdir / data['run_id'] / 'nodes/client1')
    client2 = scalars(logs.logdir / data['run_id'] / 'nodes/client2')
    assert [step for step, _ in root['evaluation/accuracy']] == [0, 1, 2]
    assert client1['evaluation/accuracy'] == [(1, pytest.approx(.8)), (2, pytest.approx(.8))]
    assert client2['evaluation/cross_entropy'] == [(1, .75), (2, .75)]
    # Original epoch granularity and ordering are preserved, even if observations
    # were received out of order. Overlay selection uses explicit epoch indices.
    assert sorted(step for step, _ in client1['training/accuracy']) == [1, 2, 3, 4]
    assert all(tag in KEY_SCALAR_TAGS for tag in client1 if tag.startswith('evaluation/'))
    assert data == original


@pytest.mark.parametrize('missing', ['config', 'local_epochs', 'client_metrics', 'final_epoch'])
def test_unrecorded_or_incomplete_history_does_not_fabricate_client_round_points(tmp_path, missing):
    data = record()
    if missing == 'config':
        data.pop('config')
    elif missing == 'local_epochs':
        data['config'] = {}
    elif missing == 'client_metrics':
        for row in data['rounds']:
            row.pop('client_training_metrics')
    else:
        for row in data['rounds']:
            row['client_training_metrics']['client1'] = [epoch(row['round'], 1)]
    logs = TensorBoardLogs(tmp_path)
    assert logs.sync(data)['available']
    path = logs.logdir / data['run_id'] / 'nodes/client1'
    if path.exists():
        assert 'evaluation/accuracy' not in scalars(path)
    assert len(scalars(logs.logdir / data['run_id'])['evaluation/accuracy']) == 3


def test_live_epochs_do_not_publish_an_immutable_round_comparison_point_early(tmp_path):
    logs = TensorBoardLogs(tmp_path)
    for index in (1, 2):
        assert logs.write_node_event('live', {'id': 'client1'}, {
            'kind': 'epoch', 'time': TIME, 'payload': epoch(1, index)})['available']
    measured = scalars(logs.logdir / 'live/nodes/client1')
    assert measured['training/accuracy'] == [(1, .75), (2, .75)]
    assert 'evaluation/accuracy' not in measured


@pytest.mark.parametrize('field,value', [
    ('round_id', 2), ('epoch', 1), ('epoch', True), ('step', 3), ('step', 2.),
])
def test_mismatched_final_epoch_metadata_is_rejected(field, value):
    item = epoch(1, 2)
    item[field] = value
    assert TensorBoardLogs._client_round_batch(1, 2, [item]) is None


def test_missing_invalid_and_zero_values_are_handled_independently():
    item = epoch(1, 2, accuracy=0., loss=float('nan'))
    assert TensorBoardLogs._client_round_batch(1, 2, [item])[2] == {'evaluation/accuracy': 0.}
    item.update(accuracy=1.1, loss=0.)
    assert TensorBoardLogs._client_round_batch(1, 2, [item])[2] == {'evaluation/cross_entropy': 0.}
    item.update(accuracy=True, loss=None)
    assert TensorBoardLogs._client_round_batch(1, 2, [item]) is None


def test_ambiguous_final_epoch_and_unknown_epoch_configuration_are_not_guessed():
    item = epoch(1, 2)
    assert TensorBoardLogs._client_round_batch(1, 2, [item, dict(item)]) is None
    for count in (None, 0, -1, True, 2.):
        assert TensorBoardLogs._client_round_batch(1, count, [item]) is None
