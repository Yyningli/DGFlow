"""Native TensorBoard charts compare global tests and round-aligned training."""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dgfl.experiments.tensorboard import KEY_SCALAR_TAGS, TensorBoardLogs
from dgfl.services.tensorboard import EmbeddedTensorBoard


@pytest.fixture(scope='module')
def comparison_board(tmp_path_factory):
    logs = TensorBoardLogs(tmp_path_factory.mktemp('native-client-comparison'))
    record = {
        'run_id': 'comparison', 'config': {'local_epochs': 2},
        'initial_metrics': {'accuracy': .125, 'loss': 2.},
        'rounds': [{
            'round': 1, 'accuracy': .5, 'loss': 1.25,
            'client_training_metrics': {
                'client1': [
                    {'round_id': 1, 'epoch': 1, 'step': 1, 'accuracy': .625, 'loss': 1.},
                    {'round_id': 1, 'epoch': 2, 'step': 2, 'accuracy': .75, 'loss': .5}],
                # Order of delivery does not change which epoch ended this round.
                'client2': [
                    {'round_id': 1, 'epoch': 2, 'step': 2, 'accuracy': .625, 'loss': .75},
                    {'round_id': 1, 'epoch': 1, 'step': 1, 'accuracy': .375, 'loss': 1.5}],
            },
        }, {
            'round': 2, 'accuracy': .875, 'loss': .25,
            'client_training_metrics': {'client1': [
                {'round_id': 2, 'epoch': 1, 'step': 3, 'accuracy': .8125, 'loss': .375}]},
        }],
    }
    assert logs.sync(record)['available']
    # History replay must neither add comparison points nor overwrite epochs.
    assert logs.sync(record)['available']
    assert logs.sync({'run_id': 'legacy', 'initial_metrics': {'accuracy': .25},
                      'rounds': [{'round': 1, 'accuracy': .5}]})['available']
    final_epoch = {'round_id': 1, 'epoch': 2, 'step': 2, 'accuracy': .75, 'loss': .5}
    assert logs.sync({
        'run_id': 'unknown-epochs',
        'rounds': [{'round': 1, 'accuracy': .5,
                    'client_training_metrics': {'client1': [final_epoch]}}],
    })['available']
    assert logs.sync({'run_id': 'incomplete', 'config': {'local_epochs': 2},
                      'initial_metrics': {'accuracy': .125}, 'rounds': []})['available']
    # A live final-epoch event alone does not prove a global round completed.
    assert logs.write_node_event('incomplete', {'id': 'client1'}, {
        'kind': 'epoch', 'time': '2026-10-10T01:00:00+00:00',
        'payload': final_epoch})['available']
    service = EmbeddedTensorBoard(logs, reload_interval=0)
    app = FastAPI()
    app.mount('/tensorboard', service)
    try:
        with TestClient(app) as http:
            yield http
    finally:
        service.close()
        logs.close_all()


def test_native_global_chart_has_root_and_two_clients_with_measured_values(comparison_board):
    response = comparison_board.get('/tensorboard/data/plugin/timeseries/tags')
    assert response.status_code == 200
    tags = response.json()['scalars']['runTagInfo']
    runs = ['comparison', *(next(run for run in tags if run.replace('\\', '/').startswith('comparison/')
                                and run.endswith(client)) for client in ('client1', 'client2'))]
    chart_tags = ('evaluation/accuracy', 'evaluation/cross_entropy')
    # Identical tags give native TensorBoard one chart with three run-colored
    # curves, rather than creating a separate chart tag for each client.
    for run in runs:
        assert {tag for tag in tags[run] if tag.startswith('evaluation/')} == set(chart_tags)
    requests = [{'plugin': 'scalars', 'tag': tag, 'run': run} for tag in chart_tags for run in runs]
    response = comparison_board.get('/tensorboard/data/plugin/timeseries/timeSeries',
                                    params={'requests': json.dumps(requests)})
    assert response.status_code == 200
    actual = [[(point['step'], point['value']) for point in row['runToSeries'][request['run']]]
              for request, row in zip(requests, response.json(), strict=True)]
    assert actual == [
        [(0, .125), (1, .5), (2, .875)], [(1, .75)], [(1, .625)],
        [(0, 2.), (1, 1.25), (2, .25)], [(1, .5)], [(1, .75)],
    ]


def test_native_training_epochs_retain_their_distinct_axis(comparison_board):
    tags = comparison_board.get('/tensorboard/data/plugin/timeseries/tags').json()['scalars']['runTagInfo']
    node_run = next(run for run in tags if run.replace('\\', '/').startswith('comparison/')
                    and run.endswith('client1'))
    requests = [{'plugin': 'scalars', 'tag': tag, 'run': node_run}
                for tag in ('evaluation/accuracy', 'training/accuracy')]
    response = comparison_board.get('/tensorboard/data/plugin/timeseries/timeSeries',
                                    params={'requests': json.dumps(requests)})
    assert response.status_code == 200
    comparison, training = response.json()
    # Epoch 2 aligns with global round 1 only on the comparison chart. Its
    # original cumulative-epoch chart still includes epoch 3 from the next round.
    assert [(row['step'], row['value']) for row in comparison['runToSeries'][node_run]] == [(1, .75)]
    assert [(row['step'], row['value']) for row in training['runToSeries'][node_run]] == [
        (1, .625), (2, .75), (3, .8125)]


def test_native_discovery_keeps_the_existing_bounded_chart_policy(comparison_board):
    tags = comparison_board.get('/tensorboard/data/plugin/timeseries/tags').json()['scalars']['runTagInfo']
    assert len(KEY_SCALAR_TAGS) == 40
    visible = {tag for run_tags in tags.values() for tag in run_tags}
    assert visible <= KEY_SCALAR_TAGS
    assert {tag for tag in visible if tag.startswith('evaluation/')} == {
        'evaluation/accuracy', 'evaluation/cross_entropy'}
    legacy_tags = comparison_board.get('/tensorboard/data/plugin/scalars/tags')
    assert legacy_tags.status_code == 200
    assert {tag for run_tags in legacy_tags.json().values() for tag in run_tags} == visible


def test_native_history_does_not_invent_unrecorded_or_uncompleted_client_points(comparison_board):
    tags = comparison_board.get('/tensorboard/data/plugin/timeseries/tags').json()['scalars']['runTagInfo']
    assert 'evaluation/accuracy' in tags['legacy']
    assert not any(run.replace('\\', '/').startswith('legacy/') for run in tags)
    requests = []
    for prefix in ('unknown-epochs/', 'incomplete/'):
        node_run = next(run for run in tags if run.replace('\\', '/').startswith(prefix) and run.endswith('client1'))
        assert 'training/accuracy' in tags[node_run]
        assert 'evaluation/accuracy' not in tags[node_run]
        assert 'evaluation/cross_entropy' not in tags[node_run]
        requests.append({'plugin': 'scalars', 'tag': 'evaluation/accuracy', 'run': node_run})
    response = comparison_board.get('/tensorboard/data/plugin/timeseries/timeSeries',
                                    params={'requests': json.dumps(requests)})
    assert response.status_code == 200
    assert all(row['runToSeries'] == {} for row in response.json())
