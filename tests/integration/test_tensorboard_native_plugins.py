"""Embedded TensorBoard registers the official Angular Time Series frontend."""

import json
import re
import zipfile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from dgfl.experiments.tensorboard import KEY_SCALAR_TAGS, TensorBoardLogs
from dgfl.services.tensorboard import EmbeddedTensorBoard


@pytest.fixture
def native_board(tmp_path):
    logs = TensorBoardLogs(tmp_path)
    assert logs.sync({'run_id': 'history', 'initial_metrics': {'accuracy': .25, 'loss': 2.}})['available']
    assert logs.write_node_event('history', {'id': 'client1', 'machine': 'local'}, {
        'kind': 'epoch', 'time': '2026-10-10T01:00:00+00:00',
        'payload': {'step': 3, 'epoch': 1, 'round_id': 2, 'accuracy': .75, 'loss': .5,
                    'samples': 12, 'elapsed_s': .1, 'metrics_eval_s': .01}})['available']
    # Already-exported diagnostics remain in the old append-only files. The
    # native provider must curate both old history and subsequent exports.
    assert logs.sync({'run_id': 'history', 'rounds': [{
        'round': 1, 'duration_s': 2., 'reference_norm_squared': 123.,
        'stage_times': {'validation_s': .2, 'aggregation_s': .8},
        'proof_verification_metrics': {'authority1': {'verifications': {
            'client1': {'proof_verify_s': .125, 'inner_product_s': .0625},
            'client2': {'proof_verify_s': .375, 'inner_product_s': .0625}}}},
        'combine_metrics': {'coordinator': {'combine_timings': {
            'combine_total_s': .7, 'gpu_profile': {
                'totals': {'host_wall_seconds': .5, 'kernel_seconds': .25, 'batches': 7},
                'by_kernel': {'dgfl_gt_validate': {'kernel_seconds': .25}},
                'configuration': {'block_size': 128}}}}}}]})['available']
    raw = EventAccumulator(str(logs.logdir / 'history')).Reload()
    assert raw.Scalars('gpu_compute/kernels/dgfl_gt_validate/kernel_ms')[0].value == 250.
    assert raw.Scalars('model/reference_norm_squared')[0].value == 123.
    service = EmbeddedTensorBoard(logs, reload_interval=0)
    app = FastAPI()
    app.mount('/tensorboard', service)
    try:
        with TestClient(app) as http:
            yield http
    finally:
        service.close()
        logs.close_all()


def test_only_official_required_plugins_are_registered_and_timeseries_is_angular(native_board):
    response = native_board.get('/tensorboard/data/plugins_listing')
    assert response.status_code == 200
    listing = response.json()
    assert set(listing) == {'timeseries', 'scalars', 'hparams'}
    assert listing['timeseries']['enabled'] is True
    assert listing['timeseries']['tab_name'] == 'Time Series'
    assert listing['timeseries']['loading_mechanism'] == {'type': 'NG_COMPONENT'}
    assert listing['scalars']['loading_mechanism'] == {
        'type': 'CUSTOM_ELEMENT', 'element_name': 'tf-scalar-dashboard'}


def test_modern_timeseries_endpoints_serve_actual_global_and_node_values(native_board):
    response = native_board.get('/tensorboard/data/plugin/timeseries/tags')
    assert response.status_code == 200
    tags = response.json()['scalars']['runTagInfo']
    assert 'evaluation/accuracy' in tags['history']
    node_run = next(run for run in tags if run.endswith('client1'))
    assert 'training/accuracy' in tags[node_run]
    requests = [{'plugin': 'scalars', 'tag': 'evaluation/accuracy', 'run': 'history'},
                {'plugin': 'scalars', 'tag': 'training/accuracy', 'run': node_run}]
    response = native_board.get('/tensorboard/data/plugin/timeseries/timeSeries',
                                params={'requests': json.dumps(requests)})
    assert response.status_code == 200
    series = response.json()
    assert [(row['step'], row['value']) for row in series[0]['runToSeries']['history']] == [(0, .25)]
    assert [(row['step'], row['value']) for row in series[1]['runToSeries'][node_run]] == [(3, .75)]
    # Older exports and existing callers keep the native scalars API.
    legacy = native_board.get('/tensorboard/data/plugin/scalars/scalars',
                             params={'run': 'history', 'tag': 'evaluation/accuracy'})
    assert legacy.status_code == 200 and legacy.json()[0][1:] == [0, .25]


def test_history_discovery_and_native_reads_only_expose_key_charts(native_board):
    tags = native_board.get('/tensorboard/data/plugin/timeseries/tags').json()['scalars']['runTagInfo']
    visible = {tag for run_tags in tags.values() for tag in run_tags}
    assert visible <= KEY_SCALAR_TAGS
    assert 'gpu_compute/totals/kernel_ms' in visible
    assert 'proof/verification/proof_verify_total_s' in visible
    legacy_tags = native_board.get('/tensorboard/data/plugin/scalars/tags').json()
    assert {tag for run_tags in legacy_tags.values() for tag in run_tags} == visible
    edge = next(run for run in tags if run.endswith('authority1'))
    requests = [{'plugin': 'scalars', 'tag': tag, 'run': run} for run, tag in (
        ('history', 'gpu_compute/totals/kernel_ms'),
        ('history', 'gpu_compute/kernels/dgfl_gt_validate/kernel_ms'),
        (edge, 'proof/verification/proof_verify_total_s'),
        (edge, 'proof/verification/client1/proof_verify_s'))]
    response = native_board.get('/tensorboard/data/plugin/timeseries/timeSeries',
                                params={'requests': json.dumps(requests)})
    assert response.status_code == 200
    series = response.json()
    assert series[0]['runToSeries']['history'][0]['value'] == 250.
    assert series[1]['runToSeries'] == {}
    assert series[2]['runToSeries'][edge][0]['value'] == .5
    assert series[3]['runToSeries'] == {}


@pytest.mark.parametrize('endpoint', ['experiment', 'session_groups'])
def test_native_run_selector_hparams_dependency_is_available(native_board, endpoint):
    response = native_board.post('/tensorboard/data/plugin/hparams/' + endpoint,
                                 content='{}', headers={'Content-Type': 'application/json'})
    assert response.status_code == 200, response.text
    assert isinstance(response.json(), dict)


def test_pinned_native_frontend_supports_persistent_auto_reload_defaults():
    """Detect an upstream preference-contract change before seeding defaults.

    The embedded application initializes only missing preferences before the
    iframe starts; TensorBoard itself subsequently owns this settings object.
    Check the shipped native asset, not a second copy of its reducer logic.
    """
    from tensorboard import assets, version

    assert version.VERSION.startswith('2.21.')
    with assets.get_default_assets_zip_provider()() as stream, zipfile.ZipFile(stream) as archive:
        scripts = '\n'.join(archive.read(name).decode('utf8') for name in archive.namelist()
                            if name.endswith('.js'))
    # Native preferences migrate the old timeseries-specific object and read
    # the current global object; preserving both avoids overriding a user's
    # previously selected autoReload=false.
    assert '"_tb_global_settings"' in scripts
    assert '"_tb_global_settings.timeseries"' in scripts
    assert re.search(r'hasOwnProperty\(["\']autoReload["\']\).*?["\']boolean["\']==typeof', scripts)
    assert 'autoReloadPeriodInMs' in scripts
    # Keep the native interval instead of silently requesting an unsupported
    # 5-second period: the official settings form enforces at least 30 seconds.
    assert re.search(r'MIN_RELOAD_PERIOD_IN_S\s*=\s*30\b', scripts)
    assert re.search(r'reloadPeriodInMs\s*:\s*(?:3e4|30000)\b', scripts)
