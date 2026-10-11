"""Training diagnostics use their own wire, including chunking and exact retry."""

import json
import threading
from contextlib import contextmanager
from copy import deepcopy

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from dgfl.crypto import backend as b
from dgfl.experiments.monitoring import RunNodeMonitor
from dgfl.experiments.tensorboard import TensorBoardLogs
from dgfl.services import node, roles
from dgfl.services.telemetry import NodeTelemetry
from dgfl.services.tensorboard import EmbeddedTensorBoard
from dgfl.training import datasets
from dgfl.transport import chunking, client
from dgfl.transport.security import Identity, create_cluster


def rpc_client(identity, http, directory):
    rpc = object.__new__(client.RPCClient)
    rpc.identity, rpc.client = identity, http
    rpc.nodes = {'client1': {'url': ''}}
    rpc.bytes_sent, rpc._counter_lock = 0, threading.Lock()
    rpc.spool_dir = directory
    directory.mkdir()
    return rpc


@pytest.mark.parametrize('request_chunked,response_chunked', [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize('lost_response', [False, True])
def test_train_wire_excludes_metrics_and_business_counter_uses_exact_frames(
        tmp_path, monkeypatch, request_chunked, response_chunked, lost_response):
    create_cluster(tmp_path / 'keys', dict.fromkeys(('coordinator', 'client1'), '127.0.0.1'))
    coordinator, receiver = Identity(tmp_path / 'keys', 'coordinator'), Identity(tmp_path / 'keys', 'client1')
    monkeypatch.setattr(client, 'SEND_THRESHOLD', 256)
    monkeypatch.setattr(client, 'CHUNK_BYTES', 128)
    monkeypatch.setattr(node, 'SEND_THRESHOLD', 256)
    monkeypatch.setattr(node, 'ResponseStore', lambda directory: chunking.ResponseStore(directory, chunk_bytes=128))
    calls = []
    packet = b'ciphertext' * (100 if response_chunked else 1)
    expected = {'packet': packet, 'training_s': .25, 'encrypt_s': .1, 'proof_s': .2}

    class Worker:
        def __init__(self, _node_id, _runtime, _identity, telemetry):
            self.telemetry = telemetry

        def execute(self, action, payload):
            assert action == 'train'
            calls.append(payload)
            epochs = []
            for epoch in (1, 2):
                metrics = {'epoch': epoch, 'accuracy': .75, 'loss': .4, 'samples': 10,
                           'elapsed_s': .1, 'metrics_eval_s': .01}
                epochs.append(metrics)
                self.telemetry.record_epoch('wire-task', 1, 2, metrics)
            result = {**expected, 'training_metrics': epochs, 'metrics_eval_s': .02, 'metrics_callback_s': .003}
            self.telemetry.record_training_summary('wire-task', 1, 2, result)
            return result

    monkeypatch.setattr(roles, 'RoleWorker', Worker)
    app = node.create_node_app('client1', tmp_path)
    with TestClient(app) as http:
        rpc = rpc_client(coordinator, http, tmp_path / 'business-spool')
        original = http.stream
        sent, received, actions = [], [], []
        lost = False

        @contextmanager
        def stream(method, endpoint, **kwargs):
            nonlocal lost
            frame = kwargs['content']
            sent.append(frame)
            message = receiver.open(frame, 'rpc', sender='coordinator')['payload']
            actions.append(message['action'])
            with original(method, endpoint, **kwargs) as response:
                is_train_result = (message['action'] == 'train'
                                   or (message['action'] == chunking.CHUNK_ACTION
                                       and message['payload']['index'] == message['payload']['total'] - 1))
                response.read()
                if lost_response and not lost and is_train_result:
                    lost = True
                    raise httpx.ReadTimeout('simulated response lost before delivery')
                received.append(response.content)
                yield response

        monkeypatch.setattr(http, 'stream', stream)
        payload = {'context': {'task_id': 'wire-task', 'round_id': 1},
                   'padding': b'x' * (1000 if request_chunked else 1)}
        result = rpc.call('client1', 'train', payload)
        assert result == expected
        assert len(calls) == 1
        assert (chunking.CHUNK_ACTION in actions) is request_chunked
        assert (chunking.DOWNLOAD_ACTION in actions) is response_chunked
        assert rpc.bytes_sent == sum(map(len, sent)) + sum(map(len, received))
        if lost_response:
            assert lost is True
            repeated = [frame for frame in sent if sent.count(frame) > 1]
            assert len(repeated) == 2, 'Exact retry sends the identical authenticated request twice.'
        before = rpc.bytes_sent
        telemetry_rpc = rpc_client(coordinator, http, tmp_path / 'telemetry-spool')
        rows = telemetry_rpc.call('client1', 'telemetry_read', {'task_id': 'wire-task'})['events']
        assert [event['kind'] for event in rows] == ['epoch', 'epoch', 'training']
        summary = rows[-1]['payload']
        assert summary == {'round_id': 1, 'epochs': 2, 'metrics_eval_s': .02,
                           'metrics_callback_s': .003, 'training_s': .25}
        assert rpc.bytes_sent == before and telemetry_rpc.bytes_sent > 0
        rpc.close()


def prepared_role(tmp_path, monkeypatch, telemetry):
    create_cluster(tmp_path / 'keys', dict.fromkeys(('coordinator', 'client1'), '127.0.0.1'))
    worker = roles.RoleWorker('client1', tmp_path, Identity(tmp_path / 'keys', 'client1'), telemetry)
    rng = np.random.default_rng(17)
    x, y = rng.normal(0., .1, (24, 64)), np.arange(24, dtype=np.int64) % 10
    monkeypatch.setattr(datasets, 'load_dataset', lambda *a, **k: (x, y, x[:5], y[:5]))
    policy = {'mode': 'plain', 'scale': 128, 'bits': 8, 'dimension': 650,
              'members': ['client1', 'client2'], 'batch_size': 2, 'dataset': 'mnist'}
    worker.execute('prepare_data', {'task_id': 'summary-task', 'train_limit': 24, 'seed': 7,
                                   'non_iid': False, 'policy': policy})
    reference = [0] * 650
    context = {'task_id': 'summary-task', 'round_id': 1, 'key_epoch': 'summary-key',
               'model_hash': b.digest(reference), 'bits': 8, 'scale': 128, 'dimension': 650}
    args = {'context': context, 'reference': reference, 'mode': 'plain', 'local_epochs': 2,
            'seed': 13, 'backend': 'numpy', 'attack': 'none', 'key_messages': []}
    return worker, args


def test_real_role_publishes_summary_without_changing_local_result_or_retry(tmp_path, monkeypatch):
    telemetry = NodeTelemetry('client1', tmp_path)
    worker, args = prepared_role(tmp_path, monkeypatch, telemetry)
    result = worker.execute('train', args)
    assert len(result['training_metrics']) == 2
    events = telemetry.read({'task_id': 'summary-task'})['events']
    assert [event['kind'] for event in events] == ['epoch', 'epoch', 'training']
    assert events[-1]['payload'] == {'round_id': 1, 'epochs': 2,
        'metrics_eval_s': result['metrics_eval_s'], 'metrics_callback_s': result['metrics_callback_s'],
        'training_s': result['training_s']}
    assert worker.execute('train', args) == result
    assert telemetry.read({'task_id': 'summary-task'})['events'] == events
    telemetry.close()
    restarted = NodeTelemetry('client1', tmp_path)
    assert not restarted.record_training_summary('summary-task', 1, 2, result)
    assert restarted.read({'task_id': 'summary-task'})['events'] == events
    restarted.close()


def test_training_summary_failure_never_changes_business_result(tmp_path, monkeypatch, caplog):
    # Publishing is optional for direct role callers and fail-open for a
    # partially available telemetry sink. This follows the production callback.
    class FailedSummary:
        def record_epoch(self, *args):
            pass

        def record_training_summary(self, *args):
            raise OSError('private summary path')

    worker, args = prepared_role(tmp_path, monkeypatch, FailedSummary())
    result = worker.execute('train', args)
    assert 'plain_model' in result and len(result['training_metrics']) == 2
    assert 'Client training telemetry unavailable (OSError)' in caplog.text
    assert 'private summary path' not in caplog.text


@pytest.mark.parametrize('late_telemetry', [False, True])
@pytest.mark.parametrize('restart_exporter', [False, True])
def test_real_training_wire_journal_reaches_native_comparison_charts(
        tmp_path, monkeypatch, late_telemetry, restart_exporter):
    worker, args = prepared_role(tmp_path, monkeypatch, None)

    def factory(_node_id, _runtime, _identity, telemetry):
        worker.telemetry = telemetry
        return worker

    monkeypatch.setattr(roles, 'RoleWorker', factory)
    coordinator = Identity(tmp_path / 'keys', 'coordinator')
    snapshot = {'id': 'client1', 'role': 'client', 'machine': 'local', 'host': 'http://client1'}
    record = {'run_id': 'summary-task', 'config': {'local_epochs': 2},
              'initial_metrics': {'accuracy': .125, 'loss': 2.}, 'rounds': [],
              'evidence': {'node_snapshot': [snapshot]}}
    completed = {'round': 1, 'accuracy': .5, 'loss': 1.25, 'client_training_metrics': {}}
    logs = TensorBoardLogs(tmp_path)
    with TestClient(node.create_node_app('client1', tmp_path)) as http:
        business = rpc_client(coordinator, http, tmp_path / 'real-business-spool')
        telemetry = rpc_client(coordinator, http, tmp_path / 'real-telemetry-spool')
        try:
            trained = business.call('client1', 'train', args)
            assert len(trained['plain_model']) == 650
            assert not node.TRAIN_MONITOR_FIELDS.intersection(trained)
            business_bytes = business.bytes_sent
            reply = telemetry.call('client1', 'telemetry_read', {'task_id': record['run_id']})
            epochs = [row['payload'] for row in reply['events'] if row['kind'] == 'epoch']
            assert [row['epoch'] for row in epochs] == [1, 2]
            final = epochs[-1]
            monitor = RunNodeMonitor(tmp_path, record['run_id'], [snapshot], logs, local_epochs=2)
            if late_telemetry:
                record['rounds'].append(completed)
                assert logs.sync(record)['available']
            monitor._accept(snapshot, reply)
            assert logs.sync(record)['available']
            if not late_telemetry:
                observed = EventAccumulator(str(logs.logdir / record['run_id'] / 'nodes/client1')).Reload()
                assert 'training/accuracy' in observed.Tags()['scalars']
                assert 'evaluation/accuracy' not in observed.Tags()['scalars']
                record['rounds'].append(completed)
            if restart_exporter:
                logs.close_all()
                logs = TensorBoardLogs(tmp_path)
            original = deepcopy(record)
            assert logs.sync(record)['available']
            assert logs.sync(record)['available']
            assert record == original
            assert business.bytes_sent == business_bytes and telemetry.bytes_sent > 0

            board = EmbeddedTensorBoard(logs, reload_interval=0)
            app = FastAPI()
            app.mount('/tensorboard', board)
            try:
                with TestClient(app) as browser:
                    tags = browser.get('/tensorboard/data/plugin/timeseries/tags').json()['scalars']['runTagInfo']
                    node_run = next(run for run in tags if run.replace('\\', '/') == 'summary-task/nodes/client1')
                    chart_tags = ('evaluation/accuracy', 'evaluation/cross_entropy')
                    assert set(chart_tags) <= set(tags[node_run])
                    requests = [{'plugin': 'scalars', 'tag': tag, 'run': run}
                                for tag in chart_tags for run in ('summary-task', node_run)]
                    response = browser.get('/tensorboard/data/plugin/timeseries/timeSeries',
                                           params={'requests': json.dumps(requests)})
                    assert response.status_code == 200
                    actual = [[(point['step'], point['value']) for point in row['runToSeries'][request['run']]]
                              for request, row in zip(requests, response.json(), strict=True)]
                    assert actual == [[(0, .125), (1, .5)], [(1, pytest.approx(final['accuracy']))],
                                      [(0, 2.), (1, 1.25)], [(1, pytest.approx(final['loss']))]]
            finally:
                board.close()
        finally:
            business.close()
            telemetry.close()
            logs.close_all()
