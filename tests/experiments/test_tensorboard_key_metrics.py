"""Dashboard cardinality stays bounded without discarding diagnostic evidence."""
from copy import deepcopy

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from dgfl.experiments.tensorboard import (
    KEY_SCALAR_TAGS,
    TensorBoardLogs,
    _actor_metrics,
    is_key_scalar,
    metric_metadata,
)


def verification_row(clients=2):
    return {'round': 1, 'proof_verification_metrics': {'authority1': {'verifications': {
        f'client{index}': {'proof_verify_s': .25, 'inner_product_s': .125,
                         'parameters_load_s': .01, 'native_threads': 2}
        for index in range(1, clients + 1)
    }}}}


def test_dashboard_has_forty_exact_key_tags_and_six_gpu_charts():
    assert len(KEY_SCALAR_TAGS) == 40
    gpu = {tag for tag in KEY_SCALAR_TAGS if tag.startswith('gpu_compute/')}
    assert gpu == {f'gpu_compute/totals/{field}_ms' for field in (
        'host_wall', 'kernel', 'upload', 'download', 'sync', 'scheduler_wait')}
    assert all(is_key_scalar(tag) for tag in KEY_SCALAR_TAGS)
    assert not is_key_scalar('gpu_compute/kernels/dgfl_gt_validate/kernel_ms')
    assert not is_key_scalar('proof/verification/client1/proof_verify_s')
    assert not is_key_scalar('protocol/stages/aggregation_s')
    assert not is_key_scalar('system/gpu_memory_total_bytes')
    assert not is_key_scalar('training/samples')
    assert not is_key_scalar('evaluation/accuracy/debug')


@pytest.mark.parametrize('clients', [1, 6, 100])
def test_verification_chart_count_does_not_grow_with_client_count(clients):
    row = verification_row(clients)
    original = deepcopy(row)
    metrics = _actor_metrics(row)[('nodes', 'authority1')]
    visible = {tag: value for tag, value in metrics.items() if is_key_scalar(tag)}
    assert visible == {'proof/verification/proof_verify_total_s': clients * .25,
                       'proof/verification/inner_product_total_s': clients * .125}
    assert metrics['proof/verification/client1/proof_verify_s'] == .25
    assert row == original


@pytest.mark.parametrize('bad', [None, -1, float('nan'), float('inf'), True, '0.1'])
def test_incomplete_verification_timing_is_not_exported_as_a_partial_sum(bad):
    row = verification_row()
    row['proof_verification_metrics']['authority1']['verifications']['client2']['proof_verify_s'] = bad
    metrics = _actor_metrics(row)[('nodes', 'authority1')]
    assert 'proof/verification/proof_verify_total_s' not in metrics
    assert metrics['proof/verification/inner_product_total_s'] == .25


def test_empty_or_malformed_verification_records_do_not_create_fake_zero_totals():
    row = verification_row(0)
    assert not _actor_metrics(row)
    row = verification_row()
    row['proof_verification_metrics']['authority1']['verifications']['client2'] = None
    metrics = _actor_metrics(row)[('nodes', 'authority1')]
    assert not any(tag.endswith('_total_s') for tag in metrics)


def test_real_zero_and_overflow_are_distinguished():
    row = verification_row()
    for timings in row['proof_verification_metrics']['authority1']['verifications'].values():
        timings['proof_verify_s'] = 0.
        timings['inner_product_s'] = 1e308
    metrics = _actor_metrics(row)[('nodes', 'authority1')]
    assert metrics['proof/verification/proof_verify_total_s'] == 0.
    assert 'proof/verification/inner_product_total_s' not in metrics


def test_historical_sync_adds_shared_totals_idempotently_and_keeps_client_diagnostics(tmp_path):
    logs = TensorBoardLogs(tmp_path)
    row = verification_row()
    record = {'run_id': 'historical', 'created_at': '2026-10-10T01:00:00+00:00', 'rounds': [row]}
    try:
        assert logs.sync(record)['available']
        assert logs.sync(record)['available']
        events = EventAccumulator(str(logs.logdir / 'historical/nodes/authority1'),
                                  size_guidance={'scalars': 0}).Reload()
        total = events.Scalars('proof/verification/proof_verify_total_s')
        assert [(point.step, point.value) for point in total] == [(1, .5)]
        assert events.Scalars('proof/verification/client2/proof_verify_s')[0].value == .25
        label, description = metric_metadata('proof/verification/proof_verify_total_s')
        assert '之和' in label and '不代表验证阶段墙钟' in description
    finally:
        logs.close_all()
