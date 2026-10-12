"""Historical replay acquires one export lock per scope, with safe retries."""
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from dgfl.experiments.tensorboard import TensorBoardLogs

TIME='2026-10-10T01:00:00+00:00'


def sample(index, *, process=True):
    row={'sample_index':index,'time':TIME,'elapsed_s':index*5.,
         'cpu':{'system_utilization_percent':25.+index},
         'system':{'memory_used_bytes':1000+index}}
    if process: row['processes']={'rss_bytes':100+index,'cpu_percent':None}
    return row


def history(rounds=3, epochs=2):
    rows=[]
    for round_id in range(1,rounds+1):
        clients={}
        for node_id in ('client1','client2'):
            clients[node_id]=[{'time':TIME,'epoch':epoch,'round_id':round_id,
                              'step':(round_id-1)*epochs+epoch,'accuracy':.75,'loss':.5,
                              'samples':12,'elapsed_s':.02,'metrics_eval_s':.003}
                             for epoch in range(1,epochs+1)]
        rows.append({'round':round_id,'accuracy':.75,'loss':.5,'client_training_metrics':clients})
    return {'run_id':'batched','created_at':TIME,'initial_metrics':{'accuracy':.25,'loss':2.},'rounds':rows,
            'evidence':{'node_snapshot':[{'id':'client1','machine':'local'},
                                        {'id':'client2','machine':'local'},
                                        {'id':'authority1','machine':'B'}],
                        'monitor_local_machine':'local',
                        'resources':{'hardware':{'samples':[sample(index) for index in range(4)]}}}}


def journal(runtime, node_id='authority1', count=3, *, summary=False):
    directory=runtime/'monitoring'/'batched'; directory.mkdir(parents=True,exist_ok=True)
    path=directory/f'{node_id}.segment.jsonl'
    rows=[]
    for index in range(count):
        event={'sequence':index+1,'kind':'resource','time':TIME,'payload':sample(index)}
        rows.append({'node_id':node_id,'event':event})
    if summary:
        rows.append({'node_id':node_id,'event':{'sequence':count+1,'kind':'training','time':TIME,
                    'payload':{'round_id':1,'epochs':2,'metrics_eval_s':.2,'metrics_callback_s':.1,'training_s':1.}}})
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf8')
    return path


def scalars(path):
    reader=EventAccumulator(str(path),size_guidance={'scalars':0},purge_orphaned_data=False).Reload()
    return {tag:reader.Scalars(tag) for tag in reader.Tags()['scalars']}


def test_each_sync_batches_all_epochs_and_samples_once_per_node_or_machine(tmp_path,monkeypatch):
    logs=TensorBoardLogs(tmp_path); record=history(rounds=15,epochs=4)
    journal(tmp_path,summary=True)
    calls=[]; original=logs._write
    def write(run_id,batches,*,scope=None):
        calls.append((scope,len(batches)))
        return original(run_id,batches,scope=scope)
    monkeypatch.setattr(logs,'_write',write)
    assert logs.sync(record)['available']
    assert Counter(scope for scope,_ in calls)==Counter({None:1,('nodes','client1'):1,('nodes','client2'):1,
                                                        ('nodes','authority1'):1,('machines','B'):1,
                                                        ('machines','local'):1})
    assert dict(calls)[('nodes','client1')]==60
    assert dict(calls)[('machines','local')]==4
    root=logs.logdir/'batched'
    assert [point.step for point in scalars(root/'nodes/client1')['training/accuracy']]==list(range(1,61))
    assert [point.step for point in scalars(root/'machines/B')['system/cpu_percent']]==[0,1,2]
    assert [point.step for point in scalars(root/'machines/local')['system/cpu_percent']]==[0,1,2,3]
    assert 'processes/rss_bytes' not in scalars(root/'machines/local')
    assert [point.step for point in scalars(root)['evaluation/accuracy']]==list(range(16))
    calls.clear()
    assert logs.sync(record)['available']
    assert all(count==1 for count in Counter(scope for scope,_ in calls).values())
    assert len(scalars(root/'nodes/client1')['training/accuracy'])==60


@pytest.mark.parametrize('failed_scope',[('nodes','authority1'),('machines','B')])
def test_partial_scope_failure_retains_journal_cursor_and_recovers_without_duplicates(tmp_path,monkeypatch,failed_scope):
    logs=TensorBoardLogs(tmp_path); record=history()
    path=journal(tmp_path)
    original=logs._write; failed=[]
    def write(run_id,batches,*,scope=None):
        if scope==failed_scope and not failed:
            failed.append(True)
            assert original(run_id,batches[:1],scope=scope)['available']
            return {'available':False,'reason':'transient failure after a persisted prefix'}
        return original(run_id,batches,scope=scope)
    monkeypatch.setattr(logs,'_write',write)
    assert logs.sync(record)['available'] is False
    assert path.name not in logs._replay_offsets['batched']
    assert logs.sync(record)['available'] is True
    assert logs._replay_offsets['batched'][path.name]==path.stat().st_size
    root=logs.logdir/'batched'
    assert [point.step for point in scalars(root/'nodes/authority1')['processes/rss_bytes']]==[0,1,2]
    assert [point.step for point in scalars(root/'machines/B')['system/cpu_percent']]==[0,1,2]
    assert logs.sync(record)['available'] is True
    assert len(scalars(root/'machines/B')['system/cpu_percent'])==3


@pytest.mark.parametrize('failed_scope',[('nodes','client1'),('machines','local')])
def test_historical_epoch_and_local_sample_batches_retry_all_missing_points(tmp_path,monkeypatch,failed_scope):
    logs=TensorBoardLogs(tmp_path); record=history()
    original=logs._write; failed=[]
    def write(run_id,batches,*,scope=None):
        if scope==failed_scope and not failed:
            failed.append(True)
            assert original(run_id,batches[:1],scope=scope)['available']
            return {'available':False,'reason':'one interrupted batch'}
        return original(run_id,batches,scope=scope)
    monkeypatch.setattr(logs,'_write',write)
    assert logs.sync(record)['available'] is False
    assert logs.sync(record)['available'] is True
    root=logs.logdir/'batched'
    assert [point.step for point in scalars(root/'nodes/client1')['training/accuracy']]==[1,2,3,4,5,6]
    assert [point.step for point in scalars(root/'machines/local')['system/cpu_percent']]==[0,1,2,3]


def test_exporter_restart_and_concurrent_instances_preserve_all_batched_points(tmp_path):
    record=history(); journal(tmp_path,summary=True)
    first=TensorBoardLogs(tmp_path)
    assert first.sync(record)['available']
    first.close_all()
    instances=[TensorBoardLogs(tmp_path) for _ in range(3)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        replies=list(pool.map(lambda logs:logs.sync(record),instances))
    assert all(reply['available'] for reply in replies)
    root=first.logdir/'batched'
    assert len(scalars(root/'nodes/client1')['training/accuracy'])==6
    assert len(scalars(root/'nodes/client2')['training/cross_entropy'])==6
    assert len(scalars(root/'nodes/authority1')['processes/rss_bytes'])==3
    assert len(scalars(root/'machines/B')['system/cpu_percent'])==3
    assert len(scalars(root/'machines/local')['system/cpu_percent'])==4


def test_training_accounting_summary_uses_round_axis_without_breaking_live_or_replay(tmp_path):
    logs=TensorBoardLogs(tmp_path)
    path=journal(tmp_path,count=0,summary=True)
    event=json.loads(path.read_text())['event']
    assert logs.write_node_event('batched',{'id':'authority1','machine':'B'},event)['available']
    assert logs.sync(history())['available']
    assert logs._replay_offsets['batched'][path.name]==path.stat().st_size
    measured=scalars(logs.logdir/'batched/nodes/authority1')
    assert [point.step for point in measured['training_round/metrics_eval_s']]==[1]
    assert measured['training_round/metrics_eval_s'][0].value==pytest.approx(.2)
    assert 'training/metrics_eval_s' not in measured


@pytest.mark.parametrize('scope', ['nodes/authority1', 'machines/B'])
@pytest.mark.parametrize('change', ['delete', 'truncate'])
def test_consumed_journal_recovers_a_removed_or_truncated_scope(tmp_path, scope, change):
    logs=TensorBoardLogs(tmp_path); record=history()
    journal(tmp_path)
    assert logs.sync(record)['available']
    root=logs.logdir/'batched'
    for path in (root/scope).glob('events.out.tfevents.*'):
        if change=='delete': path.unlink()
        else: path.write_bytes(b'')
    assert logs.sync(record)['available']
    assert [(point.step,point.value) for point in scalars(root/'nodes/authority1')['processes/rss_bytes']]==[
        (0,100.),(1,101.),(2,102.)]
    assert [(point.step,point.value) for point in scalars(root/'machines/B')['system/cpu_percent']]==[
        (0,25.),(1,26.),(2,27.)]
    assert logs.sync(record)['available']
    assert len(scalars(root/'nodes/authority1')['processes/rss_bytes'])==3
    assert len(scalars(root/'machines/B')['system/cpu_percent'])==3


def test_consumed_journal_recovers_a_scope_after_writer_cache_eviction(tmp_path):
    logs=TensorBoardLogs(tmp_path,max_cached_runs=1); record=history()
    journal(tmp_path)
    assert logs.sync(record)['available']
    root=logs.logdir/'batched'
    for path in (root/'nodes/authority1').glob('events.out.tfevents.*'):
        path.unlink()
    assert logs.sync(record)['available']
    assert [point.step for point in scalars(root/'nodes/authority1')['processes/rss_bytes']]==[0,1,2]
    assert len(scalars(root/'machines/B')['system/cpu_percent'])==3
