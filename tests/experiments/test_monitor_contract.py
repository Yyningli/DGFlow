"""Runner observability failures do not change successful business execution."""
import json
import threading
from copy import deepcopy

import numpy as np
import pytest

from dgfl.experiments import runner
from dgfl.experiments.monitoring import RunNodeMonitor

TIME='2026-10-10T01:00:00+00:00'


class LocalMonitor:
    def __init__(self,*args): self.finished=False
    def start(self): pass
    def set_context(self,*args): pass
    def snapshot(self): return {'hardware':{'samples':[]}}
    def finish(self):
        self.finished=True
        return self.snapshot()


@pytest.fixture
def experiment(tmp_path,monkeypatch):
    runtime=tmp_path/'runtime'; runtime.mkdir()
    clients=[f'client{i}' for i in range(1,7)]
    nodes=[*clients,*runner.AUTHORITIES,'aggregator1','aggregator2','aggregator3']
    (runtime/'cluster.json').write_text(json.dumps({'deployment':'single_host','nodes':{
        node:{'url':f'https://127.0.0.1:{9000+index}'} for index,node in enumerate(nodes)}}),encoding='utf8')
    monkeypatch.setattr(runner,'implementation_evidence',lambda:{})
    monkeypatch.setattr(runner,'LocalProcessMonitor',LocalMonitor)
    monkeypatch.setattr(runner,'load_mnist',lambda *a,**k:(None,None,np.zeros((3,64)),np.arange(3)))
    monkeypatch.setattr(runner,'initial_model',lambda seed,features=64:np.zeros(features*10+10))
    record={'run_id':'monitor-contract','status':'queued','current_round':0,'rounds':[],
            'events':[],'summary':{},'error':None,'evidence':{},
            'config':{'mode':'plain','rounds':2,'seed':42,'attack':'none','malicious_clients':0,
                      'non_iid':False,'offline_aggregators':0,'train_limit':6,'test_limit':3,
                      'local_epochs':2,'backend':'numpy','grid':8,'execution':'serial'}}
    manager=runner.RunManager(runtime)
    manager.records[record['run_id']]=record; manager.active=record['run_id']
    return manager,record,nodes


class BusinessRPC:
    def __init__(self,shared,*,fallback=True,telemetry=False):
        self.shared=shared; self.fallback=fallback; self.telemetry=telemetry
        self.bytes_sent=17; self.closed=False; self.calls=[]

    def call(self,node,action,payload=None,**kwargs):
        if action=='health':
            return {'node_id':node,'capabilities':{'node_telemetry':self.telemetry}}
        if action=='prepare_data':
            if self.telemetry: assert self.shared['ready'].wait(5)
            return {'samples':1,'partition_hash':'test-partition'}
        if action=='train':
            round_id=payload['context']['round_id']; epochs=payload['local_epochs']
            self.calls.append((node,round_id))
            rows=[{'time':TIME,'epoch':epoch,'round_id':round_id,'step':(round_id-1)*epochs+epoch,
                   'accuracy':.75,'loss':.5,'samples':1,'elapsed_s':.02,'metrics_eval_s':.01}
                  for epoch in range(1,epochs+1)]
            result={'plain_model':payload['reference'],'training_s':.04,'encrypt_s':0.,'proof_s':0.}
            if self.fallback:
                result.update(training_metrics=rows,metrics_eval_s=.02,metrics_callback_s=.03)
            with self.shared['lock']:
                journal=self.shared['events'].setdefault(node,[])
                for row in rows:
                    journal.append({'sequence':len(journal)+1,'kind':'epoch','time':TIME,'payload':row})
                journal.append({'sequence':len(journal)+1,'kind':'training','time':TIME,'payload':{
                    'round_id':round_id,'epochs':epochs,'metrics_eval_s':.02,
                    'metrics_callback_s':.03,'training_s':.04}})
            return result
        raise AssertionError(f'unexpected business RPC: {action}')

    def close(self): self.closed=True


class TelemetryRPC:
    def __init__(self,shared,count):
        self.shared=shared; self.count=count; self.bytes_sent=0; self.closed=False

    def call(self,node,action,payload,**kwargs):
        with self.shared['lock']:
            self.bytes_sent+=10
            if action=='telemetry_start':
                self.shared['active'].add(node)
                if len(self.shared['active'])==self.count: self.shared['ready'].set()
                return {'active':True}
            if action=='telemetry_stop':
                self.shared['active'].discard(node)
                return {'active':False}
            assert action=='telemetry_read'
            rows=[row for row in self.shared['events'].get(node,[]) if row['sequence']>payload['after']]
            rows=deepcopy(rows[:payload['limit']])
            return {'node_id':node,'task_id':payload['task_id'],'active':node in self.shared['active'],
                    'events':rows,'more':False,'next_cursor':rows[-1]['sequence'] if rows else payload['after']}

    def close(self): self.closed=True


def shared_state():
    return {'lock':threading.Lock(),'ready':threading.Event(),'events':{},'active':set()}


def assert_complete(manager,record,business):
    assert record['status']=='completed',record.get('error')
    assert record['error'] is None
    assert record['summary']['completed_rounds']==2
    assert record['summary']['bytes_sent']==17
    assert business.closed and len(business.calls)==len(set(business.calls))==12
    assert [event['round'] for event in record['events'] if event['stage']=='completed_round']==[1,2]
    assert manager.active is None
    assert record['run_id'] not in manager._monitors and record['run_id'] not in manager._node_monitors
    saved=json.loads((manager.runtime/'results'/record['run_id']/'result.json').read_text('utf8'))
    assert saved==manager.snapshot(record['run_id'])


@pytest.mark.parametrize('fallback',[True,False])
def test_two_round_runner_harvests_epochs_and_overhead_without_event_shadow_or_protocol_pollution(
        experiment,monkeypatch,fallback):
    manager,record,nodes=experiment; shared=shared_state()
    business=BusinessRPC(shared,fallback=fallback,telemetry=True)
    telemetry=TelemetryRPC(shared,len(nodes))
    monkeypatch.setattr(runner,'RPCClient',lambda *a:business)
    monkeypatch.setattr(runner,'RunNodeMonitor',lambda *a,**k:RunNodeMonitor(
        *a,**k,rpc_factory=lambda *args:telemetry))
    manager._run(record)
    assert_complete(manager,record,business)
    monitoring=record['evidence']['node_monitoring']
    assert telemetry.closed and not shared['active']
    assert monitoring['summary']['cleanup_complete'] is True
    assert monitoring['summary']['bytes_sent']>0
    assert monitoring['summary']['metrics_eval_s']==pytest.approx(.24)
    for row in record['rounds']:
        assert row['metrics_eval_s']==pytest.approx(.12)
        totals=row['monitoring_overhead']
        assert totals['clients_recorded']==6
        assert totals['metrics_eval_s']==pytest.approx(.12)
        assert totals['metrics_callback_s']==pytest.approx(.18)
        assert totals['training_s']==pytest.approx(.24)
        if fallback:
            assert len(row['client_training_metrics'])==6
            assert all(len(epochs)==2 for epochs in row['client_training_metrics'].values())
    assert all(node['training']['step']==4 for node in monitoring['nodes'] if node['role']=='client')


@pytest.mark.parametrize('stage',['construct','start','finish','context','sample_callback'])
def test_local_monitor_failure_remains_diagnostic_through_real_runner_path(experiment,monkeypatch,stage):
    manager,record,_=experiment
    business=BusinessRPC(shared_state())
    monkeypatch.setattr(runner,'RPCClient',lambda *a:business)
    class BrokenMonitor(LocalMonitor):
        def __init__(self,*args):
            if stage=='construct': raise OSError('sensor construct unavailable')
            super().__init__(*args)
        def start(self):
            if stage=='start': raise OSError('sensor start unavailable')
            if stage=='sample_callback': self.on_sample({'sample_index':0})
        def set_context(self,*args):
            if stage=='context': raise OSError('sensor context unavailable')
        def finish(self):
            if stage=='finish': raise OSError('sensor finish unavailable')
            return super().finish()
    monkeypatch.setattr(runner,'LocalProcessMonitor',BrokenMonitor)
    if stage=='sample_callback':
        monkeypatch.setattr(manager.tensorboard,'write_sample',lambda *a:(_ for _ in ()).throw(OSError('log unavailable')))
    manager._run(record)
    assert_complete(manager,record,business)
    diagnostic=record['evidence']['monitoring']
    assert diagnostic['available'] is False
    assert diagnostic['stage']==('sample_log' if stage=='sample_callback' else stage)
    assert all(len(row['client_training_metrics'])==6 for row in record['rounds'])


@pytest.mark.parametrize('status',['running','completed'])
def test_lightweight_snapshot_omits_only_detached_history_and_keeps_full_export(experiment,status):
    manager,record,_=experiment
    resources={'peak_sampled_rss_bytes':12,'hardware':{
        'metadata':{'cpu':{'source':'measured'}},'summary':{'samples':1},
        'samples':[{'sample_index':0,'cpu':{'system_utilization_percent':25.}}]}}
    record['status']=status
    record['evidence'].update(resources=deepcopy(resources),memory=deepcopy(resources))
    class CachedMonitor:
        def snapshot(self): return resources
    manager._monitors[record['run_id']]=CachedMonitor()
    before=deepcopy(record)
    brief=manager.snapshot(record['run_id'],include_history=False)
    for field in ('resources','memory'):
        hardware=brief['evidence'][field]['hardware']
        assert 'samples' not in hardware and hardware['history_omitted'] is True
        assert hardware['summary']==resources['hardware']['summary']
        assert hardware['metadata']==resources['hardware']['metadata']
    brief['evidence']['resources']['hardware']['metadata'].clear()
    assert record==before
    full=manager.snapshot(record['run_id'])
    assert full['evidence']['resources']['hardware']['samples']==resources['hardware']['samples']
    assert 'history_omitted' not in full['evidence']['resources']['hardware']
