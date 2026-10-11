"""Real local partitions and retry-safe, non-protocol training diagnostics."""
from copy import deepcopy
from datetime import datetime

import numpy as np
import pytest

from dgfl.crypto import backend as b
from dgfl.services.roles import RoleWorker
from dgfl.training import datasets, model
from dgfl.transport.security import Identity, create_cluster
from dgfl.validation.policy import quantize


def prepared_client(tmp_path,monkeypatch,telemetry=None,features=64):
    create_cluster(tmp_path/'keys',{'coordinator':'127.0.0.1','client1':'127.0.0.1'})
    worker=RoleWorker('client1',tmp_path,Identity(tmp_path/'keys','client1'),telemetry=telemetry)
    rng=np.random.default_rng(11)
    x=rng.normal(0.,.15,(32,features)); y=np.arange(32,dtype=np.int64)%10
    monkeypatch.setattr(datasets,'load_dataset',lambda *a,**k:(x,y,x[:10],y[:10]))
    dimension=model.parameter_count(features)
    policy={'mode':'plain','scale':128,'bits':8,'dimension':dimension,
            'members':['client1','client2'],'batch_size':2,
            'dataset':'mnist' if features==64 else 'cifar10'}
    worker.execute('prepare_data',{'task_id':'epoch-metrics','train_limit':32,'seed':7,
                                   'non_iid':True,'policy':policy})
    reference=[0]*dimension
    context={'task_id':'epoch-metrics','round_id':3,'key_epoch':'epoch-key-3',
             'model_hash':b.digest(reference),'bits':8,'scale':128,'dimension':dimension,
             'dataset':policy['dataset']}
    train={'context':context,'reference':reference,'mode':'plain','local_epochs':2,
           'seed':13,'backend':'numpy','attack':'none','key_messages':[]}
    with np.load(worker.folder/'data/epoch-metrics.npz',allow_pickle=False) as data:
        local_x,local_y=data['x'].copy(),data['y'].copy()
    return worker,train,local_x,local_y


@pytest.mark.parametrize('features',[64,192])
@pytest.mark.parametrize('attack',['none','label_flip','sign_flip','random'])
def test_role_records_partition_metrics_before_attack_and_quantization(tmp_path,monkeypatch,features,attack):
    events=[]
    class Telemetry:
        def record_epoch(self,*args): events.append(deepcopy(args))
    worker,args,x,y=prepared_client(tmp_path,monkeypatch,Telemetry(),features)
    args['attack']=attack
    labels=(y+1)%10 if attack=='label_flip' else y
    expected=model.train_local(np.zeros(model.parameter_count(features)),x,labels,features=features,
                               epochs=2,learning_rate=.3,batch_size=64,seed=13)
    result=worker.execute('train',args)
    metrics=model.evaluate(expected,x,labels,features=features)
    assert len(result['training_metrics'])==2
    assert [row['step'] for row in result['training_metrics']]==[5,6]
    assert [row['epoch'] for row in result['training_metrics']]==[1,2]
    assert all(row['round_id']==3 and datetime.fromisoformat(row['time']).utcoffset().total_seconds()==0
               for row in result['training_metrics'])
    assert {key:result['training_metrics'][-1][key] for key in metrics}==metrics
    assert result['metrics_eval_s']==sum(row['metrics_eval_s'] for row in result['training_metrics'])
    assert result['metrics_callback_s']>=0 and result['training_s']>=0
    sent=model.attack_weights(expected,attack,13,features=features) if attack in ('sign_flip','random') else expected
    assert result['plain_model']==quantize(sent,128,8)
    assert len(events)==2
    assert all(task=='epoch-metrics' and round_id==3 and epochs==2 for task,round_id,epochs,_ in events)
    assert [event[-1] for event in events]==result['training_metrics']
    assert worker.execute('train',deepcopy(args))==result
    assert len(events)==2,'Business-round retries must not train or publish twice'


def test_unavailable_telemetry_keeps_reply_metrics_and_training(tmp_path,monkeypatch,caplog):
    class Unavailable:
        def record_epoch(self,*args): raise OSError('private monitoring path')
    worker,args,x,y=prepared_client(tmp_path,monkeypatch,Unavailable())
    result=worker.execute('train',args)
    expected=model.train_local(np.zeros(650),x,y,epochs=2,learning_rate=.3,batch_size=64,seed=13)
    assert result['plain_model']==quantize(expected,128,8)
    assert len(result['training_metrics'])==2
    assert caplog.text.count('Client epoch telemetry unavailable')==2
    assert 'private monitoring path' not in caplog.text


def test_training_s_excludes_observational_evaluation_and_publish(tmp_path,monkeypatch):
    clock=[0.]
    class SlowTelemetry:
        def record_epoch(self,*args): clock[0]+=3.
    worker,args,_,_=prepared_client(tmp_path,monkeypatch,SlowTelemetry())
    monkeypatch.setattr(model.time,'perf_counter',lambda:clock[0])
    evaluate=model.evaluate
    def slow_evaluate(*args,**kwargs):
        metrics=evaluate(*args,**kwargs); clock[0]+=2.
        return metrics
    monkeypatch.setattr(model,'evaluate',slow_evaluate)
    result=worker.execute('train',args)
    assert result['metrics_eval_s']==4.
    assert result['metrics_callback_s']==6.
    assert result['training_s']==0.


def test_zero_epochs_returns_empty_diagnostics(tmp_path,monkeypatch):
    worker,args,_,_=prepared_client(tmp_path,monkeypatch)
    args['local_epochs']=0
    result=worker.execute('train',args)
    assert result['training_metrics']==[]
    assert result['metrics_eval_s']==result['metrics_callback_s']==0.
