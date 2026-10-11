"""Epoch diagnostics observe real SGD without changing optimizer arithmetic."""
import logging

import numpy as np
import pytest

from dgfl.training import model


def local_data(features=64):
    rng=np.random.default_rng(19)
    return rng.normal(0.,.2,(23,features)),rng.integers(0,10,23)


@pytest.mark.parametrize('backend',['numpy','torch'])
@pytest.mark.parametrize('features',[64,192])
def test_epoch_callbacks_preserve_exact_weights_and_report_actual_partition(backend,features):
    if backend=='torch': pytest.importorskip('torch')
    x,y=local_data(features); initial=model.initial_model(7,features)
    options={'features':features,'epochs':3,'learning_rate':.3,'batch_size':7,'seed':31,'backend':backend}
    expected=model.train_local(initial,x,y,**options)
    rows=[]
    actual=model.train_local(initial,x,y,on_epoch=rows.append,**options)
    np.testing.assert_array_equal(actual,expected)
    assert len(rows)==3
    for epoch,row in enumerate(rows,1):
        trained=model.train_local(initial,x,y,**{**options,'epochs':epoch})
        metrics=model.evaluate(trained,x,y,features=features)
        assert {name:row[name] for name in metrics}==metrics
        assert row['epoch']==epoch
        assert row['elapsed_s']>=0 and row['metrics_eval_s']>=0
        assert all(type(value) in (int,float) for value in row.values())


@pytest.mark.parametrize('backend',['numpy','torch'])
def test_callback_failure_does_not_repeat_or_change_training(backend,caplog):
    if backend=='torch': pytest.importorskip('torch')
    x,y=local_data(); initial=model.initial_model(5)
    expected=model.train_local(initial,x,y,epochs=3,backend=backend)
    epochs=[]
    def unavailable(row):
        epochs.append(row['epoch'])
        raise RuntimeError('sensitive internal monitoring details')
    with caplog.at_level(logging.WARNING):
        actual=model.train_local(initial,x,y,epochs=3,backend=backend,on_epoch=unavailable)
    np.testing.assert_array_equal(actual,expected)
    assert epochs==[1,2,3]
    assert caplog.text.count('Local epoch monitoring callback unavailable')==3
    assert 'sensitive internal monitoring details' not in caplog.text


@pytest.mark.parametrize('backend',['numpy','torch'])
def test_evaluation_failure_is_observational_and_keeps_overhead_recorded(backend,monkeypatch,caplog):
    if backend=='torch': pytest.importorskip('torch')
    x,y=local_data(); initial=model.initial_model()
    expected=model.train_local(initial,x,y,epochs=2,backend=backend)
    def unavailable(*args,**kwargs): raise RuntimeError('private evaluation detail')
    monkeypatch.setattr(model,'evaluate',unavailable)
    rows=[]
    with caplog.at_level(logging.WARNING):
        actual=model.train_local(initial,x,y,epochs=2,backend=backend,on_epoch=rows.append)
    np.testing.assert_array_equal(actual,expected)
    assert [row['epoch'] for row in rows]==[1,2]
    assert all(row['metrics_eval_s']>=0 and row['samples']==len(y) for row in rows)
    assert all('accuracy' not in row and 'loss' not in row and row['metrics_error'] for row in rows)
    assert 'private evaluation detail' not in caplog.text


def test_default_and_zero_epoch_paths_do_not_evaluate(monkeypatch):
    x,y=local_data(); initial=model.initial_model()
    monkeypatch.setattr(model,'evaluate',lambda *a,**k:pytest.fail('unexpected epoch evaluation'))
    model.train_local(initial,x,y,epochs=2)
    rows=[]
    np.testing.assert_array_equal(model.train_local(initial,x,y,epochs=0,on_epoch=rows.append),initial)
    assert rows==[]


def test_metric_error_boundary_does_not_swallow_optimizer_errors(monkeypatch):
    x,y=local_data()
    def broken(*args,**kwargs): raise ArithmeticError('optimizer failed')
    monkeypatch.setattr(model.np,'exp',broken)
    with pytest.raises(ArithmeticError,match='optimizer failed'):
        model.train_local(model.initial_model(),x,y,on_epoch=lambda row:None)


def test_epoch_elapsed_excludes_evaluation_and_callback_time(monkeypatch):
    x,y=local_data(); clock=[0.]
    monkeypatch.setattr(model.time,'perf_counter',lambda:clock[0])
    evaluate=model.evaluate
    def slow_evaluate(*args,**kwargs):
        result=evaluate(*args,**kwargs)
        clock[0]+=2.
        return result
    monkeypatch.setattr(model,'evaluate',slow_evaluate)
    rows=[]
    def slow_callback(row):
        rows.append(row)
        clock[0]+=3.
    model.train_local(model.initial_model(),x,y,epochs=2,on_epoch=slow_callback)
    assert [row['elapsed_s'] for row in rows]==[0.,0.]
    assert [row['metrics_eval_s'] for row in rows]==[2.,2.]
