"""Protocol and hardware scalar cards remain traceable to recorded evidence."""
from copy import deepcopy
from datetime import datetime

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from dgfl.experiments.tensorboard import TensorBoardLogs, metric_metadata, scalar_label_catalog

TIME='2026-10-10T01:00:00+00:00'
FINISHED='2026-10-10T01:01:00+00:00'


def gpu_profile(kernel_s=.3):
    return {'totals':{'batches':2,'rows':650,'allocations':3,'allocation_reuses':7,'frees':1,
                      'upload_bytes':100,'download_bytes':50,'kernel_seconds':kernel_s,
                      'host_wall_seconds':.9,'sync_seconds':.5,'scheduler_wait_seconds':.1},
            'by_kernel':{'dgfl_gt_validate':{'batches':1,'rows':650,'kernel_seconds':kernel_s},
                         'not_a_real_kernel':{'kernel_seconds':200.}},
            'pool':{'reserved_bytes':128,'peak_reserved_bytes':256,'budget_bytes':1024,'slots':3},
            'configuration':{'block_size':32,'chunk_size':1024,'scheduler_slots':1},
            'kernel_resources':{'dgfl_gt_validate':{'num_registers':255,'static_shared_bytes':0}},
            'device_resources':{'multiprocessor_count':24,'max_threads_per_multiprocessor':1536}}


def record():
    rounds=[]
    for round_id in (1,2):
        rounds.append({'round':round_id,'accuracy':.75,'loss':.5,'duration_s':10.+round_id,
                       'controller_cpu_s':1.5,'bytes_sent':round_id*100,
                       'accepted_clients':['client1','client2'],'rejected_clients':['client3'],
                       'collateral_clients':[],'client_failures':{},'attack_accepted':0,'attack_submitted':1,
                       'honest_rejected':1,'honest_total':2,'reference_norm_squared':55,
                       'stage_times':{'dkg_s':.1,'client_stage_s':3.,'training_wall_sum_s':6.,
                                      'aggregation_s':8.,'combine_and_confirmation_s':4.,
                                      'combine_cpu_s_max':2.5,'unknown_private_timing':999.},
                       'monitoring_overhead':{'clients_recorded':2,'metrics_eval_s':.12,
                                              'metrics_callback_s':.05,'training_s':.3},
                       'proof_generation_metrics':{'client1':{'proof_gen_s':.2,'parameters_load_s':.02,
                                                             'private_vector':[1,2,3]}},
                       'proof_verification_metrics':{'authority1':{'verifications':{
                           'client1':{'proof_verify_s':.12,'parameters_load_s':.01,'inner_product_s':.03,
                                      'native_threads':2,'private_timing':99.},
                           '../outside':{'proof_verify_s':1.}}}},
                       'combine_metrics':{'coordinator':{'combine_timings':{
                           'combine_total_s':.6,'combine_cpu_s':.4,'combine_commitment_cache_hits':3,
                           'gpu_profile':gpu_profile()}},
                           'authority1':{'combine_timings':{'combine_total_s':.7,'combine_cpu_s':.5,
                               'combine_commitment_cache_misses':0,'gpu_profile':gpu_profile(.2)}}}})
    return {'run_id':'protocol-cards','status':'completed','created_at':TIME,'rounds':rounds,
            'initial_metrics':{'accuracy':.25,'loss':2.},
            'summary':{'elapsed_s':30.,'controller_cpu_s':4.,'bytes_sent':999,'completed_rounds':2},
            'events':[{'stage':'finished','time':FINISHED}],
            'evidence':{'node_snapshot':[{'id':'client1','machine':'local'},
                                        {'id':'authority1','machine':'local'}],
                        'compute':{'coordinator_gpu_profile':gpu_profile(.88)}}}


def reader(path):
    return EventAccumulator(str(path),size_guidance={'scalars':0},purge_orphaned_data=False).Reload()


def values(events,tag):
    return [(point.step,point.value) for point in events.Scalars(tag)]


def test_round_protocol_counts_communication_and_measured_wall_times_are_exported(tmp_path):
    logs=TensorBoardLogs(tmp_path); data=record()
    assert logs.sync(data)['available']
    events=reader(logs.logdir/data['run_id'])
    assert values(events,'evaluation/accuracy')==[(0,.25),(1,.75),(2,.75)]
    assert values(events,'round/duration_s')==[(1,11.),(2,12.)]
    assert values(events,'round/controller_cpu_s')==[(1,1.5),(2,1.5)]
    assert values(events,'communication/round_bytes')==[(1,100.),(2,200.)]
    assert values(events,'communication/cumulative_round_bytes')==[(1,100.),(2,300.)]
    assert values(events,'communication/task_total_bytes')==[(2,999.)]
    assert values(events,'task/duration_s')==[(2,30.)]
    assert values(events,'clients/accepted')==[(1,2.),(2,2.)]
    assert values(events,'clients/rejected')==[(1,1.),(2,1.)]
    assert values(events,'clients/collateral')==[(1,0.),(2,0.)]
    assert values(events,'clients/attack_accepted')==[(1,0.),(2,0.)]
    assert values(events,'clients/honest_rejected')==[(1,1.),(2,1.)]
    assert values(events,'model/reference_norm_squared')==[(1,55.),(2,55.)]
    assert values(events,'protocol/stages/client_stage_s')==[(1,3.),(2,3.)]
    assert values(events,'protocol/stages/training_wall_sum_s')==[(1,6.),(2,6.)]
    assert values(events,'protocol/stages/aggregation_s')==[(1,8.),(2,8.)]
    assert values(events,'protocol/stages/combine_and_confirmation_s')==[(1,4.),(2,4.)]
    assert events.Scalars('monitoring/metrics_callback_s')[0].value==pytest.approx(.05)
    assert events.Scalars('task/duration_s')[0].wall_time==datetime.fromisoformat(FINISHED).timestamp()
    assert all('unknown_private' not in tag for tag in events.Tags()['scalars'])


def test_proof_combine_and_gpu_evidence_is_scoped_to_its_actual_process(tmp_path):
    logs=TensorBoardLogs(tmp_path); data=record()
    assert logs.sync(data)['available']
    root=reader(logs.logdir/data['run_id'])
    client=reader(logs.logdir/data['run_id']/'nodes/client1')
    edge=reader(logs.logdir/data['run_id']/'nodes/authority1')
    assert client.Scalars('proof/generation/proof_gen_s')[0].value==pytest.approx(.2)
    assert edge.Scalars('proof/verification/client1/inner_product_s')[0].value==pytest.approx(.03)
    assert values(edge,'proof/verification/client1/native_threads')==[(1,2.),(2,2.)]
    assert edge.Scalars('combine/combine_total_s')[0].value==pytest.approx(.7)
    assert root.Scalars('combine/combine_total_s')[0].value==pytest.approx(.6)
    assert values(root,'gpu_compute/totals/kernel_ms')==[(1,300.),(2,300.)]
    assert values(edge,'gpu_compute/totals/kernel_ms')==[(1,200.),(2,200.)]
    assert values(root,'gpu_compute/task_total/totals/kernel_ms')==[(2,880.)]
    assert values(edge,'gpu_compute/totals/allocation_reuses')==[(1,7.),(2,7.)]
    assert values(edge,'gpu_compute/pool/reserved_bytes')==[(1,128.),(2,128.)]
    assert values(edge,'gpu_compute/kernel_resources/dgfl_gt_validate/num_registers')==[(1,255.),(2,255.)]
    all_tags=[*root.Tags()['scalars'],*client.Tags()['scalars'],*edge.Tags()['scalars']]
    assert all('not_a_real_kernel' not in tag and 'private' not in tag and '../' not in tag for tag in all_tags)
    assert 'proof/generation/proof_gen_s' not in root.Tags()['scalars']


def test_resource_frequency_power_temperature_and_epoch_overhead_keep_distinct_axes(tmp_path):
    logs=TensorBoardLogs(tmp_path)
    sample={'sample_index':7,'time':TIME,'cpu':{'system_utilization_percent':30.,'effective_frequency_mhz':3200.,
                                            'nominal_frequency_mhz':2500.,'reported_frequency_mhz':None},
            'gpu':{'graphics_clock_mhz':1600.,'sm_clock_mhz':1500.,'memory_clock_mhz':7000.,
                   'power_w':85.,'temperature_c':64.,'pstate':2,'throttle_reason_mask':0,'private':42.}}
    assert logs.write_machine_sample('resources','A',sample)['available']
    host=reader(logs.logdir/'resources/machines/A')
    assert values(host,'cpu/effective_frequency_mhz')==[(7,3200.)]
    assert values(host,'gpu/power_w')==[(7,85.)]
    assert values(host,'gpu/temperature_c')==[(7,64.)]
    assert 'cpu/reported_frequency_mhz' not in host.Tags()['scalars']
    assert all('private' not in tag for tag in host.Tags()['scalars'])
    assert logs.write_node_event('resources',{'id':'client1'},{'kind':'epoch','time':TIME,'payload':{
        'step':5,'epoch':1,'round_id':3,'accuracy':.5,'loss':1.,'metrics_eval_s':.02}})['available']
    assert logs.write_node_event('resources',{'id':'client1'},{'kind':'training','time':TIME,'payload':{
        'round_id':3,'epochs':2,'metrics_eval_s':.04,'metrics_callback_s':.03,'training_s':.3}})['available']
    client=reader(logs.logdir/'resources/nodes/client1')
    assert client.Scalars('training/metrics_eval_s')[0].step==5
    assert client.Scalars('training_round/metrics_eval_s')[0].step==3
    assert client.Scalars('training_round/metrics_callback_s')[0].value==pytest.approx(.03)
    assert '累计本地 epoch' in client.SummaryMetadata('training/metrics_eval_s').summary_description
    assert '全局轮次' in client.SummaryMetadata('training_round/metrics_eval_s').summary_description


def test_invalid_and_missing_diagnostics_do_not_create_zero_or_infinite_scalars(tmp_path):
    data=record()
    bad=data['rounds'][0]
    bad['stage_times']={'dkg_s':None,'aggregation_s':float('nan'),'combine_cpu_s':True,
                        'client_stage_s':-1,'training_wall_sum_s':10**400}
    bad['duration_s']=float('inf'); bad['bytes_sent']=None
    bad['accepted_clients']=None; bad['client_failures']=None
    bad['proof_generation_metrics']['client1']['proof_gen_s']=float('nan')
    bad['combine_metrics']['authority1']['combine_timings']['gpu_profile']['totals']['kernel_seconds']=None
    logs=TensorBoardLogs(tmp_path)
    assert logs.sync(data)['available']
    root=reader(logs.logdir/data['run_id'])
    assert values(root,'protocol/stages/dkg_s')==[(2,pytest.approx(.1))]
    assert 'communication/cumulative_round_bytes' not in root.Tags()['scalars']
    assert values(root,'round/duration_s')==[(2,12.)]
    assert values(root,'clients/accepted')==[(2,2.)]
    edge=reader(logs.logdir/data['run_id']/'nodes/authority1')
    assert values(edge,'gpu_compute/totals/kernel_ms')==[(2,200.)]
    assert logs._write('overflow',[(1,1.,{'unrepresentable':1e100})])['available']
    assert reader(logs.logdir/'overflow').Tags()['scalars']==[]


def test_chinese_titles_units_metadata_and_public_catalog_are_detached(tmp_path):
    data=record(); logs=TensorBoardLogs(tmp_path)
    assert logs.sync(data)['available']
    root=reader(logs.logdir/data['run_id'])
    metadata=root.SummaryMetadata('gpu_compute/totals/kernel_ms')
    assert 'CUDA 内核事件耗时' in metadata.display_name and '毫秒' in metadata.display_name
    assert '重叠' in metadata.summary_description and '全局轮次' in metadata.summary_description
    assert metadata.plugin_data.plugin_name=='scalars'
    assert '各客户端训练耗时之和' in root.SummaryMetadata('protocol/stages/training_wall_sum_s').display_name
    label,description=metric_metadata('protocol/stages/combine_cpu_s_max')
    assert '各边缘进程最大值' in label and 'CPU 秒' in label and '不可直接相加' in description
    catalog=scalar_label_catalog()
    assert catalog['fields']['power_w']=={'label':'GPU 功耗','unit':'瓦'}
    catalog['fields']['power_w']['label']='changed'
    catalog['groups'].clear(); catalog['kernels'].clear()
    assert scalar_label_catalog()['fields']['power_w']['label']=='GPU 功耗'
    assert scalar_label_catalog()['groups'] and scalar_label_catalog()['kernels']


def test_new_round_tags_can_be_backfilled_into_old_event_files_without_duplicates(tmp_path):
    data=record(); old=deepcopy(data)
    old['rounds']=[{name:row[name] for name in ('round','accuracy','loss')} for row in data['rounds']]
    old.pop('summary'); old.pop('status'); old.pop('evidence')
    first=TensorBoardLogs(tmp_path)
    assert first.sync(old)['available']
    first.close_all()
    logs=TensorBoardLogs(tmp_path)
    assert logs.sync(data)['available']
    assert logs.sync(data)['available']
    events=reader(logs.logdir/data['run_id'])
    assert len(events.Scalars('evaluation/accuracy'))==3
    assert len(events.Scalars('round/duration_s'))==2
    assert len(events.Scalars('gpu_compute/task_total/totals/kernel_ms'))==1
