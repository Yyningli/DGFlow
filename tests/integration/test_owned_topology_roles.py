"""Real identities, sealed owner submissions and configurable threshold rounds."""
import json
from copy import deepcopy

import numpy as np
import pytest

from dgfl.crypto import backend as b
from dgfl.crypto import protocol as p
from dgfl.deployment import cluster_topology
from dgfl.services import roles
from dgfl.topology import client_authorities, validate_topology
from dgfl.training import data as training_data
from dgfl.training import model as training_model
from dgfl.transport.security import Identity, create_cluster


@pytest.mark.parametrize('authority_count,aggregator_count,threshold,cloud_ids',[
    (3,4,2,[1,4]),(10,11,3,[1,10,11]),
])
def test_owned_round_shares_signed_cores_and_enforces_dynamic_thresholds(
        tmp_path,authority_count,aggregator_count,threshold,cloud_ids):
    topology=validate_topology(4,authority_count,aggregator_count,threshold,threshold)
    topology['client_authorities']=client_authorities(4,authority_count)
    authorities=[f'authority{i}' for i in range(1,authority_count+1)]
    clouds=[f'aggregator{i}' for i in range(1,aggregator_count+1)]
    clients=[f'client{i}' for i in range(1,5)]
    names=['coordinator',*authorities,*clouds,*clients]
    create_cluster(tmp_path/'keys',dict.fromkeys(names,'127.0.0.1'))
    cluster={**topology,'deployment':'single_host','nodes':{n:{'url':'https://test.invalid'} for n in names[1:]}}
    (tmp_path/'cluster.json').write_text(json.dumps(cluster),encoding='utf8')
    assert cluster_topology(cluster)==topology
    workers={n:roles.RoleWorker(n,tmp_path,Identity(tmp_path/'keys',n)) for n in names[1:]}
    coordinator=Identity(tmp_path/'keys','coordinator'); reference=[1,1]
    policy={**topology,'mode':'encrypted','scale':16,'bits':5,'dimension':2,'members':clients,'batch_size':2}
    ctx={**topology,'task_id':'owned-test','round_id':1,'key_epoch':'owned-epoch',
         'model_hash':b.digest(reference),'bits':5,'scale':16,'dimension':2}
    edges=[workers[n] for n in authorities]
    try:
        changed={**policy,'aggregator_threshold':1}
        with pytest.raises(ValueError):
            edges[0].execute('begin',{'context':ctx,'reference':reference,'clients':4,'mode':'encrypted','policy':changed})
        commits=[a.execute('begin',{'context':ctx,'reference':reference,'clients':4,'mode':'encrypted','policy':policy}) for a in edges]
        assert edges[-1].auth.node_id==authority_count
        assert all(a.auth.threshold==threshold for a in edges)
        acks=[a.execute('transcript',{'commitments':commits}) for a in edges]
        for dealer in edges:
            for receiver in edges:
                message=dealer.execute('share',{'recipient':receiver.node_id})
                receiver.execute('receive_share',{'message':message})
        for a in edges: a.execute('finalize',{'acks':acks})
        for invalid in ([], ['client1','client1'], ['client999'], [True]):
            with pytest.raises(ValueError,match='batch'):
                edges[0].execute('client_keys',{'client_ids':invalid})
        key_batches=[a.execute('client_keys',{'client_ids':clients}) for a in edges]
        assert all(set(batch)==set(clients) for batch in key_batches)
        packets={}; sealed={}; validation_keys={}
        for index,cid in enumerate(clients,1):
            client=workers[cid].identity
            messages=[batch[cid] for batch in key_batches]
            owner=workers[topology['client_authorities'][cid]]
            wrong_owner=next(edge for edge in edges if edge is not owner)
            with pytest.raises(ValueError,match='another authority'):
                wrong_owner.execute('forward_client_key_batch',{'key_messages':{cid:messages}})
            forwarded=owner.execute('forward_client_key_batch',{'key_messages':{cid:messages}})
            bundle=client.open(forwarded[cid],'owned-client-keys',roles.envelope_context(ctx),owner.node_id)['payload']
            assert bundle=={'context':ctx,'reference':reference,'key_messages':messages}
            with pytest.raises(ValueError):
                coordinator.open(forwarded[cid],'owned-client-keys',roles.envelope_context(ctx),owner.node_id)
            shares=[client.open(message,'client-key',roles.envelope_context(ctx),a.node_id)['payload']
                    for message,a in zip(bundle['key_messages'],edges)]
            with pytest.raises(ValueError): p.recover_client_key(shares[:threshold-1],threshold)
            key=p.recover_client_key(shares,threshold)
            values=[2*index-1,2*index]
            packet=p.encrypt(ctx,cid,values,key)
            packet['proof']=p.prove(ctx,cid,values,key,packet['ciphertext']); packets[cid]=packet
            owner=topology['client_authorities'][cid]
            sealed[cid]=client.seal(owner,'client-submission',packet,roles.envelope_context(ctx))
            validation_keys[cid]=[a.execute('validation_key',{'client_id':cid}) for a in edges]
            for forbidden in ('coordinator','aggregator1',next(a for a in authorities if a!=owner)):
                identity=coordinator if forbidden=='coordinator' else workers[forbidden].identity
                with pytest.raises(ValueError): identity.open(sealed[cid],'client-submission',roles.envelope_context(ctx),cid)
        with pytest.raises(ValueError,match='another authority'):
            workers['authority2'].execute('verify_owned',{'packets':{'client1':sealed['client1']},
                                                        'validation_keys':{'client1':validation_keys['client1']}})
        with pytest.raises(ValueError,match='threshold'):
            workers['authority1'].execute('verify_owned',{'packets':{'client1':sealed['client1']},
                                                        'validation_keys':{'client1':validation_keys['client1'][:threshold-1]}})
        verdicts=[]; cores={}; verified=[]
        for edge in edges:
            owned=[cid for cid in clients if topology['client_authorities'][cid]==edge.node_id]
            original=edge._verify_jobs
            def record(jobs,workers,original=original):
                verified.extend((job[1],edge.node_id) for job in jobs)
                return original(jobs,workers)
            edge._verify_jobs=record
            signed=edge.execute('verify_owned',{'packets':{cid:sealed[cid] for cid in owned},
                                               'validation_keys':{cid:validation_keys[cid] for cid in owned}})
            value=coordinator.verify_public(signed,'client-verification',edge.node_id)
            assert value['owned_clients']==owned
            assert set(value['submissions'])==set(owned)
            for cid,row in value['submissions'].items():
                assert row['proof_valid'] is True
                assert 'proof' not in row['core'] and row['core']['proof_hash']==b.digest(packets[cid]['proof'])
                assert row['packet_hash']==b.digest(row['core'])
                cores[cid]=row['core']
            verdicts.append(signed)
        assert sorted(cid for cid,_ in verified)==clients
        assert all(topology['client_authorities'][cid]==owner for cid,owner in verified)
        first=edges[0]
        for invalid in (verdicts[:-1],[*verdicts,verdicts[0]]):
            with pytest.raises(ValueError): first.execute('authorize',{'verification_results':invalid})
        with pytest.raises(ValueError,match='signed owner'):
            first.execute('authorize',{'packets':packets,'validation_keys':validation_keys})
        tampered=deepcopy(verdicts[1]['payload'])
        tampered['submissions']['client1']=deepcopy(verdicts[0]['payload']['submissions']['client1'])
        cross_owner=edges[1].identity.sign_public('client-verification',tampered)
        with pytest.raises(ValueError,match='membership'):
            first.execute('authorize',{'verification_results':[verdicts[0],cross_owner,*verdicts[2:]]})
        tampered=deepcopy(verdicts[1]['payload']); tampered['context_hash']='ab'*32
        stale=edges[1].identity.sign_public('client-verification',tampered)
        with pytest.raises(ValueError,match='context'):
            first.execute('authorize',{'verification_results':[verdicts[0],stale,*verdicts[2:]]})
        certificates=[a.execute('authorize',{'verification_results':verdicts}) for a in edges]
        decision=roles.authorization(coordinator,certificates,ctx)
        assert decision['approved']==clients
        assert {name:decision[name] for name in roles.TOPOLOGY_FIELDS}==topology
        with pytest.raises(ValueError,match='quorum'):
            roles.authorization(coordinator,certificates[:threshold-1],ctx)
        signed_parts=[]; parts=[]; verification_certificates=[]
        for cloud_id in cloud_ids:
            cloud=workers[f'aggregator{cloud_id}']
            materials=[a.execute('aggregate_key',{'certificates':certificates,'cloud_id':cloud_id}) for a in edges]
            request={'context':ctx,'packets':cores,'certificates':certificates,'materials':materials}
            with pytest.raises(ValueError,match='cores only'): cloud.execute('partial',{**request,'packets':packets})
            with pytest.raises(ValueError,match='threshold'):
                cloud.execute('partial',{**request,'materials':materials[:threshold-1]})
            signed=cloud.execute('partial',request); signed_parts.append(signed)
            part=coordinator.verify_public(signed,'partial',cloud.node_id)
            assert part['cloud_id']==cloud_id
            parts.append(part); verification_certificates.extend(part['verification_certificates'])
        trusted=roles.aggregate_verification(coordinator,verification_certificates,ctx,decision,dkg_commitments=commits)
        with pytest.raises(ValueError,match='threshold'):
            p.combine(ctx,parts[:threshold-1],threshold,4,decision['manifest_hash'],verification_materials=trusted,packets=cores)
        assert p.combine(ctx,parts,threshold,4,decision['manifest_hash'],verification_materials=trusted,packets=cores)==[16,20]
        confirmations=[a.execute('finish',{'parts':signed_parts}) for a in edges]
        assert all(value==confirmations[0] for value in confirmations)
        assert all(a.task_state[ctx['task_id']]['status']=='ready' for a in edges)
    finally:
        for worker in workers.values(): worker.close()


def test_real_cluster_refuses_unpinned_legacy_policy(tmp_path):
    topology=validate_topology(2,2,2,2,2)
    nodes=['authority1','authority2','aggregator1','aggregator2','client1','client2']
    create_cluster(tmp_path/'keys',dict.fromkeys(nodes,'127.0.0.1'))
    (tmp_path/'cluster.json').write_text(json.dumps({**topology,'deployment':'single_host',
        'nodes':{node:{'url':'https://test.invalid'} for node in nodes}}),encoding='utf8')
    worker=roles.RoleWorker('authority1',tmp_path,Identity(tmp_path/'keys','authority1'))
    reference=[1,1]; ctx={'task_id':'legacy-rejected','round_id':1,'key_epoch':'epoch',
        'model_hash':b.digest(reference),'bits':5,'scale':16,'dimension':2}
    with pytest.raises(ValueError,match='topology'):
        worker.execute('begin',{'context':ctx,'reference':reference,'clients':2,'mode':'encrypted'})


def test_training_receives_owner_key_model_bundle_and_uploads_only_one_sealed_proof(tmp_path,monkeypatch):
    topology=validate_topology(2,2,2,2,2)
    topology['client_authorities']=client_authorities(2,2)
    nodes=['authority1','authority2','aggregator1','aggregator2','client1','client2']
    create_cluster(tmp_path/'keys',dict.fromkeys(['coordinator',*nodes],'127.0.0.1'))
    (tmp_path/'cluster.json').write_text(json.dumps({**topology,'deployment':'single_host',
        'nodes':{node:{'url':'https://test.invalid'} for node in nodes}}),encoding='utf8')
    workers={node:roles.RoleWorker(node,tmp_path,Identity(tmp_path/'keys',node)) for node in nodes}
    reference=[1]*20
    policy={**topology,'mode':'encrypted','members':['client1','client2'],'batch_size':2,
            'bits':4,'scale':16,'dimension':20}
    ctx={**topology,'task_id':'owner-train','round_id':1,'key_epoch':'owner-training-epoch',
         'model_hash':b.digest(reference),'bits':4,'scale':16,'dimension':20}
    x=np.ones((8,1)); y=np.arange(8)%10
    monkeypatch.setattr(training_data,'load_mnist',lambda *args,**kwargs:(x,y,x,y))
    monkeypatch.setattr(training_model,'train_local',lambda *args,**kwargs:np.array(reference,dtype=float)/16)
    edges=[workers['authority1'],workers['authority2']]; client=workers['client1']
    try:
        commits=[a.execute('begin',{'context':ctx,'reference':reference,'clients':2,'mode':'encrypted','policy':policy}) for a in edges]
        acks=[a.execute('transcript',{'commitments':commits}) for a in edges]
        for dealer in edges:
            for receiver in edges:
                receiver.execute('receive_share',{'message':dealer.execute('share',{'recipient':receiver.node_id})})
        for a in edges: a.execute('finalize',{'acks':acks})
        messages=[a.execute('client_key',{'client_id':'client1'}) for a in edges]
        with pytest.raises(ValueError):
            edges[0].identity.open(messages[1],'client-key',roles.envelope_context(ctx),'authority2')
        with pytest.raises(ValueError,match='another authority'):
            edges[1].execute('forward_client_keys',{'client_id':'client1','key_messages':messages})
        bundle=edges[0].execute('forward_client_keys',{'client_id':'client1','key_messages':messages})
        opened=client.identity.open(bundle,'owned-client-keys',roles.envelope_context(ctx),'authority1')['payload']
        assert opened=={'context':ctx,'reference':reference,'key_messages':messages}
        client.execute('prepare_data',{'task_id':ctx['task_id'],'train_limit':8,'seed':42,'non_iid':False,'policy':policy})
        args={'context':ctx,'reference':reference,'mode':'encrypted','local_epochs':1,
              'seed':42,'backend':'numpy','policy':policy,'key_messages':bundle}
        wrong=edges[1].identity.seal('client1','owned-client-keys',opened,roles.envelope_context(ctx))
        with pytest.raises(ValueError): client.execute('train',{**args,'key_messages':wrong})
        changed=edges[0].identity.seal('client1','owned-client-keys',{**opened,'reference':[2]*20},roles.envelope_context(ctx))
        with pytest.raises(ValueError,match='pinned round'):
            client.execute('train',{**args,'key_messages':changed})
        result=client.execute('train',args)
        assert set(result)=={'packet','training_s','encrypt_s','proof_s','training_metrics',
                             'metrics_eval_s','metrics_callback_s'}
        assert isinstance(result['packet'],bytes)
        packet=edges[0].identity.open(result['packet'],'client-submission',roles.envelope_context(ctx),'client1')['payload']
        assert packet['context']==ctx and packet['client_id']=='client1' and 'proof' in packet
        assert client.execute('train',args)==result
        for forbidden in ('coordinator','authority2','aggregator1'):
            identity=Identity(tmp_path/'keys',forbidden)
            with pytest.raises(ValueError): identity.open(result['packet'],'client-submission',roles.envelope_context(ctx),'client1')
    finally:
        for worker in workers.values(): worker.close()
