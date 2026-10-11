"""Private role workers; coordinator receives only permitted public results."""
import json
import logging
import math
import multiprocessing
import re
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from dgfl.crypto import backend as b
from dgfl.crypto import protocol as p
from dgfl.deployment import cluster_topology
from dgfl.topology import client_authorities, validate_topology
from dgfl.transport.security import atomic_bytes, atomic_json, parse_frame, read_bytes
from dgfl.validation.policy import (
    SCREENING_DEFAULTS,
    apply_batches,
    client_members,
    cosine,
    quantize,
    screen,
    screening_settings,
)

AUTHORITIES=['authority1','authority2','authority3']
_LOG=logging.getLogger(__name__)
TOPOLOGY_FIELDS={'client_count','authority_count','aggregator_count','authority_threshold',
                 'aggregator_threshold','client_authorities'}


def node_number(node_id,role):
    if not isinstance(node_id,str) or not re.fullmatch(role+r'[1-9][0-9]*',node_id):
        raise ValueError('invalid protocol node identity')
    return int(node_id[len(role):])


def pinned_topology(value):
    """Validate a complete immutable topology; omission is legacy-fixture only."""
    present=TOPOLOGY_FIELDS&set(value)
    if present and present!=TOPOLOGY_FIELDS:
        raise ValueError('complete pinned task topology is required')
    if not present:
        return None
    if not isinstance(value['client_authorities'],dict):
        raise ValueError('pinned client ownership mapping is required')
    topology=validate_topology(**{name:value[name] for name in TOPOLOGY_FIELDS-{'client_authorities'}})
    topology['client_authorities']=client_authorities(topology['client_count'],topology['authority_count'],value['client_authorities'])
    return topology


def role_topology(value):
    return pinned_topology(value) or {
        **validate_topology(client_count=len(value.get('members',[])) or 6,authority_count=3,aggregator_count=3,
                            authority_threshold=2,aggregator_threshold=2),
        'client_authorities':{},
    }


def authority_nodes(value):
    return [f'authority{i}' for i in range(1,role_topology(value)['authority_count']+1)]
_PROOF_DEFAULTS={'proof_suite':'legacy','verification':'deterministic',
                 'verification_workers':1,'verification_threads':2,'proof_block_size':128,'compute_device':'cpu'}
_LEGO_SUITE='lego_norm_v1'
COMBINE_SECONDS=('combine_numerators_s','combine_dkg_constants_s','combine_proof_verification_s',
                 'combine_interpolation_s','combine_context_materials_s','combine_cloud_E_s',
                 'combine_total_s','combine_cpu_s')
COMBINE_COUNTS=('combine_commitment_cache_hits','combine_commitment_cache_misses')


def combine_metrics(value,ctx,actor):
    """Check diagnostic evidence independently from the consensus response.

    CPU is process CPU, and overlapping processes' wall times must not be added.
    Only known diagnostic fields are retained, so future fields cannot introduce
    arbitrary payload into the experiment's public timing record.
    """
    if (not isinstance(value,dict) or value.get('actor')!=actor
            or value.get('context_hash')!=b.digest(ctx)
            or type(value.get('round_id')) is not int or value['round_id']!=ctx['round_id']
            or not isinstance(value.get('combine_timings'),dict)):
        raise ValueError('combine metrics identity or context mismatch')
    timings=value['combine_timings']; checked={}
    for name in COMBINE_SECONDS:
        seconds=timings.get(name)
        if type(seconds) not in (int,float) or seconds<0:
            raise ValueError('combine metrics contain invalid timing')
        try:
            finite=math.isfinite(seconds)
        except OverflowError:
            finite=False
        if not finite:
            raise ValueError('combine metrics contain invalid timing')
        checked[name]=seconds
    for name in COMBINE_COUNTS:
        count=timings.get(name)
        if type(count) is not int or count<0:
            raise ValueError('combine metrics contain invalid cache count')
        checked[name]=count
    if 'combine_dkg_transcript_reused' in timings:
        reused=timings['combine_dkg_transcript_reused']
        if type(reused) is not int or reused not in (0,1):
            raise ValueError('combine metrics contain invalid transcript reuse count')
        checked['combine_dkg_transcript_reused']=reused
    if 'gpu_profile' in timings:
        from dgfl.crypto.gpu_profile import checked_gpu_profile
        checked['gpu_profile']=checked_gpu_profile(timings['gpu_profile'])
    return {'actor':actor,'context_hash':value['context_hash'],'round_id':value['round_id'],
            'combine_timings':checked}


def _proof_policy(value):
    settings={k:value.get(k,default) for k,default in _PROOF_DEFAULTS.items()}
    if value.get('proof_crs_hash') is not None:
        settings['proof_crs_hash']=value['proof_crs_hash']
    if value.get('mode')=='plain' and settings['proof_suite']==_LEGO_SUITE:
        if 'proof_crs_hash' in settings:
            raise ValueError('plain task policy does not use a proof CRS')
        if type(settings['proof_block_size']) is not int or not 1<=settings['proof_block_size']<=1024:
            raise ValueError('unsupported proof block size')
    else:
        p.proof_settings(settings)
    if settings['compute_device'] not in ('cpu','gpu'):
        raise ValueError('invalid compute device policy')
    if settings['verification'] not in ('deterministic','randomized'):
        raise ValueError('invalid verification policy')
    workers=settings['verification_workers']
    if type(workers) is not int or not 1<=workers<=8:
        raise ValueError('verification worker count must be between 1 and 8')
    threads=settings['verification_threads']
    if type(threads) is not int or not 1<=threads<=4:
        raise ValueError('native verification thread budget must be between 1 and 4')
    return settings


def _matches_proof_policy(ctx,policy):
    settings=_proof_policy(policy)
    if policy.get('mode')=='plain' and settings['proof_suite']==_LEGO_SUITE:
        return all(ctx.get(name) is None for name in ('proof_suite','proof_crs_hash','proof_block_size'))
    suite,block_size=p.proof_settings(ctx)
    if suite==_LEGO_SUITE:
        return suite==settings['proof_suite'] and ctx.get('proof_crs_hash')==settings.get('proof_crs_hash')
    return suite==settings['proof_suite'] and (suite=='legacy' or block_size==settings['proof_block_size'])


def _verify_submission(job):
    """Spawned workers receive public material only, never DKG shares/signers."""
    if len(job) not in (7,8,9):
        raise ValueError('invalid public verification job')
    ctx,cid,packet,public,reference,materials,verification=job[:7]
    suite,_=p.proof_settings(ctx)
    parameters=None; parameters_load_s=0.
    if suite==_LEGO_SUITE:
        if len(job) not in (8,9):
            raise ValueError('Lego public verification job has no trusted parameters')
        from dgfl.crypto.lego_registry import public_parameters
        threads=job[8] if len(job)==9 else 1
        if type(threads) is not int or not 1<=threads<=4:
            raise ValueError('invalid public verification thread budget')
        start=time.perf_counter(); parameters=public_parameters(job[7],workers=threads)
        parameters_load_s=time.perf_counter()-start
        if (parameters.crs_hash!=ctx['proof_crs_hash'] or parameters.dimension!=ctx['dimension']
                or parameters.bits!=ctx['bits']):
            raise ValueError('public verification parameters differ from pinned task policy')
    elif len(job)!=7:
        raise ValueError('unexpected parameters for non-Lego verification')
    options={'verification':verification}
    if parameters is not None: options['proof_parameters']=parameters
    start=time.perf_counter(); checked=None
    valid=isinstance(packet,dict) and packet.get('context')==ctx and packet.get('client_id')==cid
    if valid:
        if parameters is not None:
            from dgfl.crypto.lego import verify_checked
            valid,checked=verify_checked(ctx,cid,packet.get('ciphertext',[]),packet.get('norm_squared'),
                                        packet.get('proof',{}),public,parameters,verification=verification)
        else:
            valid=p.verify(ctx,cid,packet.get('ciphertext',[]),packet.get('norm_squared'),
                           packet.get('proof',{}),public,**options)
    timings={'proof_verify_s':time.perf_counter()-start,'parameters_load_s':parameters_load_s}
    timings['checked_ciphertext_reused']=checked is not None
    if parameters is not None: timings['native_threads']=threads
    if not valid:
        return False,None,timings
    start=time.perf_counter()
    try:
        inner=p.validate_inner_product(ctx,packet['ciphertext'],reference,materials,role_topology(ctx)['authority_threshold'],
                                       norm_squared=packet['norm_squared'],checked=checked,client_id=cid)
        return True,cosine(inner,packet['norm_squared'],sum(z*z for z in reference)),timings
    except ValueError:
        return True,None,timings
    finally:
        timings['inner_product_s']=time.perf_counter()-start


def submission_core(packet):
    """Commit to the proof by hash; aggregation nodes need the ciphertext only."""
    return {k:packet[k] for k in ('context','client_id','ciphertext','norm_squared')} | {
        'proof_hash':b.digest(packet['proof']) if 'proof' in packet else packet['proof_hash']}


def _safe_id(value):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}',value):
        raise ValueError('invalid task identifier')
    return value


def _model_geometry(dimension,dataset='mnist'):
    """Map a linear-model dimension onto (features, pooling grid).

    The classifier is W[channels*grid*grid, 10] plus 10 biases. MNIST uses
    one channel and CIFAR-10 uses all three RGB channels. Missing dataset
    metadata retains the historical MNIST interpretation.
    """
    if type(dimension) is not int or dimension <= 0:
        raise ValueError('invalid model dimension')
    remainder = dimension - 10
    if remainder <= 0 or remainder % 10:
        raise ValueError('model dimension must be 10*(features+1)')
    features = remainder // 10
    from dgfl.training.datasets import dataset_spec
    spec=dataset_spec(dataset)
    channels=spec['channels']
    grid = math.isqrt(features//channels)
    if grid * grid * channels != features or not 1 <= grid <= spec['image_size']:
        raise ValueError('model dimension must use a square pooling grid matching dataset channels and resolution')
    return features, grid


def _policy(value):
    """Validate the complete immutable policy supported by these role workers."""
    fields={'mode','scale','bits','dimension','members','batch_size'}
    if (not isinstance(value,dict) or not fields<=set(value)
            or set(value)-fields-set(_PROOF_DEFAULTS)-set(SCREENING_DEFAULTS)-{'proof_crs_hash','dataset'}-TOPOLOGY_FIELDS):
        raise ValueError('complete task policy is required')
    if value['mode'] not in ('plain','encrypted','dgflow','optimized'):
        raise ValueError('invalid task policy mode')
    if type(value['scale']) is not int or value['scale']<=0:
        raise ValueError('invalid task policy scale')
    p._bounds(value)
    if 'dataset' in value:
        _model_geometry(value['dimension'],value['dataset'])
    settings=_proof_policy(value)
    screening_settings(value)
    if settings['compute_device']=='gpu' and value['mode']=='plain':
        raise ValueError('GPU mode requires an encrypted task policy')
    members=value['members']
    if (not isinstance(members,list) or not 2<=len(members)<=100
            or members!=[f'client{i}' for i in range(1,len(members)+1)]
            or type(value['batch_size']) is not int or value['batch_size']!=2):
        raise ValueError('invalid task policy members or fixed batches')
    topology=pinned_topology(value)
    if topology is not None and topology['client_count']!=len(members):
        raise ValueError('task topology membership mismatch')
    return {**value,'members':list(members)}


def envelope_context(ctx):
    """Bind sealed envelopes to the round as well as the key epoch.

    The key epoch is task-level so the DKG ceremony runs once, matching the
    paper's amortised AuthSetup. Without the round in the context, an envelope
    sealed in one round could be replayed in the next. The payload layer already
    rejects stale material; this keeps the envelope layer binding too.

    Exported so role harnesses decrypt with the same derivation instead of
    duplicating the wire format.
    """
    return f"{ctx['key_epoch']}:{ctx['round_id']}"


def _context(ctx):
    fields={'task_id','round_id','key_epoch','model_hash','bits','scale','dimension'}
    optional={'proof_suite','proof_block_size','proof_crs_hash'}
    data_fields={'dataset'}
    if not isinstance(ctx,dict) or not fields<=set(ctx) or set(ctx)-fields-optional-data_fields-TOPOLOGY_FIELDS:
        raise ValueError('invalid round context')
    _safe_id(ctx['task_id']); _safe_id(ctx['key_epoch']); p._bounds(ctx)
    if 'dataset' in ctx:
        _model_geometry(ctx['dimension'],ctx['dataset'])
    pinned_topology(ctx)
    suite,_=p.proof_settings(ctx)
    if suite=='legacy' and set(ctx)&optional:
        raise ValueError('legacy round context must omit compact proof fields')
    if suite==_LEGO_SUITE:
        if not {'proof_suite','proof_crs_hash'}<=set(ctx) or 'proof_block_size' in ctx:
            raise ValueError('Lego round context must pin suite and CRS without a block size')
    elif suite!='legacy' and (not {'proof_suite','proof_block_size'}<=set(ctx) or 'proof_crs_hash' in ctx):
        raise ValueError('compact round context must pin proof suite and block size')
    if (type(ctx['round_id']) is not int or ctx['round_id']<1
            or type(ctx['scale']) is not int or ctx['scale']<=0
            or not isinstance(ctx['model_hash'],str)
            or not re.fullmatch(r'[0-9a-f]{64}',ctx['model_hash'])):
        raise ValueError('invalid round context')
    return dict(ctx)


def authorization(identity,certificates,ctx):
    topology=role_topology(ctx); authorities=authority_nodes(ctx)
    decisions=[]; senders=[]
    for cert in certificates:
        sender=cert['sender']
        if sender not in authorities or sender in senders: raise ValueError('duplicate or unknown authorization signer')
        senders.append(sender); decisions.append(identity.verify_public(cert,'authorization',sender))
    if len(decisions)<topology['authority_threshold'] or any(v!=decisions[0] for v in decisions):
        raise ValueError('authorization quorum not met or conflicting decisions')
    decision=decisions[0]
    if decision['context_hash']!=b.digest(ctx): raise ValueError('authorization context mismatch')
    if pinned_topology(decision)!=pinned_topology(ctx):
        raise ValueError('authorization topology mismatch')
    if 'members' in decision:
        members=decision['members']; approved=decision['approved']
        if (type(members) is not list or members!=client_members(len(members))
                or type(approved) is not list or any(type(cid) is not str for cid in approved)
                or len(set(approved))!=len(approved)
                or not set(approved)<=set(members)):
            raise ValueError('authorization declared membership mismatch')
        if pinned_topology(ctx) is not None and len(members)!=topology['client_count']:
            raise ValueError('authorization topology membership mismatch')
    return decision


def aggregate_verification(identity,certificates,ctx,decision,*,dkg_commitments=None,authority=None):
    """Authenticate proof trust roots independently of cloud-supplied parts."""
    topology=role_topology(ctx); authorities=authority_nodes(ctx)
    if authority is not None:
        transcript=authority._transcript
    else:
        if dkg_commitments is None:
            raise ValueError('authenticated DKG commitments are required')
        transcript={}
        for signed in dkg_commitments:
            sender=signed['sender']
            if sender not in authorities or str(node_number(sender,'authority')) in transcript:
                raise ValueError('invalid aggregate verification DKG signer')
            transcript[str(node_number(sender,'authority'))]=identity.verify_public(signed,'dkg-commitments',sender)
        if set(transcript)!={str(i) for i in range(1,topology['authority_count']+1)}:
            raise ValueError('complete authenticated DKG transcript is required')
    records=[]; seen=set(); transcript_hash=b.digest(transcript)
    if 'members' in decision and any(row['clients']!=decision['members'] for row in transcript.values()):
        raise ValueError('aggregate DKG transcript membership differs from authorization')
    for signed in certificates:
        sender=signed['sender']
        if sender not in authorities:
            raise ValueError('invalid aggregate verification signer')
        record=identity.verify_public(signed,'aggregate-verification',sender)
        key=(record['cloud_id'],record['authority_id'])
        if (key in seen or record['authority_id']!=node_number(sender,'authority')
                or type(record['cloud_id']) is not int or not 1<=record['cloud_id']<=topology['aggregator_count']
                or record['context_hash']!=b.digest(ctx) or record['epoch']!=ctx['key_epoch']
                or record['manifest_hash']!=decision['manifest_hash']
                or record['approved']!=sorted(decision['approved']) or record['transcript_hash']!=transcript_hash):
            raise ValueError('aggregate verification identity or authorization mismatch')
        seen.add(key); records.append(record)
    return {'materials':records,'commitments':transcript}


class RoleWorker:
    def __init__(self,node_id,runtime,identity,telemetry=None):
        self.node_id=node_id; self.runtime=Path(runtime); self.identity=identity
        self.telemetry=telemetry
        self.folder=self.runtime/'nodes'/node_id; self.folder.mkdir(parents=True,exist_ok=True)
        self.auth=None; self.ctx=None; self.approval=None; self.finalized=False
        self._verification_pool=None; self._verification_pool_size=None
        self._proof_registry=None; self._public_proof_job=None
        self._proof_metrics=None; self._proof_parameters_evidence=None
        self._combine_metrics=None
        self.task_state_path=self.folder/'task-state.json'
        self.task_state=json.loads(self.task_state_path.read_text('utf8')) if self.task_state_path.exists() else {}

    def _check_cluster_policy(self,policy):
        """Bind real tasks to active deployment; isolated legacy fixtures have no cluster."""
        path=self.runtime/'cluster.json'
        if path.exists():
            cluster=json.loads(path.read_text('utf8'))
            topology=cluster_topology(cluster)
            if self.node_id not in cluster['nodes']:
                raise ValueError('task policy does not enroll this node in the active cluster')
            if (policy.get('members')!=client_members(topology['client_count'])
                    or pinned_topology(policy)!=topology):
                raise ValueError('task policy membership or topology differs from active cluster')

    def _check_context_policy(self,ctx,policy):
        if pinned_topology(ctx)!=pinned_topology(policy):
            raise ValueError('round context differs from pinned task topology')
        if ctx.get('dataset','mnist')!=policy.get('dataset','mnist'):
            raise ValueError('round context differs from pinned task dataset')

    # Same derivation the role harnesses import as ``envelope_context``.
    _envelope_context=staticmethod(envelope_context)

    def close(self):
        if self._verification_pool is not None:
            self._verification_pool.shutdown(wait=True,cancel_futures=True)
            self._verification_pool=None
            self._verification_pool_size=None

    def _verify_jobs(self,jobs,workers):
        if not jobs:
            return []
        if workers==1:
            return [_verify_submission(job) for job in jobs]
        if self._verification_pool is None or self._verification_pool_size!=workers:
            self.close()
            self._verification_pool=ProcessPoolExecutor(max_workers=workers,
                mp_context=multiprocessing.get_context('spawn'))
            self._verification_pool_size=workers
        try:
            return list(self._verification_pool.map(_verify_submission,jobs))
        except BrokenProcessPool as exc:
            self.close()
            raise ValueError('public verification worker failed; round authorization aborted') from exc

    def _registry(self):
        if self._proof_registry is None:
            from dgfl.crypto.lego_registry import Registry
            self._proof_registry=Registry(self.runtime)
        return self._proof_registry

    def execute(self,action,payload):
        if action=='prepare_compute':
            if (not self.node_id.startswith('authority') or not isinstance(payload,dict)
                    or payload!={'compute_device':'gpu'}):
                raise ValueError('invalid GPU preparation request')
            from dgfl.crypto.gpu import require_gpu
            return require_gpu()
        if self.node_id.startswith('authority'):
            return self._authority(action,payload)
        if self.node_id.startswith('client') and action=='train':
            return self._train(payload)
        if self.node_id.startswith('client') and action=='prepare_data':
            return self._prepare_data(payload)
        if self.node_id.startswith('aggregator') and action=='partial':
            return self._partial(payload)
        raise ValueError('operation not allowed for this node role')

    def _prepare_data(self,args):
        from dgfl.training.data import partition_clients
        from dgfl.training.datasets import load_dataset
        task=_safe_id(args['task_id']); target=self.folder/'data'/(task+'.npz')
        policy=_policy(args.get('policy'))
        self._check_cluster_policy(policy)
        if self.node_id not in policy['members']:
            raise ValueError('task policy does not enroll this training client')
        dataset=policy.get('dataset','mnist')
        _features,grid=_model_geometry(policy['dimension'],dataset)
        parameters_extra={}
        if policy.get('proof_suite')==_LEGO_SUITE and policy['mode']!='plain':
            start=time.perf_counter()
            _,_,evidence=self._registry().load_prover(policy['proof_crs_hash'],policy['dimension'],policy['bits'],workers=4)
            parameters_extra={'proof_parameters_load_s':time.perf_counter()-start,'proof_parameters':evidence}
        marker=target.with_suffix('.json'); config_hash=b.digest(args)
        if marker.exists():
            saved=json.loads(marker.read_text('utf8'))
            if saved['config_hash']!=config_hash or saved.get('policy')!=policy:
                raise ValueError('task data partition and policy already fixed; create a new task')
            return {**saved,**parameters_extra}
        if target.exists(): raise ValueError('existing task data has no registered policy; create a new task')
        x,y,_,_=load_dataset(self.runtime.parent/'data',dataset,args['train_limit'],100,grid=grid)
        indices=partition_clients(y,len(policy['members']),args['seed'],args['non_iid'])[policy['members'].index(self.node_id)]
        target.parent.mkdir(parents=True,exist_ok=True)
        temp=target.with_suffix('.tmp.npz'); np.savez_compressed(temp,x=x[indices],y=y[indices]); temp.replace(target)
        result={'config_hash':config_hash,'samples':len(indices),'partition_hash':b.digest(indices.tolist()),
                'policy':policy,'policy_hash':b.digest(policy),**parameters_extra}
        atomic_json(marker,result); return result

    def _verify_owned(self,payload):
        """Only the assigned edge sees a client's original proof submission."""
        topology=pinned_topology(self.policy)
        if (topology is None or not isinstance(payload,dict) or set(payload)!={'packets','validation_keys'}
                or self._owned_verification is not None):
            raise ValueError('owned verification requires a fresh pinned task')
        owned=[cid for cid in self.policy['members'] if topology['client_authorities'][cid]==self.node_id]
        envelopes=payload['packets']; all_keys=payload['validation_keys']
        if (not isinstance(envelopes,dict) or not isinstance(all_keys,dict)
                or not set(envelopes)<=set(owned) or set(all_keys)!=set(envelopes)
                or any(not isinstance(values,list) for values in all_keys.values())):
            raise ValueError('submission belongs to another authority')
        packets={cid:self.identity.open(envelope,'client-submission',self._envelope_context(self.ctx),cid)['payload']
                 for cid,envelope in envelopes.items()}
        authorities=authority_nodes(self.policy); settings=_proof_policy(self.policy)
        workers=settings['verification_workers']
        if settings['proof_suite']==_LEGO_SUITE: workers=min(workers,settings['verification_threads'])
        jobs=[]; submitted=list(packets)
        for cid in submitted:
            materials=[]; seen=set()
            for signed in all_keys[cid]:
                sender=signed['sender']
                if sender not in authorities or sender in seen: raise ValueError('invalid validation key signer')
                seen.add(sender); value=self.identity.verify_public(signed,'validation-key',sender)
                if (value['context_hash']!=b.digest(self.ctx) or value['client_id']!=cid
                        or value['material']['authority_id']!=node_number(sender,'authority')):
                    raise ValueError('validation key context or identity mismatch')
                materials.append(value['material'])
            if len(materials)<topology['authority_threshold']:
                raise ValueError('validation key threshold not met')
            job=(self.ctx,cid,packets[cid],self.auth.public_keys(cid),self.reference,materials,settings['verification'])
            if settings['proof_suite']==_LEGO_SUITE:
                if self._public_proof_job is None: raise ValueError('Lego round has no trusted public parameters')
                job=(*job,self._public_proof_job,max(1,settings['verification_threads']//workers))
            jobs.append(job)
        results=self._verify_jobs(jobs,workers); records={}; metrics={}
        for cid,result in zip(submitted,results):
            valid,score=result[:2]; packet=packets[cid]
            if not isinstance(packet,dict) or packet.get('context')!=self.ctx or packet.get('client_id')!=cid:
                raise ValueError('client submission context or identity mismatch')
            core=submission_core(packet)
            records[cid]={'core':core,'packet_hash':b.digest(core),'proof_valid':valid,'score':score}
            if len(result)==3: metrics[cid]=result[2]
        self._proof_metrics=metrics
        self._owned_verification={'context_hash':b.digest(self.ctx),'topology':topology,
                                  'owner':self.node_id,'owned_clients':owned,'submissions':records}
        return self.identity.sign_public('client-verification',self._owned_verification)

    def _authorize_shared(self,payload):
        """Trust authenticated owner verdicts, then independently apply screening.

        Peers receive ciphertext cores and signed verification results; they do
        not claim to re-verify the owner's hidden client proof.
        """
        if (not isinstance(payload,dict) or set(payload)!={'verification_results'}
                or not isinstance(payload['verification_results'],list)):
            raise ValueError('pinned topology requires signed owner verification results')
        topology=pinned_topology(self.policy); authorities=authority_nodes(self.policy)
        packets={}; results={}; senders=set()
        for signed in payload['verification_results']:
            if not isinstance(signed,dict): raise ValueError('invalid owner verification certificate')
            sender=signed.get('sender')
            if sender not in authorities or sender in senders: raise ValueError('invalid owner verification signer')
            value=self.identity.verify_public(signed,'client-verification',sender); senders.add(sender)
            owned=[cid for cid in self.policy['members'] if topology['client_authorities'][cid]==sender]
            if (not isinstance(value,dict) or set(value)!={'context_hash','topology','owner','owned_clients','submissions'}
                    or value['context_hash']!=b.digest(self.ctx) or value['topology']!=topology
                    or value['owner']!=sender or value['owned_clients']!=owned
                    or not isinstance(value['submissions'],dict) or not set(value['submissions'])<=set(owned)
                    or (sender==self.node_id and value!=self._owned_verification)):
                raise ValueError('owner verification context, topology or membership mismatch')
            for cid,row in value['submissions'].items():
                if (not isinstance(row,dict) or set(row)!={'core','packet_hash','proof_valid','score'}
                        or type(row['proof_valid']) is not bool
                        or (row['score'] is not None and (type(row['score']) not in (int,float)
                            or not math.isfinite(row['score']) or not -1.000001<=row['score']<=1.000001))
                        or (not row['proof_valid'] and row['score'] is not None)):
                    raise ValueError('invalid signed owner verification result')
                core=row['core']
                if (not isinstance(core,dict) or set(core)!={'context','client_id','ciphertext','norm_squared','proof_hash'}
                        or core['context']!=self.ctx or core['client_id']!=cid or b.digest(core)!=row['packet_hash']
                        or not isinstance(core['ciphertext'],list) or len(core['ciphertext'])!=self.ctx['dimension']
                        or type(core['norm_squared']) is not int or core['norm_squared']<0
                        or not isinstance(core['proof_hash'],str) or re.fullmatch('[0-9a-f]{64}',core['proof_hash']) is None):
                    raise ValueError('owner ciphertext core differs from signed manifest')
                packets[cid]=core; results[cid]=(row['proof_valid'],row['score'])
        if senders!=set(authorities): raise ValueError('complete owner verification results are required')
        return self._complete_authorization(packets,results)

    def _complete_authorization(self,packets,results):
        scores={}; valid_clients=set(); validations=[]
        for cid in self.auth.clients:
            valid=False; score=None; reason='客户端未在本轮提交'
            if cid in results:
                valid,score=results[cid][:2]; reason='证明未通过'
                if valid:
                    valid_clients.add(cid)
                    if score is not None: scores[cid]=score; reason='密码验证通过'
                    else: reason='零范数或内积解码失败'
            validations.append({'client_id':cid,'proof_valid':valid,'score':score,'decision':'pending','reason':reason})
        screening=screening_settings(self.policy); rejection_reasons={}
        if self.mode=='encrypted': candidates=valid_clients
        else: candidates,rejection_reasons=screen(scores,{cid:packets[cid]['norm_squared'] for cid in scores},screening)
        strategy=screening.get('batch_strategy','fixed')
        approved=apply_batches(candidates,self.clients,strategy=strategy); collateral=sorted(candidates-set(approved))
        for row in validations:
            cid=row['client_id']
            if cid in approved: row.update(decision='accepted',reason='通过并纳入聚合')
            elif cid in collateral: row.update(decision='collateral',reason=(
                '固定批次中其他成员未通过或缺席' if strategy=='fixed' else '合格成员不足两人，不能发布单人聚合'))
            elif cid in scores: row.update(decision='rejected',reason=rejection_reasons[cid])
            else: row['decision']='rejected'
        manifest={'context_hash':b.digest(self.ctx),'members':list(self.policy['members']),'approved':approved,
                  'packet_hashes':{cid:b.digest(submission_core(packets[cid])) for cid in approved},'batch_size':2}
        if screening: manifest['screening']=screening
        topology=pinned_topology(self.policy)
        if topology is not None: manifest.update(topology)
        self.approval={**manifest,'manifest_hash':b.digest(manifest),'validations':validations,'collateral':collateral}
        self._approved_packets={cid:packets[cid] for cid in approved}
        return self.identity.sign_public('authorization',self.approval)

    def _authority(self,action,payload):
        if action=='begin':
            ctx=_context(payload['context'])
            reference=payload['reference']; clients=payload['clients']; mode=payload['mode']
            if type(clients) is not int or not 2<=clients<=100 or mode not in ('encrypted','dgflow','optimized'):
                raise ValueError('invalid task policy')
            policy=_policy({'mode':mode,'scale':ctx['scale'],'bits':ctx['bits'],'dimension':ctx['dimension'],
                            'members':[f'client{i}' for i in range(1,clients+1)],'batch_size':2})
            if 'policy' in payload:
                supplied=_policy(payload['policy'])
                if any(supplied[k]!=policy[k] for k in policy):
                    raise ValueError('authority task policy mismatch')
                policy=supplied
            self._check_cluster_policy(policy)
            self._check_context_policy(ctx,policy)
            topology=role_topology(policy); authorities=authority_nodes(policy)
            if self.node_id not in authorities:
                raise ValueError('authority is not enrolled in task topology')
            settings=_proof_policy(policy)
            if not _matches_proof_policy(ctx,policy):
                raise ValueError('authority proof policy/context mismatch')
            offset,d=p._bounds(ctx)
            if len(reference)!=d or any(type(x) is not int or not -offset<=x<offset for x in reference) or b.digest(reference)!=ctx['model_hash']:
                raise ValueError('reference model hash or range mismatch')
            epoch_path=self.folder/'epochs'/(b.digest(ctx['key_epoch'])+'.json')
            # The key epoch is task-level, so later rounds reuse the participant
            # built on the first one instead of rebuilding it. Reuse keys on the
            # participant's own epoch, so a genuinely new epoch still initialises
            # and the replay guard below still fires for it.
            reuse=self.auth is not None and self.auth.epoch==ctx['key_epoch']
            if reuse and (self.ctx['task_id']!=ctx['task_id'] or self.auth.clients!=policy['members']
                    or self.auth.dimension!=d or self.auth.threshold!=topology['authority_threshold']
                    or self.auth.members!=list(range(1,topology['authority_count']+1))):
                raise ValueError('key epoch already pinned to a different task topology')
            if epoch_path.exists() and not reuse:
                raise ValueError('epoch already initialized; query cached request or create a new task')
            previous=self.task_state.get(ctx['task_id'])
            if previous and previous.get('policy')!=policy:
                raise ValueError('task policy already fixed; create a new task')
            if previous and previous.get('status')!='ready':
                raise ValueError('round already started; retry cached request or create a new task')
            if previous and (ctx['round_id']!=previous['next_round'] or reference!=previous['reference'] or mode!=previous['mode']):
                raise ValueError('round/model fork rejected')
            if not previous and ctx['round_id']!=1: raise ValueError('task must start at round one')
            public_job=None; parameters_evidence=None
            if settings['proof_suite']==_LEGO_SUITE:
                # Validate local pinned public parameters before committing any
                # task state or constructing the private DKG participant.
                _,parameters_evidence=self._registry().load_verifier(
                    settings['proof_crs_hash'],d,ctx['bits'],workers=1)
                public_job=self._registry().public_job(settings['proof_crs_hash'],d,ctx['bits'])
            self.task_state[ctx['task_id']]={'next_round':ctx['round_id'],'reference':list(reference),'mode':mode,
                                            'policy':policy,'status':'active','round_context':ctx}
            atomic_json(self.task_state_path,self.task_state)
            self.ctx=ctx; self.reference=list(reference); self.mode=mode; self.clients=clients; self.policy=policy
            self._public_proof_job=public_job; self._proof_parameters_evidence=parameters_evidence
            self._proof_metrics=None
            self._combine_metrics=None
            self._owned_verification=None
            if not reuse:
                self.auth=p.Authority(node_number(self.node_id,'authority'),list(range(1,topology['authority_count']+1)),
                                      policy['members'],d,topology['authority_threshold'],ctx['key_epoch'],
                                      verification=policy.get('verification','deterministic'),
                                      workers=policy.get('verification_threads',2))
                atomic_json(epoch_path,{'context':ctx,'status':'started','mode':mode})
                # Only a freshly built participant is unfinalised; a reused one
                # already completed the ceremony and must keep saying so, or
                # every later round's key derivation is rejected.
                self.finalized=False
            self.approval=None
            return self.identity.sign_public('dkg-commitments',self.auth.commitments())
        if self.auth is None: raise ValueError('epoch state unavailable; abort task and start a new task')
        topology=role_topology(self.policy); authorities=authority_nodes(self.policy)
        epoch=self.ctx['key_epoch']
        if action=='transcript':
            commits={}
            for signed in payload['commitments']:
                sender=signed['sender']
                if sender not in authorities or node_number(sender,'authority') in commits: raise ValueError('invalid DKG member')
                commits[node_number(sender,'authority')]=self.identity.verify_public(signed,'dkg-commitments',sender)
            if commits.get(self.auth.node_id)!=self.auth.commitments(): raise ValueError('own commitment replaced')
            self.auth.set_commitments(commits)
            return self.identity.sign_public('dkg-ack',{'epoch':epoch,'transcript_hash':self.auth.transcript_hash})
        if action=='share':
            recipient=payload['recipient']
            if recipient not in authorities: raise ValueError('private DKG share recipient forbidden')
            return self.identity.seal(recipient,'dkg-share',self.auth.share_for(node_number(recipient,'authority')),self._envelope_context(self.ctx))
        if action=='receive_share':
            env=payload['message']; sender=parse_frame(env)['sender']
            if sender not in authorities: raise ValueError('invalid DKG sender')
            packet=self.identity.open(env,'dkg-share',self._envelope_context(self.ctx),sender)['payload']
            self.auth.receive_share(node_number(sender,'authority'),packet)
            return {'received':sender}
        if action=='finalize':
            senders=[]
            for signed in payload['acks']:
                sender=signed['sender']
                if sender not in authorities or sender in senders: raise ValueError('invalid DKG acknowledgement signer')
                ack=self.identity.verify_public(signed,'dkg-ack',sender); senders.append(sender)
                if ack!={'epoch':epoch,'transcript_hash':self.auth.transcript_hash}: raise ValueError('DKG transcript disagreement')
            if set(senders)!=set(authorities): raise ValueError('all DKG members must acknowledge one transcript')
            self.auth.finalize(); self.finalized=True
            return {'status':'ready','transcript_hash':self.auth.transcript_hash}
        if not self.finalized: raise ValueError('DKG agreement not finalized')
        if action=='combine_metrics':
            if (not isinstance(payload,dict) or set(payload)!={'context_hash','round_id'}
                    or payload['context_hash']!=b.digest(self.ctx)
                    or type(payload['round_id']) is not int or payload['round_id']!=self.ctx['round_id']
                    or self._combine_metrics is None):
                raise ValueError('combine metrics are available only after a confirmed round')
            return combine_metrics(self._combine_metrics,self.ctx,self.node_id)
        if action=='proof_metrics':
            if payload or self.approval is None or self._proof_metrics is None:
                raise ValueError('proof metrics are available only after fixed round authorization')
            return {'context_hash':b.digest(self.ctx),'verifications':self._proof_metrics,
                    'parameters':self._proof_parameters_evidence}
        if action=='client_keys':
            if not isinstance(payload,dict) or set(payload)!={'client_ids'}:
                raise ValueError('invalid client-key batch')
            clients=payload['client_ids']
            if (type(clients) is not list or not clients or len(clients)>len(self.auth.clients)
                    or any(type(cid) is not str or cid not in self.auth.clients for cid in clients)
                    or len(set(clients))!=len(clients)):
                raise ValueError('invalid client-key batch')
            return {cid:self._authority('client_key',{'client_id':cid}) for cid in clients}
        if action=='client_key':
            if set(payload)!={'client_id'}: raise ValueError('invalid client-key request')
            cid=payload['client_id']
            return self.identity.seal(cid,'client-key',self.auth.client_share(cid),self._envelope_context(self.ctx))
        if action=='forward_client_key_batch':
            if (pinned_topology(self.policy) is None or not isinstance(payload,dict)
                    or set(payload)!={'key_messages'} or not isinstance(payload['key_messages'],dict)
                    or not payload['key_messages'] or len(payload['key_messages'])>len(self.auth.clients)
                    or any(topology['client_authorities'].get(cid)!=self.node_id for cid in payload['key_messages'])):
                raise ValueError('client key forwarding belongs to another authority or invalid batch')
            return {cid:self._authority('forward_client_keys',{'client_id':cid,'key_messages':messages})
                    for cid,messages in payload['key_messages'].items()}
        if action=='forward_client_keys':
            if pinned_topology(self.policy) is None or set(payload)!={'client_id','key_messages'}:
                raise ValueError('owner key forwarding requires pinned task topology')
            cid=payload['client_id']; messages=payload['key_messages']
            if topology['client_authorities'].get(cid)!=self.node_id or not isinstance(messages,list):
                raise ValueError('client key forwarding belongs to another authority')
            seen=set()
            for message in messages:
                header=parse_frame(message); sender=header['sender']
                if (sender not in authorities or sender in seen or header['recipient']!=cid
                        or header['purpose']!='client-key' or header['context']!=self._envelope_context(self.ctx)):
                    raise ValueError('invalid owner-forwarded client key share')
                seen.add(sender)
            if len(seen)<topology['authority_threshold']: raise ValueError('client key forwarding threshold not met')
            return self.identity.seal(cid,'owned-client-keys',{'context':self.ctx,'reference':self.reference,'key_messages':messages},
                                      self._envelope_context(self.ctx))
        if action=='validation_key':
            if set(payload)!={'client_id'}: raise ValueError('only the fixed reference function is authorized')
            cid=payload['client_id']; material=self.auth.validation_key(cid,self.reference)
            return self.identity.sign_public('validation-key',{'context_hash':b.digest(self.ctx),'client_id':cid,'material':material})
        if action=='validation_keys':
            if set(payload)!={'client_ids'}: raise ValueError('only fixed reference functions are authorized')
            clients=payload['client_ids']
            if (type(clients) is not list or not clients or len(clients)>len(self.auth.clients)
                    or any(type(cid) is not str or cid not in self.auth.clients for cid in clients)
                    or len(set(clients))!=len(clients)):
                raise ValueError('invalid validation key batch')
            return {cid:self._authority('validation_key',{'client_id':cid}) for cid in clients}
        if action=='verify_owned':
            if self.approval is not None: raise ValueError('round authorization already fixed')
            return self._verify_owned(payload)
        if action=='authorize':
            if self.approval is not None: raise ValueError('round authorization already fixed; retry original request')
            if pinned_topology(self.policy) is not None:
                return self._authorize_shared(payload)
            packets=payload['packets']; all_keys=payload['validation_keys']
            if any(cid not in self.auth.clients for cid in packets): raise ValueError('unregistered submission')
            settings=_proof_policy(self.policy); submitted=[]; jobs=[]
            workers=settings['verification_workers']
            if settings['proof_suite']==_LEGO_SUITE:
                workers=min(workers,settings['verification_threads'])
            for cid in self.auth.clients:
                if packets.get(cid) is None:
                    continue
                materials=[]; seen=set()
                for signed in all_keys.get(cid,[]):
                    sender=signed['sender']
                    if sender not in authorities or sender in seen: raise ValueError('invalid validation key signer')
                    seen.add(sender); signed_value=self.identity.verify_public(signed,'validation-key',sender)
                    if signed_value['context_hash']!=b.digest(self.ctx) or signed_value['client_id']!=cid:
                        raise ValueError('validation key context mismatch')
                    material=signed_value['material']
                    if material['authority_id']!=node_number(sender,'authority'): raise ValueError('validation key identity mismatch')
                    materials.append(material)
                submitted.append(cid)
                job=(self.ctx,cid,packets[cid],self.auth.public_keys(cid),self.reference,
                     materials,settings['verification'])
                if settings['proof_suite']==_LEGO_SUITE:
                    if self._public_proof_job is None:
                        raise ValueError('Lego round has no trusted public verification parameters')
                    threads=max(1,settings['verification_threads']//workers)
                    job=(*job,self._public_proof_job,threads)
                jobs.append(job)
            results=dict(zip(submitted,self._verify_jobs(jobs,workers)))
            self._proof_metrics={cid:result[2] for cid,result in results.items() if len(result)==3}
            return self._complete_authorization(packets,results)
        if action in ('aggregate_key','aggregate_keys'):
            required={'certificates','cloud_id' if action=='aggregate_key' else 'cloud_ids'}
            if set(payload) not in (required,required|{'compact_materials'}): raise ValueError('invalid aggregate key request')
            compact=payload.get('compact_materials',False)
            if type(compact) is not bool: raise ValueError('invalid aggregate material encoding')
            clouds=[payload['cloud_id']] if action=='aggregate_key' else payload['cloud_ids']
            if (type(clouds) is not list or not clouds or len(clouds)>topology['aggregator_count']
                    or any(type(cloud) is not int or not 1<=cloud<=topology['aggregator_count'] for cloud in clouds)
                    or len(set(clouds))!=len(clouds)):
                raise ValueError('invalid aggregate key batch')
            decision=authorization(self.identity,payload['certificates'],self.ctx)
            if decision!=self.approval or not decision['approved']: raise ValueError('unapproved aggregate function')
            output={}
            materials=self.auth.aggregate_keys(decision['approved'],clouds,topology['aggregator_threshold'],decision['manifest_hash'],context=self.ctx)
            for cloud_id in clouds:
                material=dict(materials[cloud_id])
                # Each certificate remains sealed to its designated cloud.
                material['verification_certificate']=self.identity.sign_public('aggregate-verification',material['verification'])
                if compact:
                    del material['verification']
                output[str(cloud_id)]=self.identity.seal(f'aggregator{cloud_id}','aggregate-key',material,self._envelope_context(self.ctx))
            return output[str(clouds[0])] if action=='aggregate_key' else output
        if action=='finish':
            self._combine_metrics=None
            if self.approval is None or not self.approval['approved']: raise ValueError('no approved aggregate')
            if (not isinstance(payload,dict)
                    or set(payload) not in ({'parts'},{'parts','verification_certificates'})
                    or not isinstance(payload['parts'],list)):
                raise ValueError('invalid aggregate confirmation request')
            parts=[]; seen=set(); embedded_certificates=[]
            for signed in payload['parts']:
                if not isinstance(signed,dict): raise ValueError('invalid signed partial')
                sender=signed.get('sender')
                if sender not in [f'aggregator{i}' for i in range(1,topology['aggregator_count']+1)] or sender in seen: raise ValueError('invalid partial signer')
                seen.add(sender); part=self.identity.verify_public(signed,'partial',sender)
                if (not isinstance(part,dict) or type(part.get('cloud_id')) is not int
                        or part['cloud_id']!=node_number(sender,'aggregator')):
                    raise ValueError('partial identity mismatch')
                # A cloud's signature authenticates transport, not the authority
                # certificates it carries. Verify their independent signers and
                # all math below, after reading only the authenticated body.
                embedded=part.get('verification_certificates')
                if not isinstance(embedded,list) or any(not isinstance(cert,dict) for cert in embedded):
                    raise ValueError('missing embedded aggregate verification certificates')
                embedded_certificates.extend(embedded)
                parts.append(part)
            if 'verification_certificates' in payload:
                explicit=payload['verification_certificates']
                if (not isinstance(explicit,list) or any(not isinstance(cert,dict) for cert in explicit)
                        or sorted(b.digest(cert) for cert in explicit)!=sorted(b.digest(cert) for cert in embedded_certificates)):
                    raise ValueError('explicit aggregate verification certificates differ from authenticated parts')
            verification=aggregate_verification(self.identity,embedded_certificates,
                                                  self.ctx,self.approval,authority=self.auth)
            combine_timings={}
            total=self.auth.combine(self.ctx,parts,topology['aggregator_threshold'],len(self.approval['approved']),self.approval['manifest_hash'],
                            verification_materials=verification,packets=self._approved_packets,
                            timings=combine_timings,compute_device=self.policy.get('compute_device','cpu'),
                            verification_threads=self.policy.get('verification_threads',2))
            # Kept off the response: confirmations from all authorities are
            # compared for equality, so diagnostic timings must not enter them.
            measured=combine_metrics({'actor':self.node_id,'context_hash':b.digest(self.ctx),
                                      'round_id':self.ctx['round_id'],'combine_timings':combine_timings},
                                     self.ctx,self.node_id)
            reference=quantize(np.asarray(total)/(len(self.approval['approved'])*self.ctx['scale']),self.ctx['scale'],self.ctx['bits'])
            self.task_state[self.ctx['task_id']]={'next_round':self.ctx['round_id']+1,'reference':reference,'mode':self.mode,
                                                 'policy':self.policy,'status':'ready'}
            atomic_json(self.task_state_path,self.task_state)
            self._combine_metrics=measured
            return {'reference':reference,'reference_hash':b.digest(reference)}
        raise ValueError('unsupported authority operation')

    def _train(self,args):
        from dgfl.training.model import attack_weights, train_local
        ctx=_context(args['context']); mode=args['mode']; reference=args['reference']
        data_path=self.folder/'data'/(ctx['task_id']+'.npz'); registration=data_path.with_suffix('.json')
        if not registration.exists(): raise ValueError('training task has no registered policy')
        saved=json.loads(registration.read_text('utf8')); policy=_policy(saved.get('policy'))
        self._check_cluster_policy(policy)
        self._check_context_policy(ctx,policy)
        topology=role_topology(policy); authorities=authority_nodes(policy)
        if (saved.get('policy_hash')!=b.digest(policy) or self.node_id not in policy['members']
                or mode!=policy['mode']
                or any(ctx[field]!=policy[field] for field in ('scale','bits','dimension'))
                or not _matches_proof_policy(ctx,policy)
                or ('policy' in args and _policy(args['policy'])!=policy)):
            raise ValueError('training request violates registered task policy')
        features,_=_model_geometry(policy['dimension'],policy.get('dataset','mnist'))
        offset,d=p._bounds(ctx)
        if (b.digest(reference)!=ctx['model_hash'] or len(reference)!=d
                or any(type(x) is not int or not -offset<=x<offset for x in reference)):
            raise ValueError('public model mismatch')
        proof_options={}; parameters_extra={}
        if policy.get('proof_suite')==_LEGO_SUITE and mode!='plain':
            start=time.perf_counter()
            prover,parameters,evidence=self._registry().load_prover(policy['proof_crs_hash'],d,ctx['bits'],workers=4)
            parameters_extra={'proof_parameters_load_s':time.perf_counter()-start,'proof_parameters':evidence}
            proof_options={'proof_parameters':parameters,'proof_prover':prover,'proof_workers':4}
        business_round={'task_id':ctx['task_id'],'round_id':ctx['round_id'],'client_id':self.node_id}
        marker=self.folder/'submissions'/(b.digest(business_round)+'.bin')
        business={k:v for k,v in args.items() if k!='key_messages'}; request_hash=b.digest(business)
        if marker.exists():
            old=read_bytes(marker)
            if old.get('context')!=ctx or old['request_hash']!=request_hash:
                raise ValueError('client round already fixed to different context or content')
            if 'result' in old: return old['result']
        key=None
        if mode!='plain':
            key_messages=args['key_messages']
            if pinned_topology(policy) is not None:
                owner=topology['client_authorities'][self.node_id]
                bundle=self.identity.open(key_messages,'owned-client-keys',self._envelope_context(ctx),owner)['payload']
                if (not isinstance(bundle,dict) or set(bundle)!={'context','reference','key_messages'}
                        or bundle['context']!=ctx or bundle['reference']!=reference
                        or not isinstance(bundle['key_messages'],list)):
                    raise ValueError('client key bundle differs from pinned round')
                key_messages=bundle['key_messages']
            shares=[]; seen=set()
            for env in key_messages:
                sender=parse_frame(env)['sender']
                if sender not in authorities or sender in seen: raise ValueError('invalid encryption-share sender')
                seen.add(sender); share=self.identity.open(env,'client-key',self._envelope_context(ctx),sender)['payload']
                if share['authority_id']!=node_number(sender,'authority'): raise ValueError('encryption-share identity mismatch')
                shares.append(share)
            key=p.recover_client_key(shares,topology['authority_threshold'])
            if key['client_id']!=self.node_id or key['epoch']!=ctx['key_epoch'] or len(key['s'])!=d:
                raise ValueError('encryption key/context mismatch')
        # Persist the business-round binding before training, including interrupted attempts.
        binding={**business_round,'context':ctx,'request_hash':request_hash,'status':'pending'}
        atomic_bytes(marker,binding)
        with np.load(data_path,allow_pickle=False) as data: x,y=data['x'].copy(),data['y'].copy()
        attack=args.get('attack','none')
        if attack=='label_flip': y=(y+1)%10
        training_metrics=[]; metrics_eval_s=0.; metrics_callback_s=0.

        def record_epoch(metrics):
            nonlocal metrics_eval_s,metrics_callback_s
            callback_start=time.perf_counter()
            try:
                row={**metrics,'time':datetime.now(UTC).isoformat(),'round_id':ctx['round_id'],
                     'step':(ctx['round_id']-1)*args['local_epochs']+metrics['epoch']}
                training_metrics.append(row)
                metrics_eval_s+=metrics['metrics_eval_s']
                if self.telemetry is not None:
                    try:
                        self.telemetry.record_epoch(ctx['task_id'],ctx['round_id'],args['local_epochs'],row)
                    except Exception as exc:
                        _LOG.warning('Client epoch telemetry unavailable (%s)',type(exc).__name__)
            finally:
                metrics_callback_s+=time.perf_counter()-callback_start

        start=time.perf_counter()
        weights=train_local(np.asarray(reference,dtype=float)/ctx['scale'],x,y,features=features,epochs=args['local_epochs'],
                            learning_rate=.3,batch_size=64,seed=args['seed'],backend=args.get('backend','numpy'),
                            on_epoch=record_epoch)
        if attack in ('sign_flip','random'): weights=attack_weights(weights,attack,args['seed'],features=features)
        q=quantize(weights,ctx['scale'],ctx['bits'])
        training_s=max(0.,time.perf_counter()-start-metrics_eval_s-metrics_callback_s)
        if mode=='plain':
            result={'plain_model':q,'training_s':training_s,'encrypt_s':0.,'proof_s':0.}
        else:
            start=time.perf_counter(); packet=p.encrypt(ctx,self.node_id,q,key,
                workers=policy.get('verification_threads',2)); encrypt_s=time.perf_counter()-start
            start=time.perf_counter(); packet['proof']=p.prove(ctx,self.node_id,q,key,packet['ciphertext'],**proof_options); proof_s=time.perf_counter()-start
            if attack=='tamper_proof': packet['norm_squared']+=1
            if pinned_topology(policy) is not None:
                packet=self.identity.seal(topology['client_authorities'][self.node_id],'client-submission',packet,self._envelope_context(ctx))
            result={'packet':packet,'training_s':training_s,'encrypt_s':encrypt_s,'proof_s':proof_s,**parameters_extra}
        result.update(training_metrics=training_metrics,metrics_eval_s=metrics_eval_s,
                      metrics_callback_s=metrics_callback_s)
        if self.telemetry is not None:
            try:
                publish=getattr(self.telemetry,'record_training_summary',None)
                if callable(publish):
                    publish(ctx['task_id'],ctx['round_id'],args['local_epochs'],result)
            except Exception as exc:
                _LOG.warning('Client training telemetry unavailable (%s)',type(exc).__name__)
        atomic_bytes(marker,{**binding,'status':'complete','result':result})
        return result

    def _partial(self,args):
        ctx=_context(args['context']); decision=authorization(self.identity,args['certificates'],ctx); packets=args['packets']
        self._check_cluster_policy(decision)
        topology=role_topology(ctx); authorities=authority_nodes(ctx)
        if pinned_topology(ctx) is not None and any(set(packet)!={'context','client_id','ciphertext','norm_squared','proof_hash'} for packet in packets.values()):
            raise ValueError('cloud accepts ciphertext cores only')
        if not decision['approved'] or set(packets)!=set(decision['approved']): raise ValueError('unapproved ciphertext membership')
        if any(b.digest(submission_core(packets[cid]))!=decision['packet_hashes'][cid] for cid in packets): raise ValueError('ciphertext manifest changed')
        cloud_id=node_number(self.node_id,'aggregator'); materials=[]; seen=set()
        if cloud_id>topology['aggregator_count']: raise ValueError('cloud is not enrolled in task topology')
        for env in args['materials']:
            sender=parse_frame(env)['sender']
            if sender not in authorities or sender in seen: raise ValueError('invalid aggregate material sender')
            seen.add(sender); material=self.identity.open(env,'aggregate-key',self._envelope_context(ctx),sender)['payload']
            fields={'authority_id','cloud_id','epoch','manifest_hash','keys','verification_certificate'}
            if (not isinstance(material,dict) or set(material) not in (fields,fields|{'verification'})
                    or type(material['authority_id']) is not int or material['authority_id']!=node_number(sender,'authority')
                    or type(material['cloud_id']) is not int or material['cloud_id']!=cloud_id or material['epoch']!=ctx['key_epoch']
                    or material['manifest_hash']!=decision['manifest_hash']):
                raise ValueError('aggregate authority identity or context mismatch')
            record=self.identity.verify_public(material['verification_certificate'],'aggregate-verification',sender)
            if (not isinstance(record,dict) or type(record.get('authority_id')) is not int
                    or record['authority_id']!=material['authority_id']
                    or type(record.get('cloud_id')) is not int or record['cloud_id']!=cloud_id
                    or record.get('context_hash')!=b.digest(ctx) or record.get('epoch')!=ctx['key_epoch']
                    or record.get('manifest_hash')!=decision['manifest_hash']
                    or record.get('approved')!=sorted(decision['approved'])
                    or ('verification' in material and b.digest(material['verification'])!=b.digest(record))):
                raise ValueError('aggregate certificate differs from authorized material')
            # Derive the exact record from its independently authenticated
            # certificate. Legacy duplicated records remain accepted if equal.
            material={**material,'verification':record}
            materials.append(material)
        part=p.partial_decrypt(ctx,packets,materials,topology['authority_threshold'],cloud_id,decision['manifest_hash'],
                              workers=args.get('verification_threads',1))
        part['verification_certificates']=[material['verification_certificate'] for material in materials]
        return self.identity.sign_public('partial',part)
