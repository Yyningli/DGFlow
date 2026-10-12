"""Real multi-process federated experiment orchestration.

Private DKG shares and encryption materials remain sealed to their owning role.
The controller never receives individual plaintext models in secure modes.
"""
import hashlib
import importlib.machinery
import importlib.metadata
import json
import math
import platform
import re
import ssl
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np

from dgfl import system_info
from dgfl.crypto import backend as b
from dgfl.crypto import protocol as p
from dgfl.deployment import cluster_topology
from dgfl.experiments.monitoring import RunNodeMonitor, clean_event, frozen_nodes, monitor_snapshot
from dgfl.experiments.resources import check_wire_resources
from dgfl.experiments.telemetry import LocalProcessMonitor
from dgfl.experiments.tensorboard import TensorBoardLogs
from dgfl.services.roles import (
    AUTHORITIES as AUTHORITIES,
)
from dgfl.services.roles import (
    COMBINE_SECONDS,
    TOPOLOGY_FIELDS,
    aggregate_verification,
    authorization,
    combine_metrics,
    node_number,
    submission_core,
)
from dgfl.training.data import load_mnist
from dgfl.training.datasets import dataset_spec, feature_count, load_dataset, preprocessing_description
from dgfl.training.model import evaluate, initial_model
from dgfl.transport.client import RPCClient
from dgfl.transport.security import atomic_json
from dgfl.validation.policy import apply_batches, client_members, quantize, screening_settings


def utc(): return datetime.now(UTC).isoformat()


def execution_settings(config):
    execution=config.get('execution','auto')
    if execution not in ('auto','serial','parallel'):
        raise ValueError('invalid execution strategy')
    workers=config.get('rpc_workers',6)
    if type(workers) is not int or not 1<=workers<=24:
        raise ValueError('RPC worker count must be between 1 and 24')
    # The protocol and its checks do not depend on scheduling. Automatic runs
    # use the bounded pool for every mode; explicit serial remains a baseline.
    resolved='parallel' if execution=='auto' else execution
    return resolved,workers


def cloud_strategy_settings(config,*,cloud_count=None):
    """Validate requested scheduling, resolving auto only from actual eligibility.

    This overridable heuristic reflects the measured CPU/GPU tradeoff, not a
    universally optimal strategy. It never changes the cryptographic threshold
    or any independent proof check. Deployment counts alone do not resolve it.
    """
    requested=config.get('cloud_strategy','auto')
    if type(requested) is not str or requested not in ('auto','threshold','all'):
        raise ValueError('cloud strategy must be auto, threshold or all')
    if cloud_count is None: return requested
    if type(cloud_count) is not int or not 0<=cloud_count<=32:
        raise ValueError('eligible cloud count must be an integer between 0 and 32')
    if requested!='auto': return requested
    threshold=config.get('aggregator_threshold')
    if type(threshold) is not int or not 2<=threshold<=32:
        raise ValueError('resolved cloud strategy requires the actual aggregate threshold')
    compute_device=config.get('compute_device','cpu')
    if compute_device not in ('cpu','gpu'): raise ValueError('invalid aggregate compute device')
    if cloud_count<=threshold: return 'all'
    return 'threshold' if compute_device=='gpu' or cloud_count>=2*threshold else 'all'


def cloud_network_absence(exc):
    """Only missing transport may select a backup; TLS/validation must abort."""
    if not isinstance(exc,(httpx.TimeoutException,httpx.NetworkError)):
        return False
    pending=[exc]; seen=set()
    while pending:
        cause=pending.pop()
        if id(cause) in seen: continue
        seen.add(id(cause))
        if isinstance(cause,ssl.SSLError): return False
        # Some HTTP adapters replace the original SSL exception. Conservatively
        # fail closed for those messages too, never downgrade an identity error.
        text=str(cause).lower()
        if any(word in text for word in ('certificate','tls','ssl')): return False
        pending.extend(value for value in (cause.__cause__,cause.__context__) if value is not None)
    return True


_CLOUD_UNAVAILABLE=object()


def authenticate_cloud_part(identity,signed,cloud,ctx,manifest_hash):
    """Bind a returned part to its enrolled signer before collecting any backup."""
    part=identity.verify_public(signed,'partial',cloud)
    fields={'cloud_id','context_hash','manifest_hash','D','E','authority_ids',
            'verification_hash','verification_certificates'}
    dimension=ctx['dimension']
    if (not isinstance(part,dict) or set(part)!=fields or type(part['cloud_id']) is not int
            or part['cloud_id']!=node_number(cloud,'aggregator')
            or part['context_hash']!=b.digest(ctx) or part['manifest_hash']!=manifest_hash
            or type(part['D']) is not list or type(part['E']) is not list
            or len(part['D'])!=dimension or len(part['E'])!=dimension
            or type(part['authority_ids']) is not list or not part['authority_ids']
            or any(type(nid) is not int for nid in part['authority_ids'])
            or part['authority_ids']!=sorted(set(part['authority_ids']))
            or len(part['authority_ids'])<ctx.get('authority_threshold',2)
            or any(not 1<=nid<=ctx.get('authority_count',3) for nid in part['authority_ids'])
            or type(part['verification_hash']) is not str
            or len(part['verification_hash'])!=64
            or any(char not in '0123456789abcdef' for char in part['verification_hash'])
            or type(part['verification_certificates']) is not list
            or any(not isinstance(cert,dict) for cert in part['verification_certificates'])):
        raise ValueError('invalid authenticated cloud partial identity, context or structure')
    for value in [*part['D'],*part['E']]: b._blob(value,576)
    return part


def collect_cloud_parts(clouds,threshold,*,strategy,generate,request,authenticate,evidence):
    """Collect one common subset; backups receive keys only after absence.

    Request catches only explicitly classified missing transports. Authentication
    and the subsequent full combine are allowed to fail rather than select a
    different mathematical answer. Every batch completes before the next starts,
    so no late future can append another cloud after the subset is frozen.
    """
    if (type(threshold) is not int or not 2<=threshold<=32
            or strategy not in ('threshold','all') or not isinstance(clouds,(list,tuple))
            or any(type(cloud) is not str or cloud not in {f'aggregator{i}' for i in range(1,33)} for cloud in clouds)
            or len(set(clouds))!=len(clouds)):
        raise ValueError('invalid cloud collection strategy or membership')
    remaining=sorted(clouds,key=lambda cloud:node_number(cloud,'aggregator'))
    if len(remaining)<threshold: raise ValueError('cloud threshold not met')
    evidence.update(strategy=strategy,eligible_clouds=list(remaining),batches=[],
                    attempted_clouds=[],unavailable_clouds=[],selected_clouds=[])
    signed_parts=[]; parts=[]
    while remaining and (strategy=='all' or len(parts)<threshold):
        count=len(remaining) if strategy=='all' else threshold-len(parts)
        batch=remaining[:count]; del remaining[:count]
        evidence['batches'].append(list(batch)); evidence['attempted_clouds'].extend(batch)
        replies=request(batch,generate(batch))
        if not isinstance(replies,(list,tuple)) or len(replies)!=len(batch):
            raise ValueError('cloud response batch membership mismatch')
        for cloud,signed in zip(batch,replies):
            if signed is _CLOUD_UNAVAILABLE:
                evidence['unavailable_clouds'].append(cloud)
                continue
            part=authenticate(cloud,signed)
            signed_parts.append(signed); parts.append(part)
            evidence['selected_clouds'].append(cloud)
        if strategy=='all': break
    if len(parts)<threshold: raise ValueError('cloud threshold not met')
    return signed_parts,parts


def proof_description(suite):
    return {
        'legacy':'Pedersen range + linked square Sigma/Fiat-Shamir, research implementation',
        'compact_range_v1':'experimental same-curve aggregated range IPA + ciphertext/key/commitment/square Sigma',
        'compact_norm_v1':'experimental same-curve joint range/squared-norm IPA + ciphertext/key/commitment Sigma',
        'lego_norm_v1':'experimental circuit-specific trusted-setup LegoGroth16 range/norm + shared-response ciphertext/key/committed-witness Sigma',
    }[suite]


def training_failure_reason(node, action, exc):
    """Keep useful failure categories without publishing remote exception text."""
    if isinstance(exc, httpx.TimeoutException):
        return '训练请求超时，请检查客户端日志'
    if isinstance(exc, httpx.NetworkError):
        return '训练连接失败，请检查客户端进程与连接'
    status = None
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
    elif isinstance(exc, ValueError):
        match = re.match(r'^' + re.escape(f'{node}/{action}: ') + r'([1-5][0-9]{2})\b', str(exc))
        if match:
            status = int(match[1])
    if status is not None:
        text = str(exc).lower()
        dependency = ('；PyTorch 训练依赖无法加载' if 'torch' in text
                      else '；训练依赖无法加载' if 'importerror' in text else '')
        return f'训练请求失败（HTTP {status}{dependency}），请检查客户端日志'
    return '训练请求或响应校验失败（' + type(exc).__name__ + '），请检查客户端日志'


def require_client_training_backend(nodes, backend, clients):
    """Torch runs at the clients; the coordinator does not need its dependency."""
    if backend == 'numpy':
        return
    if backend != 'torch':
        raise ValueError('训练后端必须为 numpy 或 torch')
    indexed = {node['id']: node for node in nodes}
    unsupported = []
    for client in clients:
        node = indexed.get(client, {})
        capability = node.get('capabilities', {}).get('training_backends', {}).get('torch', {})
        if node.get('status') != 'online' or capability.get('available') is not True:
            unsupported.append(client)
    if unsupported:
        raise ValueError('Torch 训练后端未就绪：' + ', '.join(unsupported)
                         + ' 未在线、未安装 PyTorch 或尚未声明支持。请选择 NumPy，或安装并检查客户端 PyTorch。')


class RunExecutor:
    """One bounded pool per run; job order never depends on completion order."""
    def __init__(self,execution,workers):
        self.parallel=execution=='parallel'
        self.pool=ThreadPoolExecutor(max_workers=workers,thread_name_prefix='dgflow-rpc')

    def map(self,function,jobs,*,always_parallel=False):
        jobs=list(jobs)
        if (self.parallel or always_parallel) and len(jobs)>1:
            return list(self.pool.map(function,jobs))
        return [function(job) for job in jobs]

    def combine_and_confirm(self,combine,confirm,jobs):
        if not self.parallel:
            return combine(),self.map(confirm,jobs)
        # Submit independent calculations directly: a worker must never wait
        # for other work submitted to this same bounded pool.
        total=self.pool.submit(combine)
        confirmations=[self.pool.submit(confirm,job) for job in jobs]
        return total.result(),[future.result() for future in confirmations]

    def close(self):
        self.pool.shutdown(wait=True,cancel_futures=True)


def implementation_evidence():
    root=Path(__file__).resolve().parents[1]
    files={str(path.relative_to(root)).replace('\\','/'):hashlib.sha256(path.read_bytes()).hexdigest()
           for path in sorted(root.rglob('*.py'))}
    versions={}
    for name in ('numpy','py-arkworks-bls12381','dgfl-native','cryptography','fastapi','httpx'):
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]='unavailable'
    project=root.parents[1]
    native_sources={str(path.relative_to(project)).replace('\\','/'):hashlib.sha256(path.read_bytes()).hexdigest()
                    for relative in ('Cargo.toml','Cargo.lock','pyproject.toml','src/lib.rs','src/lego.rs','src/sigma.rs',
                                     'src/linked.rs','src/aggregate.rs','src/batch.rs')
                    if (path:=project/'native'/'dgfl-native'/relative).is_file()}
    native_artifacts={}
    try:
        distribution=importlib.metadata.distribution('dgfl-native')
        for relative in distribution.files or []:
            if relative.suffix in ('.pyd','.so','.dll'):
                path=Path(distribution.locate_file(relative))
                if path.is_file(): native_artifacts[str(relative).replace('\\','/')]=hashlib.sha256(path.read_bytes()).hexdigest()
    except importlib.metadata.PackageNotFoundError:
        pass
    loaded_artifacts={}
    for name,module in tuple(sys.modules.items()):
        if name=='dgfl_native' or name.startswith('dgfl_native.'):
            location=getattr(module,'__file__',None)
            if location and any(location.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES):
                path=Path(location)
                if path.is_file(): loaded_artifacts[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    from dgfl.crypto.lego_registry import available
    return {'source_sha256':b.digest(files),'source_files':files,'dependencies':versions,
            'native':{'enabled':bool(getattr(b,'NATIVE_EXTENSION',False)),
                      'package_version':versions['dgfl-native'],'build_input_sha256':native_sources,
                      'installed_artifact_sha256':native_artifacts,
                      'loaded_artifact_sha256':loaded_artifacts,
                      'loaded_capabilities':{'lego_norm_v1':available(),
                          'aggregate_verification_native':bool(b.NATIVE_EXTENSION and b.PublicAggregateVerifier is not None),
                          'ciphertext_sum_pairing_batch':b._native_batch('g1_sum_pairing_vector') is not None,
                          'cloud_key_pairing_batch':b._native_batch('g2_msm_pairing_vector') is not None,
                          'checked_gpu_point_batch':b._native_batch('checked_g1_coordinates_batch') is not None},
                      'evidence_scope':'controller imported extension paths and their current on-disk digests, plus checked loaded APIs; installed metadata does not attest node imports or an in-memory image after file replacement'},
            'python':platform.python_version(),'platform':system_info.system_name(),'machine':system_info.machine_name()}


def _run_members(config,cluster):
    topology=cluster_topology(cluster)
    for field in TOPOLOGY_FIELDS:
        if config.get(field) is not None and config[field]!=topology[field]:
            raise ValueError(f'实验拓扑 {field} 与已部署集群不一致，请先调整部署')
    config.update(topology)
    members=client_members(topology['client_count'])
    malicious=config.get('malicious_clients',0)
    if type(malicious) is not int or not 0<=malicious<=len(members):
        raise ValueError('恶意客户端数量必须为整数，且不能超过实验客户端数量')
    return members


def require_dataset_support(nodes,dataset,participants):
    """Check the selected roles, without changing existing absence tolerance."""
    dataset_spec(dataset)
    if dataset=='mnist':
        return
    observed={node['id']:node for node in nodes}
    unsupported=[name for name in participants if name not in observed
                 or observed[name].get('status')!='online'
                 or dataset not in observed[name].get('capabilities',{}).get('datasets',[])]
    if unsupported:
        raise ValueError('以下参与节点尚未支持 CIFAR-10，请更新并重启：'+', '.join(sorted(unsupported)))


def choose_participants(online,mode,excluded_clouds,*,batch_strategy='regroup',client_count=6,
                        authority_count=3,aggregator_count=3,authority_threshold=2,aggregator_threshold=2):
    from dgfl.topology import validate_topology
    validate_topology(client_count,authority_count,aggregator_count,authority_threshold,aggregator_threshold)
    if type(excluded_clouds) is not int or not 0<=excluded_clouds<=aggregator_count:
        raise ValueError('排除聚合节点数量必须为整数且不能超过部署数量')
    members=client_members(client_count)
    clients=[cid for cid in members if cid in online]
    clouds=[f'aggregator{i}' for i in range(1,aggregator_count+1-excluded_clouds) if f'aggregator{i}' in online]
    if mode!='plain':
        if not {f'authority{i}' for i in range(1,authority_count+1)}<=set(online): raise ValueError('本实现要求全部 DKG 节点参加建钥；不能启动新密钥周期')
        if len(clouds)<aggregator_threshold: raise ValueError(f'实际可用聚合节点不足 {aggregator_threshold}/{aggregator_count} 门限，未开始训练')
    if not apply_batches(set(clients),client_count,strategy=batch_strategy): raise ValueError('没有至少两人的合格在线客户端组')
    return clients,clouds


class RunManager:
    def __init__(self,runtime):
        self.runtime=Path(runtime).resolve(); self.runtime.mkdir(parents=True,exist_ok=True)
        self.lock=threading.RLock(); self.active=None; self.stop_event=threading.Event(); self.records={}
        self.last_nodes=[]
        self._monitors={}
        self._node_monitors={}
        self.tensorboard=TensorBoardLogs(self.runtime)
        for path in sorted((self.runtime/'results').glob('*/result.json')):
            try:
                record=json.loads(path.read_text('utf8'))
                if record['status'] in ('queued','running','stopping'):
                    record['status']='aborted'; record['error']='控制进程曾停止；请创建新任务。'
                    atomic_json(path,record)
                self.records[record['run_id']]=record
            except (ValueError,KeyError): continue

    def snapshot(self,run_id,*,include_history=True):
        def resource_view(value):
            if include_history or not isinstance(value,dict): return value
            result=dict(value)
            if isinstance(value.get('hardware'),dict):
                hardware=dict(value['hardware'])
                if 'samples' in hardware:
                    hardware.pop('samples')
                    hardware['history_omitted']=True
                result['hardware']=hardware
            return result

        with self.lock:
            if run_id not in self.records: raise KeyError(run_id)
            source=self.records[run_id]
            if not include_history:
                evidence=dict(source.get('evidence') or {})
                for field in ('resources','memory'):
                    if field in evidence: evidence[field]=resource_view(evidence[field])
                source={**source,'evidence':evidence}
            record=json.loads(json.dumps(source,allow_nan=False))
            monitor=self._monitors.get(run_id) if record['status'] in ('queued','running','stopping') else None
            node_monitor=self._node_monitors.get(run_id)
        # Cached sampling only: never wait for sensors while holding the run lock.
        if monitor is not None and callable(getattr(monitor,'snapshot',None)):
            try:
                resources=monitor.snapshot()
                record.setdefault('evidence',{})['resources']=json.loads(json.dumps(resource_view(resources),allow_nan=False))
            except Exception:
                pass  # A transient monitoring error must not hide experiment status.
        if node_monitor is not None:
            self._capture_node_monitor(record,node_monitor)
        return record

    def monitor_snapshot(self,run_id):
        with self.lock:
            if run_id not in self.records: raise KeyError(run_id)
            source=self.records[run_id]
            evidence=source.get('evidence') or {}
            record={'run_id':run_id,'status':source.get('status'),'config':dict(source.get('config') or {}),
                    'evidence':{key:json.loads(json.dumps(evidence[key])) for key in
                                ('node_snapshot','node_monitoring','members') if key in evidence}}
            collector=self._node_monitors.get(run_id)
        if collector is not None:
            self._capture_node_monitor(record,collector)
        return monitor_snapshot(record)

    def _sync_tensorboard(self,record):
        try:
            result=self.tensorboard.sync(record)
        except Exception as exc:
            result={'available':False,'url':'/tensorboard/',
                    'reason':'TensorBoard 日志写入失败：'+type(exc).__name__}
        with self.lock:
            record.setdefault('evidence',{})['tensorboard']=result
        return result

    def sync_tensorboard(self,run_id):
        # Export a detached snapshot so API requests cannot iterate a live list.
        return self._sync_tensorboard(self.snapshot(run_id))

    def _save(self,record):
        collector=self._node_monitors.get(record['run_id'])
        if collector is not None and self._capture_node_monitor(record,collector):
            totals=record['evidence']['node_monitoring'].get('summary',{}).get('training_rounds',{})
            for row in record.get('rounds',[]):
                measured=totals.get(str(row['round']))
                if measured:
                    row['monitoring_overhead']=dict(measured)
                    row['metrics_eval_s']=measured.get('metrics_eval_s')
        atomic_json(self.runtime/'results'/record['run_id']/'result.json',record)

    def _capture_node_monitor(self,record,collector):
        # Optional telemetry must not break persistence or hide a running job.
        # Validate a detached copy before replacing the last usable snapshot.
        try:
            snapshot=json.loads(json.dumps(collector.snapshot(),allow_nan=False))
            if not isinstance(snapshot,dict) or not isinstance(snapshot.get('summary',{}),dict):
                raise ValueError('invalid node monitoring snapshot')
        except Exception as exc:
            record.setdefault('evidence',{})['node_monitoring_error']=self._failure_detail(exc)
            return False
        record.setdefault('evidence',{})['node_monitoring']=snapshot
        return True

    def _failure_detail(self,exc):
        return f'{type(exc).__name__}: {exc}'.replace(str(self.runtime.parent),'[workspace]')[:600]

    def _emit(self,record,stage,message,round_id=0):
        with self.lock:
            record['events'].append({'time':utc(),'stage':stage,'message':message,'round':round_id})
            self._save(record)

    def start(self,config):
        config=dict(config)
        config.setdefault('proof_suite','lego_norm_v1')
        if config['proof_suite']!='lego_norm_v1':
            raise ValueError('新实验仅支持 LegoGroth16（lego_norm_v1）证明方案')
        config.setdefault('grid',8)
        config.setdefault('dataset','mnist')
        config['cloud_strategy']=cloud_strategy_settings(config)
        if config.get('client_count') is not None:
            config['client_count']=len(client_members(config['client_count']))
        compute_device=config.get('compute_device','cpu')
        if compute_device not in ('cpu','gpu'):
            raise ValueError('计算设备必须为 CPU 或 GPU')
        if compute_device=='gpu':
            if config['mode']=='plain':
                raise ValueError('GPU 模式用于密码批量计算，请选择加密实验模式')
            from dgfl.crypto.gpu import compute_capabilities
            capability=compute_capabilities()['gpu']
            if not capability['available']:
                raise ValueError(capability['reason'])
        config['compute_device']=compute_device
        config.update(screening_settings(config,defaults=True))
        execution,workers=execution_settings(config)
        with self.lock:
            if self.active is not None: raise ValueError('已有任务正在运行')
            if not (self.runtime/'cluster.json').exists(): raise ValueError('请先执行 dgflow init 和 dgflow start')
            cluster=json.loads((self.runtime/'cluster.json').read_text('utf8'))
            members = _run_members(config,cluster)
            resource_checks=check_wire_resources(config)
            authorities=[f'authority{i}' for i in range(1,config['authority_count']+1)]
            backend = config.get('backend', 'numpy')
            if backend not in ('numpy', 'torch'):
                raise ValueError('训练后端必须为 numpy 或 torch')
            if backend == 'torch':
                rpc = RPCClient(self.runtime, cluster['nodes'])
                try:
                    def training_health(client):
                        try:
                            health = rpc.call(client, 'health', retries=0, timeout=3)
                            return {'id': client, 'status': 'online',
                                    'capabilities': health.get('capabilities', {})}
                        except (httpx.HTTPError, ValueError):
                            return {'id': client, 'status': 'offline', 'capabilities': {}}
                    with ThreadPoolExecutor(max_workers=min(24, len(members))) as pool:
                        health = list(pool.map(training_health, members))
                    require_client_training_backend(health, backend, members)
                finally:
                    rpc.close()
            if compute_device=='gpu':
                rpc=RPCClient(self.runtime,cluster['nodes'])
                rpc.client.timeout=httpx.Timeout(3,connect=.5)
                try:
                    unsupported=[]
                    for authority in authorities:
                        try:
                            health=rpc.call(authority,'health',retries=0)
                            capability=health.get('capabilities',{}).get('compute',{}).get('gpu',{})
                            if not capability.get('available') or not capability.get('verified'):
                                unsupported.append(authority)
                        except (httpx.HTTPError,ValueError):
                            unsupported.append(authority)
                    if unsupported:
                        raise ValueError('以下授权节点未通过 GPU 自检，任务未创建：'+', '.join(unsupported))
                finally:
                    rpc.close()
            suite=config['proof_suite']
            crs_evidence=None
            if config['mode']!='plain':
                from dgfl.crypto.lego_registry import Registry, available
                crs_hash=config.get('proof_crs_hash')
                if not isinstance(crs_hash,str) or re.fullmatch(r'[0-9a-fA-F]{64}',crs_hash) is None:
                    raise ValueError('Lego 证明必须指定已安装参数的 64 位十六进制 CRS 摘要')
                config['proof_crs_hash']=crs_hash.lower()
                if not available(): raise ValueError('当前已加载原生扩展不支持 Lego；请安装匹配构建后重启服务')
                try:
                    crs_evidence=Registry(self.runtime).describe(config['proof_crs_hash'],
                        10*(feature_count(config['dataset'],config['grid'])+1),8)
                except OSError as exc:
                    raise ValueError('未安装匹配的 Lego 证明参数，任务未创建') from exc
            elif config.pop('proof_crs_hash',None) is not None:
                raise ValueError('明文基线不使用证明或 CRS 摘要')
            run_id=uuid.uuid4().hex
            record={'run_id':run_id,'status':'queued','config':dict(config),'created_at':utc(),'current_round':0,
                    'rounds':[],'events':[],'error':None,'summary':{},'evidence':{
                        'data':dataset_spec(config['dataset'])['name']+' public replay',
                        'dataset':config['dataset'],
                        'preprocessing':preprocessing_description(config['dataset'],config['grid']),
                        'crypto_backend':'BLS12-381 / py-arkworks-bls12381 wrapper + optional dgfl-native (locked dependencies)',
                        'proof':proof_description(suite) if config['mode']!='plain' else 'none (plain baseline)',
                        'wire_format':'binary-v2: raw curve points, chunked above 24 MiB, hash domain revised',
                        'wire_measurement':'application request + response bytes, excluding TLS/HTTP framing',
                        'training_backend':config.get('backend','numpy'),
                        'compute':{'requested':compute_device,'resolved':compute_device},
                        'execution':{'requested':config.get('execution','auto'),'resolved':execution,'rpc_workers':workers},
                        'cloud_strategy':{'requested':config['cloud_strategy'],'resolved':None},
                        'wire_resources':resource_checks,
                        'proof_settings':{name:config.get(name,default) for name,default in
                                          (('proof_suite','lego_norm_v1'),('verification','deterministic'),
                                           ('verification_workers',1),('verification_threads',2),('proof_block_size',128))},
                        'implementation':implementation_evidence()}}
            record['evidence']['node_snapshot']=frozen_nodes(cluster,config)
            if crs_evidence is not None:
                record['evidence']['proof_parameters']=crs_evidence
                record['evidence']['proof_settings'].pop('proof_block_size',None)
                record['evidence']['proof_settings']['proof_crs_hash']=config['proof_crs_hash']
            try:
                self._save(record)
            except Exception as exc:
                raise ValueError('任务初始记录无法保存，任务未启动：'+self._failure_detail(exc)) from exc
            self.records[run_id]=record; self.active=run_id; self.stop_event.clear()
            try:
                threading.Thread(target=self._run,args=(record,),daemon=True,name='dgflow-experiment').start()
            except Exception as exc:
                record['status']='failed'; record['error']='任务线程无法启动：'+self._failure_detail(exc)
                record['summary']={'accuracy':None,'elapsed_s':0.,'controller_cpu_s':0.,
                                   'bytes_sent':0,'completed_rounds':0}
                record['events'].append({'time':utc(),'stage':'finished','message':record['error'],'round':0})
                try:
                    self._save(record)
                except Exception as save_exc:
                    detail=self._failure_detail(save_exc)
                    record['evidence']['persistence']={'saved':False,'stage':'thread_start_failure','error':detail}
                    record['error']+='；失败记录无法保存：'+detail
                    record['events'][-1]['message']=record['error']
                finally:
                    self.active=None
                raise ValueError(record['error']) from exc
            return {'run_id':run_id,'status':'queued'}

    def stop(self,run_id):
        with self.lock:
            if self.active!=run_id: raise ValueError('该任务未在运行')
            self.stop_event.set(); self.records[run_id]['status']='stopping'; self._save(self.records[run_id])
        return {'status':'stopping'}

    def _check_stop(self):
        if self.stop_event.is_set(): raise InterruptedError('用户停止任务；已完成的密码调用会安全结束')

    def _run(self,record):
        rpc=None; executor=None; start=time.perf_counter(); cpu_start=time.process_time(); cfg=record['config']
        gpu_device=None; gpu_before=None
        monitor=None; node_monitor=None; monitoring_stage='construct'
        outcome='completed'; failure=None

        def monitoring_failure(stage,exc):
            # Sensor, callback and optional log failures are observability
            # diagnostics. They never change the protocol's result or outcome.
            with self.lock:
                record.setdefault('evidence',{})['monitoring']={
                    'available':False,'stage':stage,'error':self._failure_detail(exc)}

        def sample_observed(row):
            try:
                self.tensorboard.write_sample(record['run_id'],row)
            except Exception as exc:
                monitoring_failure('sample_log',exc)
            if node_monitor is not None:
                try: node_monitor.observe_local(row)
                except Exception as exc: monitoring_failure('sample_observer',exc)

        try:
            try:
                monitor=LocalProcessMonitor(self.runtime)
                monitor.interval_seconds=5
                monitor.on_sample=sample_observed
                with self.lock:
                    self._monitors[record['run_id']]=monitor
                monitoring_stage='start'; monitor.start()
            except Exception as exc:
                monitoring_failure(monitoring_stage,exc)
            execution,workers=execution_settings(cfg)
            executor=RunExecutor(execution,workers)
            record['evidence']['execution']={'requested':cfg.get('execution','auto'),'resolved':execution,'rpc_workers':workers}
            cluster=json.loads((self.runtime/'cluster.json').read_text('utf8'))
            record['evidence'].setdefault('node_snapshot',frozen_nodes(cluster,cfg))
            local_machine='local' if cluster.get('deployment')=='single_host' else 'A'
            machine_marker=self.runtime/'machine.json'
            if machine_marker.exists(): local_machine=json.loads(machine_marker.read_text('utf8'))['machine']
            record['evidence']['monitor_local_machine']=local_machine
            members=_run_members(cfg,cluster)
            cfg.setdefault('grid',8)
            cfg.setdefault('dataset','mnist')
            requested_strategy=cloud_strategy_settings(cfg)
            cfg['cloud_strategy']=requested_strategy
            record['evidence']['wire_resources']=check_wire_resources(cfg)
            record['evidence']['cloud_strategy']={'requested':requested_strategy,'resolved':None}
            topology={name:cfg[name] for name in TOPOLOGY_FIELDS}
            authorities=[f'authority{i}' for i in range(1,cfg['authority_count']+1)]
            cfg['client_count']=len(members)
            record['evidence']['client_count']=len(members)
            record['evidence']['members']=list(members)
            record['evidence']['deployment']=cluster['deployment']
            rpc=RPCClient(self.runtime,cluster['nodes']); record['status']='running'
            def many(jobs):
                self._check_stop()
                return executor.map(lambda job:rpc.call(*job),jobs)
            def shared_payload(payload):
                prepare=getattr(rpc,'prepare_payload',None)
                return prepare(payload) if callable(prepare) else payload
            def event(stage,message,r=0):
                if hasattr(monitor,'set_context'):
                    try: monitor.set_context(r,stage)
                    except Exception as exc: monitoring_failure('context',exc)
                self._emit(record,stage,message,r)
            def optional(jobs,stage,round_id,failures=None):
                def one(job):
                    node,action,payload=job
                    try: return rpc.call(node,action,payload)
                    except (httpx.HTTPError,ValueError) as exc:
                        if failures is not None:
                            reason = training_failure_reason(node, action, exc)
                            with self.lock:
                                failures[node] = reason
                            event(stage, f'{node} 本轮训练未成功返回；按缺席处理：{reason}', round_id)
                        else:
                            event(stage,f'{node} 本轮未成功返回；按缺席处理（{type(exc).__name__}）',round_id)
                        return None
                return executor.map(one,jobs)
            event('health','检查节点身份与双向 TLS 连接')
            def health(node):
                capabilities={}
                try:
                    reply=rpc.call(node,'health',retries=0,timeout=3); state='online'
                    capabilities=reply.get('capabilities',{}) if isinstance(reply,dict) else {}
                except (httpx.HTTPError,ValueError): state='offline'
                return {'id':node,'role':''.join(c for c in node if not c.isdigit()),'status':state,
                        'host':cluster['nodes'][node]['url'],'capabilities':capabilities}
            self.last_nodes=executor.map(health,cluster['nodes'],always_parallel=True)
            telemetry_nodes={row['id'] for row in self.last_nodes if row['capabilities'].get('node_telemetry') is True}
            if telemetry_nodes:
                try:
                    node_monitor=RunNodeMonitor(self.runtime,record['run_id'],record['evidence']['node_snapshot'],
                        self.tensorboard,local_machine=local_machine,local_epochs=cfg['local_epochs'],eligible_ids=telemetry_nodes)
                    with self.lock: self._node_monitors[record['run_id']]=node_monitor
                    node_monitor.start()
                    if callable(getattr(monitor,'snapshot',None)):
                        cached=monitor.snapshot().get('hardware',{}).get('samples',[])
                        if cached: node_monitor.observe_local(cached[-1])
                except Exception as exc:
                    record['evidence']['node_monitoring_error']=self._failure_detail(exc)
            online={s['id'] for s in self.last_nodes if s['status']=='online'}
            batched_authorization=all(any(s['id']==authority and s['capabilities'].get('batched_authorization') is True
                                         for s in self.last_nodes) for authority in authorities)
            batched_client_keys=all(any(s['id']==authority and s['capabilities'].get('batched_client_keys') is True
                                        for s in self.last_nodes) for authority in authorities)
            record['evidence']['execution']['batched_client_keys']=batched_client_keys
            record['evidence']['execution']['batched_authorization']=batched_authorization
            embedded_aggregate_certificates=all(any(
                s['id']==authority and s['capabilities'].get('embedded_aggregate_certificates') is True
                for s in self.last_nodes) for authority in authorities)
            record['evidence']['execution']['embedded_aggregate_certificates']=embedded_aggregate_certificates
            cloud_nodes=[f'aggregator{i}' for i in range(1,cfg['aggregator_count']+1)]
            if cfg['mode']!='plain' and not all(any(
                s['id']==authority and s['capabilities'].get('owned_validation') is True
                for s in self.last_nodes) for authority in authorities):
                raise ValueError('授权节点尚未支持单归属验证，请更新并重启全部授权节点')
            record['evidence']['protocol_topology']={**topology,'client_submission':'one sealed upload to assigned authority',
                'peer_verification':'authenticated owner verdicts and ciphertext cores; peers do not re-verify hidden client proofs',
                'cloud_communication':'no cloud-to-cloud messages','coordinator_role':'engineering RPC relay and public-result combiner'}
            compact_aggregate_materials=all(any(
                s['id']==node and s['capabilities'].get('compact_aggregate_materials') is True
                for s in self.last_nodes) for node in (*authorities,*cloud_nodes))
            ciphertext_only_aggregation=all(any(
                s['id']==node and s['capabilities'].get('ciphertext_only_aggregation') is True
                for s in self.last_nodes) for node in cloud_nodes)
            record['evidence']['execution']['compact_aggregate_materials']=compact_aggregate_materials
            record['evidence']['execution']['ciphertext_only_aggregation']=ciphertext_only_aggregation
            if cfg.get('compute_device','cpu')=='gpu':
                required=set(authorities)
                unsupported=[node['id'] for node in self.last_nodes if node['id'] in required
                             and (node['status']!='online' or not node['capabilities'].get('compute',{}).get('gpu',{}).get('available'))]
                if unsupported:
                    raise ValueError('以下授权节点不能执行 GPU 密码计算：'+', '.join(unsupported))
                from dgfl.crypto.gpu import require_gpu
                from dgfl.crypto.gpu import runtime as gpu_runtime
                event('gpu_prepare','自检并初始化各节点的 CUDA 密码批处理后端')
                coordinator_gpu=require_gpu()
                gpu_device=gpu_runtime(); gpu_before=gpu_device.stats_snapshot()
                gpu_roles=many([(a,'prepare_compute',{'compute_device':'gpu'}) for a in authorities])
                record['evidence']['compute'].update(coordinator=coordinator_gpu,
                                                    authorities=dict(zip(authorities,gpu_roles)),
                                                    accelerated_operations=coordinator_gpu['accelerated_operations'])
            screening=screening_settings(cfg,defaults=True)
            active_clients,active_clouds=choose_participants(online,cfg['mode'],cfg['offline_aggregators'],
                batch_strategy=screening['batch_strategy'],**{name:topology[name] for name in TOPOLOGY_FIELDS-{'client_authorities'}})
            required=set(active_clients)
            if cfg['mode']!='plain':
                required.update(authorities)
                required.update(active_clouds)
            require_dataset_support(self.last_nodes,cfg['dataset'],required)
            strategy=cloud_strategy_settings(cfg,cloud_count=len(active_clouds))
            record['evidence']['cloud_strategy'].update(
                resolved=strategy,eligible_cloud_count=len(active_clouds),
                aggregator_threshold=cfg['aggregator_threshold'],compute_device=cfg.get('compute_device','cpu'),
                resolution_rule='measured_gpu_surplus_or_cpu_twice_threshold' if requested_strategy=='auto' else 'explicit')
            require_client_training_backend(self.last_nodes, cfg.get('backend', 'numpy'), active_clients)
            if cfg.get('proof_suite')=='lego_norm_v1' and cfg['mode']!='plain':
                required=set(active_clients)|set(authorities)
                unsupported=[node['id'] for node in self.last_nodes if node['id'] in required
                             and not node['capabilities'].get('lego_norm_v1')]
                if unsupported:
                    raise ValueError('以下节点已加载的原生扩展不支持 Lego，请安装并重启：'+', '.join(unsupported))
            record['evidence']['initial_nodes']=self.last_nodes
            event('data','各客户端从本机公开数据缓存建立独立分区；原始样本不经节点 RPC 上传')
            scale,bits=128,8
            grid=cfg.get('grid',8)
            features=feature_count(cfg['dataset'],grid)
            dimension=features*10+10
            policy={'mode':cfg['mode'],'scale':scale,'bits':bits,'dimension':dimension,
                    'dataset':cfg['dataset'],
                    'members':list(members),'batch_size':2,
                    'proof_suite':cfg.get('proof_suite','legacy'),
                    'verification':cfg.get('verification','deterministic'),
                    'verification_workers':cfg.get('verification_workers',1),
                    'verification_threads':cfg.get('verification_threads',2),
                    'compute_device':cfg.get('compute_device','cpu'),
                    'proof_block_size':cfg.get('proof_block_size',128),**screening,**topology}
            record['evidence']['screening']=screening
            if policy['proof_suite']=='lego_norm_v1' and cfg['mode']!='plain':
                from dgfl.crypto.lego_registry import Registry, available
                if not available(): raise ValueError('当前原生扩展缺少 Lego 能力')
                policy.pop('proof_block_size')
                policy['proof_crs_hash']=cfg['proof_crs_hash']
                record['evidence']['proof_parameters']=Registry(self.runtime).describe(cfg['proof_crs_hash'],dimension,bits)
            record['evidence']['proof_settings']={name:policy[name] for name in
                                                 ('proof_suite','verification','verification_workers','proof_block_size','proof_crs_hash') if name in policy}
            record['evidence']['proof']=proof_description(policy['proof_suite']) if cfg['mode']!='plain' else 'none (plain baseline)'
            data_cfg={'task_id':record['run_id'],'train_limit':cfg['train_limit'],'seed':cfg['seed'],
                      'non_iid':cfg['non_iid'],'policy':policy}
            partitions=many([(cid,'prepare_data',data_cfg) for cid in active_clients])
            record['evidence']['partitions']=dict(zip(active_clients,partitions))
            if cfg['dataset']=='mnist':
                # Retain this adapter for existing callers and offline test fixtures.
                _,_,tx,ty=load_mnist(self.runtime.parent/'data'/'mnist',cfg['train_limit'],cfg['test_limit'],grid=grid)
            else:
                _,_,tx,ty=load_dataset(self.runtime.parent/'data',cfg['dataset'],cfg['train_limit'],cfg['test_limit'],grid=grid)
            reference=quantize(initial_model(cfg['seed'],features=features),scale,bits)
            initial=evaluate(np.asarray(reference)/scale,tx,ty,features=features)
            record['initial_metrics']=initial
            self._sync_tensorboard(record); self._save(record)
            # The paper amortises AuthSetup: the authorities are set up once and
            # only re-run when the client set changes. The key epoch is therefore
            # task-level and the ceremony below runs on the first round only.
            # Round binding for MaMA is untouched: ctx still carries round_id and
            # model_hash, and both are hashed into f = hash_point(ctx) every round.
            epoch=uuid.uuid4().hex; dkg_ready=False
            for round_id in range(1,cfg['rounds']+1):
                self._check_stop(); record['current_round']=round_id; round_start=time.perf_counter(); round_cpu_start=time.process_time(); bytes_before=rpc.bytes_sent
                record['current_validations']=[]; record['current_validations_round']=round_id
                times={}
                ctx={'task_id':record['run_id'],'round_id':round_id,'key_epoch':epoch,'model_hash':b.digest(reference),
                     'bits':bits,'scale':scale,'dimension':dimension,'dataset':cfg['dataset'],**topology}
                if policy['proof_suite']=='lego_norm_v1' and cfg['mode']!='plain':
                    ctx.update(proof_suite=policy['proof_suite'],proof_crs_hash=policy['proof_crs_hash'])
                elif policy['proof_suite']!='legacy' and cfg['mode']!='plain':
                    ctx.update(proof_suite=policy['proof_suite'],proof_block_size=policy['proof_block_size'])
                key_messages={cid:[] for cid in members}
                if cfg['mode']!='plain':
                    t=time.perf_counter(); event('dkg','密钥节点共同建钥，核对同一公开转录',round_id)
                    begin={'context':ctx,'reference':reference,'clients':len(members),'mode':cfg['mode'],'policy':policy}
                    commits=many([(a,'begin',begin) for a in authorities])
                    if not dkg_ready:
                        acks=many([(a,'transcript',{'commitments':commits}) for a in authorities])
                        messages=many([(a,'share',{'recipient':r}) for a in authorities for r in authorities])
                        many([(r,'receive_share',{'message':msg}) for (_,r),msg in zip([(a,r) for a in authorities for r in authorities],messages)])
                        many([(a,'finalize',{'acks':acks}) for a in authorities])
                        dkg_ready=True
                    if batched_client_keys:
                        key_batches=many([(a,'client_keys',{'client_ids':list(key_messages)}) for a in authorities])
                        if any(not isinstance(batch,dict) or set(batch)!=set(key_messages) for batch in key_batches):
                            raise ValueError('client key batch membership mismatch')
                        key_messages={cid:[batch[cid] for batch in key_batches] for cid in key_messages}
                        forwarding=[(a,'forward_client_key_batch',{'key_messages':{
                            cid:messages for cid,messages in key_messages.items() if topology['client_authorities'][cid]==a}})
                            for a in authorities if a in topology['client_authorities'].values()]
                        forwarded=many(forwarding); combined={}
                        for job,batch in zip(forwarding,forwarded):
                            if not isinstance(batch,dict) or set(batch)!=set(job[2]['key_messages']):
                                raise ValueError('owner-forwarded key batch membership mismatch')
                            combined.update(batch)
                        key_messages=combined
                    else:
                        client_key_jobs=[(a,'client_key',{'client_id':cid}) for cid in key_messages for a in authorities]
                        for index,material in enumerate(many(client_key_jobs)):
                            key_messages[client_key_jobs[index][2]['client_id']].append(material)
                        forwarded=many([(topology['client_authorities'][cid],'forward_client_keys',
                            {'client_id':cid,'key_messages':messages}) for cid,messages in key_messages.items()])
                        key_messages=dict(zip(key_messages,forwarded))
                    times['dkg_s']=time.perf_counter()-t
                attack_ids={f'client{i}' for i in range(1,cfg['malicious_clients']+1)} if cfg['attack']!='none' else set()
                dropped=attack_ids if cfg['attack']=='dropout' else set()
                participants=[c for c in active_clients if c not in dropped]
                t=time.perf_counter(); event('training','客户端本地训练、量化；安全模式生成密文与范围/范数证明',round_id)
                jobs=[]
                for cid in participants:
                    args={'context':ctx,'reference':reference,'mode':cfg['mode'],'local_epochs':cfg['local_epochs'],
                          'seed':cfg['seed']+round_id-1,'backend':cfg.get('backend','numpy'),
                          'attack':cfg['attack'] if cid in attack_ids else 'none','key_messages':key_messages[cid],
                          'policy':policy}
                    jobs.append((cid,'train',args))
                training_failures = {}
                raw_results=optional(jobs,'client_unavailable',round_id,failures=training_failures)
                replies={cid:result for cid,result in zip(participants,raw_results) if result is not None}; results=list(replies.values())
                record['current_client_failures'] = training_failures
                event('received',f'已收到 {len(replies)} 份客户端提交；正在检查参与组是否完整',round_id)
                times['client_stage_s']=time.perf_counter()-t
                times['training_wall_sum_s']=sum(v['training_s'] for v in results)
                times['encryption_wall_sum_s']=sum(v['encrypt_s'] for v in results)
                times['proof_wall_sum_s']=sum(v['proof_s'] for v in results)
                self._check_stop()
                if not apply_batches(set(replies), len(members), strategy=screening['batch_strategy']):
                    detail = '；'.join(f'{cid}：{training_failures[cid]}' for cid in members if cid in training_failures)
                    cause = ('本轮未收到任何客户端训练提交' if not replies
                             else '本轮客户端提交不足以组成至少两人的有效参与组')
                    raise ValueError(cause + ('；' + detail if detail else '，请检查客户端状态与掉线配置'))
                proof_verification_metrics=None
                combine_verification_metrics=None
                if cfg['mode']=='plain':
                    approved=apply_batches(set(replies),len(members),strategy=screening['batch_strategy'])
                    collateral=sorted(set(replies)-set(approved))
                    validations=[{'client_id':cid,'proof_valid':None,'score':None,
                                  'decision':'accepted' if cid in approved else 'rejected',
                                  'reason':'明文基线，不执行密码证明与异常筛选'} for cid in members]
                    if not approved: raise ValueError('无完整参与批次，本轮中止')
                    total=np.sum([replies[c]['plain_model'] for c in approved],axis=0).astype(int).tolist()
                else:
                    t=time.perf_counter(); event('validation','归属密钥节点验证客户端证明；共享签名验证结果并按统一门限筛选成员',round_id)
                    sealed_packets={cid:response['packet'] for cid,response in replies.items()}
                    validation_keys={cid:[] for cid in replies}
                    key_start=time.perf_counter()
                    if batched_authorization:
                        values=many([(a,'validation_keys',{'client_ids':list(replies)}) for a in authorities])
                        for value in values:
                            if not isinstance(value,dict) or set(value)!=set(replies):
                                raise ValueError('validation key batch membership differs from submitted clients')
                            for cid in replies: validation_keys[cid].append(value[cid])
                    else:
                        key_jobs=[(a,'validation_key',{'client_id':cid}) for cid in replies for a in authorities]
                        for index,material in enumerate(many(key_jobs)):
                            validation_keys[key_jobs[index][2]['client_id']].append(material)
                    times['validation_key_s']=time.perf_counter()-key_start
                    authorization_start=time.perf_counter()
                    owner_results=many([(a,'verify_owned',{
                        'packets':{cid:sealed_packets[cid] for cid in replies if topology['client_authorities'][cid]==a},
                        'validation_keys':{cid:validation_keys[cid] for cid in replies if topology['client_authorities'][cid]==a}})
                        for a in authorities])
                    packets={}
                    for owner,signed in zip(authorities,owner_results):
                        result=rpc.identity.verify_public(signed,'client-verification',owner)
                        if result['context_hash']!=b.digest(ctx) or result['topology']!=topology or result['owner']!=owner:
                            raise ValueError('归属节点验证结果与本轮拓扑不一致')
                        for cid,row in result['submissions'].items():
                            if cid not in replies or topology['client_authorities'][cid]!=owner or cid in packets:
                                raise ValueError('归属节点验证结果包含错误客户端')
                            packets[cid]=row['core']
                    if set(packets)!=set(replies): raise ValueError('归属节点验证结果未覆盖本轮提交')
                    authorization_payload=shared_payload({'verification_results':owner_results})
                    certs=many([(a,'authorize',authorization_payload) for a in authorities])
                    decision=authorization(rpc.identity,certs,ctx)
                    if decision.get('members')!=members:
                        raise ValueError('授权签名中的客户端成员与实验部署人数不一致')
                    approved=decision['approved']; collateral=decision['collateral']; validations=decision['validations']
                    times['authorization_s']=time.perf_counter()-authorization_start
                    times['validation_s']=time.perf_counter()-t
                    if policy['proof_suite']=='lego_norm_v1':
                        metrics_start=time.perf_counter()
                        measured=many([(authority,'proof_metrics',{}) for authority in authorities])
                        proof_verification_metrics={}
                        for authority,value in zip(authorities,measured):
                            if (not isinstance(value,dict) or value.get('context_hash')!=b.digest(ctx)
                                    or not isinstance(value.get('verifications'),dict)
                                    or set(value['verifications'])!={cid for cid in replies if topology['client_authorities'][cid]==authority}):
                                raise ValueError('Lego 证明耗时证据与本轮提交不一致')
                            for timing in value['verifications'].values():
                                if not isinstance(timing,dict) or any(
                                        type(timing.get(name)) not in (int,float)
                                        or not math.isfinite(timing[name]) or timing[name]<0
                                        for name in ('proof_verify_s','parameters_load_s')):
                                    raise ValueError('Lego 证明耗时证据包含无效计时')
                            proof_verification_metrics[authority]=value
                        times['proof_metrics_s']=time.perf_counter()-metrics_start
                    record['current_validations']=validations; record['current_validations_round']=round_id; self._save(record)
                    if not approved: raise ValueError('无完整合格批次，按协议中止本轮')
                    if len(active_clouds)<cfg['aggregator_threshold']:
                        raise ValueError(f"聚合节点不足 {cfg['aggregator_threshold']}/{cfg['aggregator_count']} 门限，未发布新模型")
                    t=time.perf_counter(); event('aggregation',f"{len(approved)} 个客户端获准；等待至少 {cfg['aggregator_threshold']} 份聚合结果",round_id)
                    packets={cid:submission_core(packets[cid]) for cid in approved}
                    times['aggregate_key_s']=0.; times['partial_decryption_s']=0.
                    cloud_selection={'requested_strategy':requested_strategy}
                    record['current_cloud_selection']=cloud_selection

                    def generate_materials(batch):
                        key_start=time.perf_counter()
                        cloud_ids=[node_number(cloud,'aggregator') for cloud in batch]
                        if batched_authorization:
                            values=many([(a,'aggregate_keys',{'certificates':certs,'cloud_ids':cloud_ids,
                                **({'compact_materials':True} if compact_aggregate_materials else {})}) for a in authorities])
                            if any(not isinstance(value,dict) or set(value)!={str(cid) for cid in cloud_ids} for value in values):
                                raise ValueError('aggregate key batch cloud membership differs from authorized recipients')
                            material_map={cloud:[value[str(node_number(cloud,'aggregator'))] for value in values] for cloud in batch}
                        else:
                            material_jobs=[(a,'aggregate_key',{'certificates':certs,'cloud_id':node_number(cloud,'aggregator'),
                                **({'compact_materials':True} if compact_aggregate_materials else {})})
                                           for cloud in batch for a in authorities]
                            all_materials=many(material_jobs)
                            material_map={cloud:[material for job,material in zip(material_jobs,all_materials)
                                          if job[2]['cloud_id']==node_number(cloud,'aggregator')] for cloud in batch}
                        times['aggregate_key_s']+=time.perf_counter()-key_start
                        return material_map

                    def request_parts(batch,material_map):
                        self._check_stop()
                        jobs=[(cloud,'partial',{'context':ctx,'packets':packets,'certificates':certs,
                                               'materials':material_map[cloud],
                                               'verification_threads':cfg.get('verification_threads',2)}) for cloud in batch]

                        def one(job):
                            try: return rpc.call(*job)
                            except Exception as exc:
                                if not cloud_network_absence(exc): raise
                                event('aggregator_unavailable',
                                      f'{job[0]} 网络缺席；仅为缺失结果选择候补（{type(exc).__name__}）',round_id)
                                return _CLOUD_UNAVAILABLE

                        partial_start=time.perf_counter()
                        replies=executor.map(one,jobs)
                        times['partial_decryption_s']+=time.perf_counter()-partial_start
                        return replies

                    signed_parts,parts=collect_cloud_parts(active_clouds,cfg['aggregator_threshold'],
                        strategy=strategy,generate=generate_materials,request=request_parts,
                        authenticate=lambda cloud,signed:authenticate_cloud_part(
                            rpc.identity,signed,cloud,ctx,decision['manifest_hash']),evidence=cloud_selection)
                    cloud_selection['selected_cloud_ids']=[part['cloud_id'] for part in parts]
                    cloud_selection['required_authorities']=list(authorities)
                    cloud_selection['coordinator_verifies_before_finish']=strategy=='threshold'
                    verification_start=time.perf_counter()
                    verification_certificates=[certificate for part in parts for certificate in part['verification_certificates']]
                    verification_materials=aggregate_verification(rpc.identity,verification_certificates,ctx,decision,
                                                                 dkg_commitments=commits)
                    times['aggregate_verification_s']=time.perf_counter()-verification_start
                    combine_start=time.perf_counter()
                    coordinator_timings={}
                    finish_payload={'parts':signed_parts}
                    if not embedded_aggregate_certificates:
                        finish_payload['verification_certificates']=verification_certificates
                    finish_payload=shared_payload(finish_payload)
                    def coordinator_combine():
                        return p.combine(ctx,parts,cfg['aggregator_threshold'],len(approved),decision['manifest_hash'],
                                         verification_materials=verification_materials,packets=packets,
                                         timings=coordinator_timings,compute_device=cfg.get('compute_device','cpu'),
                                         verification_threads=cfg.get('verification_threads',2))
                    finish_jobs=[(a,'finish',finish_payload) for a in authorities]
                    if strategy=='threshold':
                        # A bad selected cloud must fail full local math before
                        # any stateful finish is dispatched. Each edge then still
                        # performs its own unchanged independent full combine.
                        self._check_stop()
                        total=coordinator_combine()
                        confirmations=many(finish_jobs)
                    else:
                        total,confirmations=executor.combine_and_confirm(
                            coordinator_combine,lambda job:rpc.call(*job),finish_jobs)
                    times['combine_and_confirmation_s']=time.perf_counter()-combine_start
                    coordinator_metric=combine_metrics({'actor':'coordinator','context_hash':b.digest(ctx),
                                                        'round_id':round_id,'combine_timings':coordinator_timings},
                                                       ctx,'coordinator')
                    for name in COMBINE_SECONDS:
                        times[name]=coordinator_metric['combine_timings'][name]
                    # Each authority runs the identical combine inside finish, so
                    # retain every process's breakdown. Maxima remain available
                    # for older charts, but different stages' maxima may come from
                    # different processes and cannot be summed into a wall time.
                    metrics_start=time.perf_counter()
                    metric_request={'context_hash':b.digest(ctx),'round_id':round_id}
                    authority_metrics=optional([(a,'combine_metrics',metric_request) for a in authorities],
                                               'combine_metrics_unavailable',round_id)
                    times['combine_metrics_s']=time.perf_counter()-metrics_start
                    combine_verification_metrics={'coordinator':coordinator_metric}
                    for authority,measured in zip(authorities,authority_metrics):
                        combine_verification_metrics[authority]=(
                            None if measured is None else combine_metrics(measured,ctx,authority))
                    for name in COMBINE_SECONDS:
                        seen=[measured['combine_timings'][name] for actor,measured in combine_verification_metrics.items()
                              if actor!='coordinator' and measured is not None]
                        if seen:
                            times[name+'_max']=max(seen)
                    times['aggregation_s']=time.perf_counter()-t
                reference=quantize(np.asarray(total)/(len(approved)*scale),scale,bits)
                if cfg['mode']!='plain' and any(c['reference']!=reference for c in confirmations):
                    raise ValueError('下一轮模型与独立节点计算结果不一致')
                metrics=evaluate(np.asarray(reference)/scale,tx,ty,features=features)
                honest=set(key_messages)-attack_ids
                row={'round':round_id,**metrics,'accepted_clients':approved,'rejected_clients':sorted(set(key_messages)-set(approved)),
                     'collateral_clients':collateral,'duration_s':time.perf_counter()-round_start,'bytes_sent':rpc.bytes_sent-bytes_before,
                     'stage_times':times,'controller_cpu_s':time.process_time()-round_cpu_start,
                     'validations':validations,'model_hash':b.digest(reference),
                     'client_failures': dict(training_failures),
                     'attack_accepted':len(set(approved)&attack_ids),'attack_submitted':len(attack_ids & set(replies)),
                     'honest_rejected':len(honest-set(approved)),'honest_total':len(honest),
                     'reference_norm_squared':sum(x*x for x in reference)}
                client_metrics={}
                for cid,reply in replies.items():
                    epochs=reply.get('training_metrics') or []
                    if not isinstance(epochs,list) or len(epochs)>cfg['local_epochs']: continue
                    accepted=[]
                    for epoch in epochs:
                        try:
                            epoch_event=clean_event({'sequence':1,'kind':'epoch','time':epoch.get('time'),
                                                     'payload':epoch},cfg['local_epochs'])
                            if epoch_event['payload']['round_id']==round_id:
                                accepted.append({**epoch_event['payload'],'time':epoch_event['time']})
                        except (ValueError,TypeError,AttributeError): pass
                    if accepted: client_metrics[cid]=accepted
                    if node_monitor is not None: node_monitor.observe_training(cid,accepted)
                row['client_training_metrics']=client_metrics
                row['metrics_eval_s']=(sum(epoch.get('metrics_eval_s') or 0.
                    for epochs in client_metrics.values() for epoch in epochs) if client_metrics else None)
                if combine_verification_metrics is not None:
                    row['combine_metrics']=combine_verification_metrics
                    row['cloud_selection']=cloud_selection
                if proof_verification_metrics is not None:
                    row['proof_verification_metrics']=proof_verification_metrics
                    row['proof_generation_metrics']={}
                    for cid,reply in replies.items():
                        timing={'proof_gen_s':reply['proof_s']}
                        if 'proof_parameters_load_s' in reply:
                            timing['parameters_load_s']=reply['proof_parameters_load_s']
                            timing['parameters']=reply['proof_parameters']
                            timing['params_cached']=reply['proof_parameters']['prover_cached']
                        row['proof_generation_metrics'][cid]=timing
                with self.lock:
                    record['rounds'].append(row); self._save(record)
                event('completed_round',f'第 {round_id} 轮完成，测试准确率 {metrics["accuracy"]:.2%}',round_id)
                self._sync_tensorboard(record)
        except InterruptedError as exc:
            outcome='aborted'; failure=str(exc)
        except Exception as exc:
            outcome='aborted' if isinstance(exc,ValueError) else 'failed'
            failure=str(exc).replace(str(self.runtime.parent),'[workspace]')[:600]
        finally:
            cleanup_errors=[]

            def cleanup_failure(stage,exc):
                nonlocal outcome,failure
                detail=self._failure_detail(exc)
                cleanup_errors.append({'stage':stage,'error':detail})
                outcome='failed'
                failure=(failure+'；' if failure else '')+f'{stage}失败：{detail}'

            if executor:
                try: executor.close()
                except Exception as exc: cleanup_failure('executor_close',exc)
            if gpu_device is not None:
                try:
                    with gpu_device.lock:
                        gpu_profile=gpu_device.stats_delta(gpu_before)
                        record['evidence']['compute']['coordinator_kernel_execution']={
                            name:gpu_profile['totals'][name] for name in ('batches','rows')}
                        record['evidence']['compute']['coordinator_gpu_profile']=gpu_profile
                except Exception as exc: cleanup_failure('gpu_profile',exc)
            memory={'available':False,'scope':'local process monitor unavailable'}
            if monitor is not None:
                try: memory=monitor.finish()
                except Exception as exc:
                    monitoring_failure('finish',exc)
                    memory['error']=self._failure_detail(exc)
            record['evidence']['memory']=memory
            record['evidence']['resources']=memory
            if node_monitor is not None:
                try:
                    cached=memory.get('hardware',{}).get('samples',[])
                    if cached: node_monitor.observe_local(cached[-1])
                except Exception as exc: record['evidence']['node_monitoring_error']=self._failure_detail(exc)
                # A failed final sample/log write must never skip sampler stop.
                try:
                    record['evidence']['node_monitoring']=node_monitor.finish()
                except Exception as exc: record['evidence']['node_monitoring_error']=self._failure_detail(exc)
            self._sync_tensorboard(record)
            try:
                self.tensorboard.close(record['run_id'])
            except Exception as exc:
                record['evidence']['tensorboard']={'available':False,'url':'/tensorboard/',
                    'reason':'TensorBoard 日志关闭失败：'+type(exc).__name__}
            if rpc:
                try: rpc.close()
                except Exception as exc: cleanup_failure('rpc_close',exc)
            if cleanup_errors: record['evidence']['cleanup_errors']=cleanup_errors
            elapsed=time.perf_counter()-start
            with self.lock:
                # Publish status, error and summary in one atomic step. snapshot()
                # returns the in-memory record, so a poll must never observe a
                # terminal status before the matching summary exists.
                record['status']=outcome; record['error']=failure
                record['summary']={'accuracy':record['rounds'][-1]['accuracy'] if record['rounds'] else None,
                                   'elapsed_s':elapsed,'controller_cpu_s':time.process_time()-cpu_start,
                                   'bytes_sent':rpc.bytes_sent if rpc else 0,'completed_rounds':len(record['rounds'])}
                record['events'].append({'time':utc(),'stage':'finished',
                    'message':failure or '全部配置轮次已完成','round':record['current_round']})
                try:
                    self._save(record)
                except Exception as exc:
                    detail=self._failure_detail(exc)
                    record['status']='failed'
                    record['error']=(failure+'；' if failure else '')+'最终记录无法保存：'+detail
                    record['evidence']['persistence']={'saved':False,'stage':'finalize','error':detail}
                    record['events'][-1]['message']=record['error']
                finally:
                    self._monitors.pop(record['run_id'],None)
                    self._node_monitors.pop(record['run_id'],None)
                    if self.active==record['run_id']: self.active=None
