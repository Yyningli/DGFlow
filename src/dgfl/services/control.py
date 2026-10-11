"""Local experiment control and dashboard API."""
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.datastructures import URL, Headers
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from dgfl import __version__
from dgfl.deployment import configure_cluster, start_nodes
from dgfl.experiments.runner import RunManager
from dgfl.services.node import training_backends
from dgfl.services.tensorboard import EmbeddedTensorBoard
from dgfl.topology import TOPOLOGY_FIELDS, client_authorities, cluster_topology, validate_topology
from dgfl.training.data import prepare_mnist
from dgfl.training.datasets import dataset_spec
from dgfl.transport.client import RPCClient

CONTROL_BODY_LIMIT = 32 * 1024
DEPLOYMENT_SCHEMA_VERSION = 2
GPU_AUTOMATIC_REFRESH_SECONDS = 30
GPU_AUTOMATIC_RETRY_SECONDS = 1
GPU_AUTOMATIC_RETRY_ROUNDS = 15


def _origin(value: str):
    """Parse a single HTTP origin, rejecting URLs and ambiguous headers."""
    if not value or any(character.isspace() for character in value) or '?' in value or '#' in value:
        return None
    try:
        parsed=urlsplit(value)
        if (parsed.scheme not in ('http','https') or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.path or parsed.query or parsed.fragment or parsed.netloc.endswith(':')):
            return None
        port=parsed.port
    except ValueError:
        return None
    return parsed.scheme,parsed.hostname,port if port is not None else (443 if parsed.scheme=='https' else 80)


def _declared_body_allowed(lengths:list[str]):
    if not lengths:
        return True
    if len(lengths)!=1 or not lengths[0].isascii() or not lengths[0].isdigit():
        return False
    digits=lengths[0].lstrip('0') or '0'
    # Avoid unbounded integer parsing of malformed or extremely long headers.
    return len(digits)<=len(str(CONTROL_BODY_LIMIT)) and int(digits)<=CONTROL_BODY_LIMIT


class ControlSafetyMiddleware:
    """Check browser origin and bound actual bytes before any control action."""

    def __init__(self,app:ASGIApp):
        self.app=app

    async def __call__(self,scope:Scope,receive:Receive,send:Send):
        if scope['type']!='http':
            await self.app(scope,receive,send)
            return
        headers=Headers(scope=scope)
        origins=headers.getlist('origin')
        request_url=URL(scope=scope)
        expected=_origin(f'{request_url.scheme}://{request_url.netloc}')
        origin=_origin(origins[0]) if len(origins)==1 else None
        if origins and (origin is None or origin!=expected):
            response=JSONResponse({'detail':'控制操作仅允许同源本机页面'},status_code=403)
            await response(scope,receive,send)
            return
        lengths=headers.getlist('content-length')
        if not _declared_body_allowed(lengths):
            response=JSONResponse({'detail':'控制请求过大'},status_code=413)
            await response(scope,receive,send)
            return
        # Buffer at most 32 KiB even without Content-Length or when an action
        # ignores its body (for example stop). Do not dispatch until checked.
        body=bytearray()
        while True:
            message=await receive()
            if message['type']=='http.disconnect':
                return
            chunk=message.get('body',b'')
            if len(body)+len(chunk)>CONTROL_BODY_LIMIT:
                response=JSONResponse({'detail':'控制请求过大'},status_code=413)
                await response(scope,receive,send)
                return
            body.extend(chunk)
            if not message.get('more_body',False):
                break
        replayed=False

        async def checked_receive():
            nonlocal replayed
            if not replayed:
                replayed=True
                return {'type':'http.request','body':bytes(body),'more_body':False}
            return await receive()

        await self.app(scope,checked_receive,send)


class ArchivedRunConfig(BaseModel):
    """Read-only validation of historical declarations and experiment evidence."""
    model_config=ConfigDict(extra='forbid')
    dataset:Literal['mnist','cifar10']='mnist'
    mode:Literal['plain','encrypted','dgflow','optimized']='optimized'
    # Execution strategy is independent of the cryptographic protocol mode.
    # auto uses bounded parallelism; explicit serial remains a comparison.
    execution:Literal['auto','serial','parallel']='auto'
    cloud_strategy:Literal['auto','threshold','all']='auto'
    compute_device:Literal['cpu','gpu']='cpu'
    rpc_workers:int=Field(6,ge=1,le=24)
    proof_suite:Literal['legacy','compact_range_v1','compact_norm_v1','lego_norm_v1']='legacy'
    proof_crs_hash:str|None=Field(None,pattern=r'^[0-9a-fA-F]{64}$')
    verification:Literal['deterministic','randomized']='deterministic'
    verification_workers:int=Field(1,ge=1,le=8)
    verification_threads:int=Field(2,ge=1,le=4,strict=True)
    proof_block_size:int=Field(128,ge=1,le=1024)
    rounds:int=Field(3,ge=1,le=20)
    client_count:int=Field(6,ge=2,le=100,strict=True)
    authority_count:int|None=Field(None,ge=2,le=32,strict=True)
    aggregator_count:int|None=Field(None,ge=2,le=32,strict=True)
    authority_threshold:int|None=Field(None,ge=2,le=32,strict=True)
    aggregator_threshold:int|None=Field(None,ge=2,le=32,strict=True)
    seed:int=Field(42,ge=0,le=2**31-1)
    attack:Literal['none','label_flip','sign_flip','random','tamper_proof','dropout']='none'
    malicious_clients:int=Field(1,ge=0,le=100,strict=True)
    non_iid:bool=False
    offline_aggregators:int=Field(0,ge=0,le=32,strict=True)
    train_limit:int=Field(1200,ge=120,le=60000)
    test_limit:int=Field(400,ge=100,le=10000)
    local_epochs:int=Field(2,ge=1,le=5)
    backend:Literal['numpy','torch']='numpy'
    # MNIST has one channel; CIFAR-10 preserves all three RGB channels.
    # Resource preflight applies image resolution and the 20,000-coordinate cap.
    grid:int=Field(8,ge=2,le=32)
    min_cosine:float=Field(0.0,ge=-1,le=1,allow_inf_nan=False)
    max_norm_squared:int|None=Field(None,ge=1,strict=True)
    max_norm_ratio:float=Field(2.0,ge=1,allow_inf_nan=False)
    batch_strategy:Literal['fixed','regroup']='regroup'

    @model_validator(mode='after')
    def crs_matches_suite(self):
        from dgfl.experiments.resources import check_wire_resources

        check_wire_resources(self.model_dump())
        for role in ('authority','aggregator'):
            count=getattr(self,role+'_count'); threshold=getattr(self,role+'_threshold')
            if count is not None and threshold is not None and threshold>count:
                raise ValueError(f'{role}_threshold cannot exceed {role}_count')
        if self.aggregator_count is not None and self.offline_aggregators>self.aggregator_count:
            raise ValueError('offline_aggregators cannot exceed aggregator_count')
        if self.malicious_clients>self.client_count:
            raise ValueError('malicious_clients cannot exceed client_count')
        if self.compute_device=='gpu' and self.mode=='plain':
            raise ValueError('GPU 模式用于密码批量计算，请选择加密实验模式')
        if self.proof_suite=='lego_norm_v1':
            if self.mode=='plain':
                if self.proof_crs_hash is not None:
                    raise ValueError('Plain experiments do not use a proof CRS')
            elif self.proof_crs_hash is None:
                raise ValueError('Lego proof requires the fingerprint of an installed trusted setup')
            if self.proof_crs_hash is not None:
                self.proof_crs_hash=self.proof_crs_hash.lower()
        elif self.proof_crs_hash is not None:
            raise ValueError('proof_crs_hash is only valid for the Lego proof suite')
        return self


class RunConfig(ArchivedRunConfig):
    """New encrypted experiments support only the installed LegoGroth16 suite."""
    proof_suite:Literal['lego_norm_v1']='lego_norm_v1'


class DataConfig(BaseModel):
    model_config=ConfigDict(extra='forbid')
    dataset:Literal['mnist','cifar10']='mnist'


class DeploymentConfig(BaseModel):
    model_config=ConfigDict(extra='forbid')
    client_count:int=Field(6,ge=2,le=100,strict=True)
    authority_count:int|None=Field(None,ge=2,le=32,strict=True)
    aggregator_count:int|None=Field(None,ge=2,le=32,strict=True)
    authority_threshold:int|None=Field(None,ge=2,le=32,strict=True)
    aggregator_threshold:int|None=Field(None,ge=2,le=32,strict=True)

    @model_validator(mode='after')
    def thresholds_fit(self):
        for role in ('authority','aggregator'):
            count=getattr(self,role+'_count'); threshold=getattr(self,role+'_threshold')
            if count is not None and threshold is not None and threshold>count:
                raise ValueError(f'{role}_threshold cannot exceed {role}_count')
        return self


def create_control_app(runtime, *, auto_prepare_compute=True):
    runtime=Path(runtime).resolve(); manager=RunManager(runtime)
    tensorboard=EmbeddedTensorBoard(manager.tensorboard)
    compute_shutdown=threading.Event(); compute_wakeup=threading.Event()

    @asynccontextmanager
    async def lifespan(_app):
        if auto_prepare_compute:
            compute_wakeup.set()
            worker=threading.Thread(target=automatic_compute_worker,name='dgflow-gpu-discovery',daemon=True)
            worker.start()
            _app.state.compute_discovery_worker=worker
        try:
            yield
        finally:
            compute_shutdown.set(); compute_wakeup.set()
            tensorboard.close()
            manager.tensorboard.close_all()

    app=FastAPI(title='DGFlow Lab',docs_url=None,redoc_url=None,openapi_url=None,lifespan=lifespan)
    app.state.manager=manager
    app.state.tensorboard=tensorboard
    app.add_middleware(ControlSafetyMiddleware)
    app.add_middleware(TrustedHostMiddleware,allowed_hosts=['127.0.0.1','localhost','testserver','[::1]'])
    monitor={'time':0.,'nodes':[]}; monitor_lock=threading.Lock(); prepare_lock=threading.Lock()
    compute_lock=threading.Lock()
    compute_preparation={'state':'idle','reason':''}
    app.state.compute_preparation=compute_preparation

    def idle_for_deployment():
        if manager.active is not None:
            raise HTTPException(409,'实验运行中不能修改部署或启动节点')
        if prepare_lock.locked():
            raise HTTPException(409,'数据正在准备，暂时不能修改部署')
        with compute_lock:
            if compute_preparation['state']=='initializing':
                raise HTTPException(409,'GPU 后端正在初始化，暂时不能修改部署')

    def invalidate_nodes():
        with monitor_lock:
            monitor.update(time=0.,nodes=[])

    @app.post('/api/deployment/init')
    def deployment_init(config:DeploymentConfig):
        with manager.lock:
            idle_for_deployment()
            try:
                result=configure_cluster(runtime,control_locked=True,**config.model_dump(exclude_none=True))
            except (ValueError,OSError,RuntimeError) as exc:
                raise HTTPException(409,str(exc)) from exc
            finally:
                invalidate_nodes()
            if auto_prepare_compute:
                compute_wakeup.set()
            return result

    @app.post('/api/deployment/start')
    def deployment_start(config:DeploymentConfig):
        with manager.lock:
            idle_for_deployment()
            result=None
            try:
                result=configure_cluster(runtime,control_locked=True,**config.model_dump(exclude_none=True))
                result.update(start_nodes(runtime,control_locked=True))
            except (ValueError,OSError,RuntimeError) as exc:
                if result is not None:
                    return JSONResponse({'detail':f"节点启动失败；当前部署为 {result['client_count']} 个客户端、"
                                         f"{result['authority_count']} 个边缘节点、{result['aggregator_count']} 个云节点，"
                                         f"请查看节点日志后重试：{exc}",
                                         **{name:result[name] for name in TOPOLOGY_FIELDS},
                                         'client_authorities':result['client_authorities'],
                                         'deployment_committed':True},status_code=409)
                raise HTTPException(409,str(exc)) from exc
            finally:
                invalidate_nodes()
            if auto_prepare_compute:
                compute_wakeup.set()
            return result

    def refresh_nodes(cluster,topology):
        with monitor_lock:
            if time.monotonic()-monitor['time']<3: return monitor['nodes']
            if not (runtime/'keys'/'coordinator'/'identity.json').exists(): return []
            rpc=RPCClient(runtime,cluster['nodes']); rpc.client.timeout=httpx.Timeout(2,connect=.5)
            def one(item):
                node,settings=item
                capabilities={}
                try:
                    health=rpc.call(node,'health',retries=0); status='online'
                    capabilities=health.get('capabilities',{}) if isinstance(health,dict) else {}
                except Exception: status='offline'
                return {'id':node,'role':''.join(c for c in node if not c.isdigit()),'status':status,
                        'host':settings['url'],'capabilities':capabilities,
                        **({'authority_id':topology['client_authorities'][node]}
                           if node in topology['client_authorities'] else {})}
            try:
                with ThreadPoolExecutor(max_workers=min(32,len(cluster['nodes']))) as pool: nodes=list(pool.map(one,cluster['nodes'].items()))
            finally: rpc.close()
            monitor.update(time=time.monotonic(),nodes=nodes); return nodes

    def status_snapshot():
        from dgfl.crypto.gpu import compute_capabilities
        cluster=json.loads((runtime/'cluster.json').read_text('utf8')) if (runtime/'cluster.json').exists() else {'nodes':{},'deployment':'single_host'}
        topology=validate_topology(); cluster_error=None
        try:
            if cluster['nodes']:
                topology=cluster_topology(cluster)
        except ValueError as exc:
            cluster_error=str(exc)
            for name in TOPOLOGY_FIELDS[:3]:
                candidate=cluster.get(name)
                maximum=100 if name=='client_count' else 32
                if type(candidate) is int and 2<=candidate<=maximum:
                    topology[name]=candidate
            for role in ('authority','aggregator'):
                candidate=cluster.get(role+'_threshold',2)
                if type(candidate) is int and 2<=candidate<=topology[role+'_count']:
                    topology[role+'_threshold']=candidate
        topology.setdefault('client_authorities',client_authorities(topology['client_count'],topology['authority_count']))
        count=topology['client_count']
        nodes=refresh_nodes(cluster,topology) if cluster['nodes'] else []
        raw=runtime.parent/'data'/'mnist'/'raw'
        ready=all((raw/name).is_file() for name in ('train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz','t10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz'))
        from dgfl.training.cifar10 import cache_ready
        datasets={name:{**dataset_spec(name),'ready':ready if name=='mnist' else cache_ready(runtime.parent/'data'/'cifar10')}
                  for name in ('mnist','cifar10')}
        compute=compute_capabilities()
        backends=training_backends()
        for capability in backends.values():
            capability['unsupported_nodes']=[]
        by_id={node['id']:node for node in nodes}
        unsupported=[]; details=[]
        for client_id in (f'client{i}' for i in range(1,count+1)):
            client=by_id.get(client_id)
            if client is None or client['status']!='online':
                reason='未在线，无法确认 PyTorch 可用'
            else:
                capability=client['capabilities'].get('training_backends',{}).get('torch',{})
                if capability.get('installed') is True and capability.get('available') is True:
                    continue
                reason=capability.get('reason') or '未安装或未上报 PyTorch 支持信息'
            unsupported.append(client_id); details.append(f'{client_id}（{reason}）')
        # Only clients run Torch. A LAN controller can evaluate with NumPy while
        # every remote client has Torch; its local installed flag is diagnostic.
        backends['torch'].update(available=not unsupported and not cluster_error,
                                 unsupported_nodes=unsupported,
                                 reason=('以下训练客户端无法使用 PyTorch：'+', '.join(details) if unsupported
                                         else '部署配置不完整：'+cluster_error if cluster_error else ''))
        with compute_lock:
            compute['preparation']=dict(compute_preparation)
        if compute['gpu']['available']:
            unsupported=[node['id'] for node in nodes if node['role']=='authority'
                         and (node['status']!='online'
                              or not node['capabilities'].get('compute',{}).get('gpu',{}).get('available')
                              or not node['capabilities'].get('compute',{}).get('gpu',{}).get('verified'))]
            if len([node for node in nodes if node['role']=='authority'])!=topology['authority_count']:
                unsupported.append('配置的边缘节点尚未全部就绪')
            if unsupported:
                compute['gpu']={**compute['gpu'],'available':False,
                                'reason':'GPU 模式需要所有授权节点可用：'+', '.join(unsupported)}
            elif cluster_error:
                compute['gpu']={**compute['gpu'],'available':False,'reason':'部署配置不完整：'+cluster_error}
        return {'version':__version__,'backend':'BLS12-381/arkworks','node_count':len(nodes),'active_run_id':manager.active,
                'deployment_schema_version':DEPLOYMENT_SCHEMA_VERSION,
                **topology,'deployment_error':cluster_error,
                'dataset_ready':ready,'datasets':datasets,'deployment':cluster['deployment'],'nodes':nodes,
                'data_preparation':{'state':'preparing' if prepare_lock.locked() else 'idle'},
                'capabilities':{'proofs':True,'proof_suites':['lego_norm_v1'],'threshold':True,'compute':compute,'training_backends':backends,
                                'cloud_strategies':['auto','threshold','all']}}

    @app.get('/api/status')
    def status():
        with manager.lock:
            return status_snapshot()

    def begin_compute():
        from dgfl.crypto.gpu import compute_capabilities, require_gpu
        if manager.active is not None:
            raise HTTPException(409,'运行中不能初始化 GPU 后端')
        if prepare_lock.locked():
            raise HTTPException(409,'数据正在准备，暂时不能初始化 GPU 后端')
        capability=compute_capabilities()['gpu']
        if not capability.get('hardware_available',capability.get('available')):
            raise HTTPException(409,capability.get('reason') or '没有可用的 CUDA 密码后端')
        if capability.get('driver_available') is False or capability.get('compiler_available') is False:
            raise HTTPException(409,capability.get('reason') or 'CUDA 驱动或编译依赖不可用；CPU 计算可用')
        with compute_lock:
            if compute_preparation['state']=='initializing':
                return dict(compute_preparation)
            compute_preparation.update(state='initializing',reason='正在编译并核验 CUDA 密码内核')

        def initialize():
            rpc=None
            outcome='failed'; reason='GPU 初始化未完成'
            try:
                with manager.lock:
                    cluster=json.loads((runtime/'cluster.json').read_text('utf8'))
                    topology=cluster_topology(cluster)
                authorities=[f'authority{i}' for i in range(1,topology['authority_count']+1)]
                if compute_shutdown.is_set():
                    return
                require_gpu()
                rpc=RPCClient(runtime,cluster['nodes'])
                rpc.client.timeout=httpx.Timeout(300,connect=3)
                def one(node):
                    evidence=rpc.call(node,'prepare_compute',{'compute_device':'gpu'},retries=0)
                    if (not isinstance(evidence,dict) or evidence.get('verified') is not True
                            or evidence.get('available') is not True):
                        raise ValueError(f'{node} 的 GPU 密码内核未通过自检')
                with ThreadPoolExecutor(max_workers=min(32,len(authorities))) as pool:
                    list(pool.map(one,authorities))
                outcome='ready'; reason=f'CUDA 密码内核和 {len(authorities)} 个边缘节点已通过自检'
            except Exception as exc:
                reason=str(exc)+'；CPU 计算可用'
            finally:
                if rpc is not None:
                    rpc.close()
                with manager.lock:
                    with compute_lock:
                        compute_preparation.update(state=outcome,reason=reason)
                    invalidate_nodes()

        threading.Thread(target=initialize,name='dgflow-gpu-prepare',daemon=True).start()
        with compute_lock:
            return dict(compute_preparation)

    def automatic_compute_state(state, reason):
        with manager.lock, compute_lock:
            # A manual request can begin while background health RPCs are
            # in flight. Discovery cannot release its admission guard.
            if compute_preparation['state']!='initializing':
                compute_preparation.update(state=state,reason=reason)

    def automatic_compute_attempt():
        """Discover outside admission locks; compile only after roles are healthy."""
        from dgfl.crypto.gpu import compute_capabilities

        with manager.lock:
            with compute_lock:
                if compute_preparation['state']=='initializing':
                    return False
            if manager.active is not None or prepare_lock.locked():
                automatic_compute_state('waiting','当前任务完成后自动核验 GPU；CPU 计算可用')
                return True
        capability=compute_capabilities()['gpu']
        if (not capability.get('hardware_available',capability.get('available'))
                or capability.get('driver_available') is False or capability.get('compiler_available') is False):
            automatic_compute_state('unavailable',(capability.get('reason') or '没有可用的 CUDA 密码后端')+'；CPU 计算可用')
            return False
        with manager.lock:
            if not (runtime/'cluster.json').exists():
                automatic_compute_state('waiting','已识别 GPU，部署边缘节点后自动核验；CPU 计算可用')
                return False
            cluster=json.loads((runtime/'cluster.json').read_text('utf8'))
            topology=cluster_topology(cluster)
        authorities=[f'authority{i}' for i in range(1,topology['authority_count']+1)]
        rpc=RPCClient(runtime,cluster['nodes'])
        rpc.client.timeout=httpx.Timeout(2,connect=.5)
        def one(node):
            try:
                health=rpc.call(node,'health',retries=0)
                return node,health.get('capabilities',{}).get('compute',{}).get('gpu',{})
            except Exception:
                return node,None
        try:
            with ThreadPoolExecutor(max_workers=min(32,len(authorities))) as pool:
                reported=dict(pool.map(one,authorities))
        finally:
            rpc.close()
        if compute_shutdown.is_set():
            return False
        offline=[name for name,item in reported.items() if item is None]
        if offline:
            automatic_compute_state('waiting','等待边缘节点就绪后自动核验 GPU：'+', '.join(offline)+'；CPU 计算可用')
            return True
        unsupported=[name for name,item in reported.items()
                     if (not item.get('hardware_available',item.get('available'))
                         or item.get('driver_available') is False or item.get('compiler_available') is False)]
        if unsupported:
            automatic_compute_state('unavailable','以下边缘节点的 CUDA 驱动或编译依赖不可用：'+', '.join(unsupported)+'；CPU 计算可用')
            return False
        with manager.lock:
            if manager.active is not None or prepare_lock.locked():
                return True
            if json.loads((runtime/'cluster.json').read_text('utf8'))!=cluster:
                return True
            with compute_lock:
                if compute_preparation['state']=='initializing':
                    return False
            if (capability.get('available') is True and capability.get('verified') is True
                    and all(item.get('available') is True and item.get('verified') is True for item in reported.values())):
                automatic_compute_state('ready',f'CUDA 密码内核和 {len(authorities)} 个边缘节点已通过自检')
                return False
            begin_compute()
        return False

    def automatic_compute_worker():
        while not compute_shutdown.is_set():
            requested=compute_wakeup.wait(timeout=GPU_AUTOMATIC_REFRESH_SECONDS)
            compute_wakeup.clear()
            if compute_shutdown.is_set():
                return
            if not requested:
                with compute_lock:
                    if compute_preparation['state']!='waiting':
                        continue
            # Roles spawned immediately before the controller can still be
            # booting. Retry healthy admission, never retry a failed self-test.
            for _ in range(GPU_AUTOMATIC_RETRY_ROUNDS):
                try:
                    retry=automatic_compute_attempt()
                except Exception as exc:
                    automatic_compute_state('failed',str(exc)+'；CPU 计算可用')
                    break
                if not retry or compute_shutdown.wait(GPU_AUTOMATIC_RETRY_SECONDS):
                    break

    @app.post('/api/compute/prepare',status_code=202)
    def prepare_compute():
        with manager.lock:
            return begin_compute()

    @app.post('/api/data/prepare')
    def prepare(config:DataConfig|None=None):
        with manager.lock:
            if manager.active is not None: raise HTTPException(409,'运行中不能修改数据缓存')
            with compute_lock:
                if compute_preparation['state']=='initializing':
                    raise HTTPException(409,'GPU 后端正在初始化，暂时不能准备数据')
            if not prepare_lock.acquire(blocking=False): raise HTTPException(409,'数据正在准备')
        # Keep status polling responsive during the much larger CIFAR download.
        # Admission above and all conflicting actions check this same lock.
        try:
            dataset=config.dataset if config is not None else 'mnist'
            if dataset=='mnist':
                metadata=prepare_mnist(runtime.parent/'data'/'mnist')
            else:
                from dgfl.training.cifar10 import prepare_cifar10
                metadata=prepare_cifar10(runtime.parent/'data'/'cifar10')
            from dgfl.transport.security import atomic_json
            atomic_json(runtime.parent/'data'/dataset/'metadata.json',metadata)
            return {'status':'ready','metadata':metadata}
        except (ValueError,OSError) as exc:
            raise HTTPException(409,str(exc)) from exc
        finally: prepare_lock.release()

    @app.post('/api/runs',status_code=202)
    def start(config:RunConfig):
        with manager.lock:
            if prepare_lock.locked():
                raise HTTPException(409,'数据正在准备，请等待完成后创建实验')
            with compute_lock:
                if compute_preparation['state']=='initializing':
                    raise HTTPException(409,'GPU 后端正在初始化，请等待完成后创建实验')
            try: return manager.start(config.model_dump(exclude_none=True))
            except ValueError as exc: raise HTTPException(409,str(exc)) from exc

    @app.get('/api/proof-parameters')
    def proof_parameters():
        from dgfl.crypto.lego_registry import Registry, available
        try:
            return {'available':available(),'parameters':Registry(runtime).list()}
        except (ValueError,OSError) as exc:
            raise HTTPException(409,str(exc)) from exc

    @app.get('/api/runs')
    def listing():
        with manager.lock:
            return {'runs':[{k:r[k] for k in ('run_id','mode','status','created_at','config','summary') if k in r}
                            | {'mode':r['config']['mode']} for r in reversed(list(manager.records.values()))]}

    @app.get('/api/runs/{run_id}')
    def detail(run_id:str):
        try: return manager.snapshot(run_id,include_history=False)
        except KeyError as exc: raise HTTPException(404,'任务不存在') from exc

    @app.get('/api/runs/{run_id}/monitor')
    def monitor_detail(run_id:str):
        try: return manager.monitor_snapshot(run_id)
        except KeyError as exc: raise HTTPException(404,'任务不存在') from exc

    @app.post('/api/runs/{run_id}/stop')
    def stop(run_id:str):
        try: return manager.stop(run_id)
        except ValueError as exc: raise HTTPException(409,str(exc)) from exc

    @app.get('/api/runs/{run_id}/export')
    def export(run_id:str):
        try: record=manager.snapshot(run_id)
        except KeyError as exc: raise HTTPException(404,'任务不存在') from exc
        return JSONResponse(record,headers={'Content-Disposition':f'attachment; filename="run-{run_id}.json"'})

    @app.get('/api/tensorboard/status')
    def tensorboard_status():
        return tensorboard.status()

    @app.post('/api/runs/{run_id}/tensorboard')
    def tensorboard_sync(run_id:str):
        try: return manager.sync_tensorboard(run_id)
        except KeyError as exc: raise HTTPException(404,'任务不存在') from exc

    app.mount('/tensorboard',tensorboard,name='tensorboard')

    frontend=Path(__file__).resolve().parents[3]/'web'/'dist'
    if (frontend/'assets').exists(): app.mount('/assets',StaticFiles(directory=frontend/'assets'),name='assets')

    @app.get('/{path:path}')
    def index(path:str):
        if path.startswith('api/'): raise HTTPException(404,'未知接口')
        if (frontend/'index.html').exists(): return FileResponse(frontend/'index.html')
        return JSONResponse({'message':'前端尚未构建；请在 web 目录执行 npm ci 与 npm run build'})

    return app
