"""Role-specific RPC services over a binary, authenticated envelope.

Bodies larger than one request are delivered as ``_chunk`` calls and reassembled
here before the target action runs (see :mod:`dgfl.transport.chunking`).
"""
import hashlib
import importlib.machinery
import importlib.util
import sys
import threading
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from dgfl.crypto.backend import digest
from dgfl.transport.binary import binary_digest, unpack_from, unpackb
from dgfl.transport.chunking import (
    CHUNK_ACTION,
    DOWNLOAD_ACTION,
    DOWNLOAD_COMPLETE_ACTION,
    DOWNLOAD_MARKER,
    SEND_THRESHOLD,
    ChunkAssembler,
    EncodedMessage,
    ResponseStore,
)
from dgfl.transport.security import Identity, atomic_bytes, read_bytes

from .telemetry import NodeTelemetry

#: Largest single RPC body accepted. Chunked submissions stay below this.
MESSAGE_LIMIT = 64 * 1024 * 1024
CACHE_SUFFIX = '.bin'
TRAIN_MONITOR_FIELDS = frozenset(('training_metrics', 'metrics_eval_s', 'metrics_callback_s'))


def training_backends():
    """Check module presence without importing Torch during health polling.

    An installed module spec is not an import or DLL-load self-test. Training
    still reports a controlled failure if that optional dependency cannot load.
    Both implementations currently train on CPU, independently of crypto CUDA.
    """
    try:
        installed = importlib.util.find_spec('torch') is not None
        reason = '' if installed else '未安装 PyTorch，请在训练客户端安装项目 train 可选依赖'
    except (ImportError, AttributeError, ValueError, OSError):
        installed = False
        reason = '无法检测 PyTorch 安装，请检查训练客户端的 Python 环境'
    return {'numpy': {'installed': True, 'available': True, 'reason': '', 'training_device': 'cpu'},
            'torch': {'installed': installed, 'available': installed, 'reason': reason, 'training_device': 'cpu'}}


@lru_cache(maxsize=1)
def loaded_native_evidence():
    """Public imported-extension file hashes captured at this process's first health.

    This is a file digest tied to actual imported modules, not an in-memory image
    measurement. Retaining the first observation avoids relabelling a running
    process after replacing an installed wheel. No private role state is read.
    """
    artifacts={}; names={}
    for name,module in tuple(sys.modules.items()):
        if name=='dgfl_native' or name.startswith('dgfl_native.'):
            location=getattr(module,'__file__',None)
            if location and any(location.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES):
                path=Path(location)
                if path.is_file():
                    artifacts[name]=hashlib.sha256(path.read_bytes()).hexdigest()
                    names[name]=path.name
    return {'loaded_artifact_sha256':artifacts,'artifact_names':names,
            'evidence_scope':'actual imported extension file hashes captured at first health; not an in-memory image attestation'}


def _reassemble(outcome):
    """Decode a completed chunked submission into (action, payload)."""
    try:
        if 'path' in outcome:
            path = outcome['path']
            try:
                with path.open('rb') as stream:
                    rebuilt = unpack_from(stream)
            finally:
                path.unlink(missing_ok=True)
        else:
            rebuilt = unpackb(outcome['body'])
    except (ValueError, TypeError) as exc:
        raise ValueError('assembled chunked body is not a valid message') from exc
    if not isinstance(rebuilt, dict) or set(rebuilt) != {'action', 'payload'}:
        raise ValueError('assembled chunked body has an invalid schema')
    if rebuilt['action'] != outcome['action']:
        raise ValueError('assembled chunked body declares a different action')
    if not isinstance(rebuilt['payload'], dict):
        raise ValueError('assembled chunked payload must be a mapping')
    return rebuilt['action'], rebuilt['payload']


def create_node_app(node_id, runtime):
    runtime = Path(runtime)
    identity = Identity(runtime / 'keys', node_id)
    lock = threading.Lock()
    folder = runtime / 'nodes' / node_id
    assembler = ChunkAssembler(directory=folder / 'rpc-uploads', file_mode=True)
    responses = ResponseStore(folder / 'rpc-downloads')
    telemetry = NodeTelemetry(node_id, runtime)

    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            telemetry.close()
            with lock:
                assembler.close()
                responses.cleanup_expired()
                if worker is not None and callable(getattr(worker, 'close', None)):
                    worker.close()

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.chunk_assembler = assembler
    app.state.response_store = responses
    app.state.telemetry = telemetry
    worker = None

    def encode_reply(result, context, fingerprint):
        budget = responses.encoding_budget()
        if budget <= 0:
            raise ValueError('response transfers exceed the configured disk capacity')
        with EncodedMessage(result, directory=responses.directory, disk=True,
                            max_total_bytes=budget) as encoded:
            if encoded.size <= SEND_THRESHOLD:
                prepared = encoded.canonical_payload(max_bytes=SEND_THRESHOLD)
                return identity.seal_encoded('coordinator', 'rpc-response', prepared, context), {'result': prepared}
            descriptor = responses.publish(encoded, context, fingerprint)
            response = identity.seal('coordinator', 'rpc-response', {DOWNLOAD_MARKER: descriptor}, context)
            return response, {'response_token': descriptor['token']}

    def replay_reply(saved, context, fingerprint):
        if 'response_token' in saved:
            descriptor = responses.descriptor(saved['response_token'], context, fingerprint)
            return identity.seal('coordinator', 'rpc-response', {DOWNLOAD_MARKER: descriptor}, context)
        return identity.seal('coordinator', 'rpc-response', saved['result'], context)

    def execute(blob):
        nonlocal worker
        try:
            opened = identity.open(blob, 'rpc', sender='coordinator')
        except ValueError as exc:
            raise HTTPException(403, 'invalid node identity') from exc
        context = opened['context']
        sender = opened['sender']
        data = opened['payload']
        if (set(data) != {'action', 'payload'} or not isinstance(data['payload'], dict)
                or not isinstance(data['action'], str) or not data['action']):
            raise HTTPException(400, 'invalid RPC schema')
        if data['action'] == 'health':
            from dgfl.crypto import backend as b
            from dgfl.crypto.gpu import compute_capabilities
            from dgfl.crypto.lego_registry import available
            result = {'node_id': node_id, 'status': 'online', 'backend': 'BLS12-381/arkworks',
                      'capabilities': {'lego_norm_v1': available(), 'compute':compute_capabilities(),
                                       'datasets':['mnist','cifar10'],
                                       'node_telemetry':True,
                                       'training_backends':training_backends(),
                                       'embedded_aggregate_certificates':node_id.startswith('authority'),
                                       'compact_aggregate_materials':node_id.startswith(('authority','aggregator')),
                                       'ciphertext_only_aggregation':node_id.startswith('aggregator'),
                                       'aggregate_verification_native':bool(node_id.startswith('authority')
                                           and b.NATIVE_EXTENSION and b.PublicAggregateVerifier is not None),
                                       'batched_authorization':node_id.startswith('authority'),
                                       'owned_validation':node_id.startswith('authority'),
                                       'batched_client_keys':node_id.startswith('authority')},
                      'native':loaded_native_evidence()}
            return identity.seal('coordinator', 'rpc-response', result, context)
        if data['action'] in ('telemetry_start', 'telemetry_read', 'telemetry_stop'):
            # Authentication above is shared with role operations; telemetry
            # uses its own cache lock and never waits for stateful execution or
            # creates durable RPC execution guards.
            try:
                operation = {'telemetry_start': telemetry.start,
                             'telemetry_read': telemetry.read,
                             'telemetry_stop': telemetry.stop}[data['action']]
                result = operation(data['payload'])
                return identity.seal('coordinator', 'rpc-response', result, context)
            except (ValueError, KeyError, TypeError) as exc:
                raise HTTPException(409, str(exc)) from exc
        fingerprint = binary_digest(data)
        path = runtime / 'nodes' / node_id / 'rpc-cache' / (digest(context) + CACHE_SUFFIX)
        with lock:
            if data['action'] in (DOWNLOAD_ACTION, DOWNLOAD_COMPLETE_ACTION):
                try:
                    operation = responses.fetch if data['action'] == DOWNLOAD_ACTION else responses.complete
                    return identity.seal('coordinator', 'rpc-response', operation(data['payload']), context)
                except (ValueError, KeyError, TypeError, FileNotFoundError) as exc:
                    raise HTTPException(409, str(exc)) from exc
            if path.exists():
                saved = read_bytes(path)
                if saved['fingerprint'] != fingerprint:
                    raise HTTPException(409, 'request id reused for different content')
                if 'error' in saved:
                    raise HTTPException(409, saved['error'])
                if 'response' in saved:
                    # Replaying an intermediate block after a service restart
                    # must restore its bytes, rather than merely repeat an ACK.
                    if data['action'] == CHUNK_ACTION and saved.get('interim'):
                        logical = folder / 'rpc-submissions' / (data['payload']['digest'] + CACHE_SUFFIX)
                        if not logical.exists():
                            try:
                                outcome = assembler.accept(sender, data['payload'])
                            except ValueError as exc:
                                raise HTTPException(409, str(exc)) from exc
                            # This request originally returned an ACK. Keep its
                            # exact response; a new logical retry will resend all
                            # blocks and execute the action.
                            if outcome['complete'] and 'path' in outcome:
                                outcome['path'].unlink(missing_ok=True)
                    if 'response_token' in saved:
                        try:
                            responses.descriptor(saved['response_token'], context, fingerprint)
                        except ValueError as exc:
                            raise HTTPException(409, str(exc)) from exc
                    return saved['response']
                raise HTTPException(409, 'previous RPC execution did not complete; action will not be repeated')
            try:
                action, payload = data['action'], data['payload']
                logical_path = None
                block_hashes = None
                if action == CHUNK_ACTION:
                    assembler._spec(payload)
                    logical_path = folder / 'rpc-submissions' / (payload['digest'] + CACHE_SUFFIX)
                    if logical_path.exists():
                        completed = read_bytes(logical_path)
                        if (completed['action'] != payload['action'] or completed['total'] != payload['total']
                                or hashlib.sha256(payload['data']).hexdigest() != completed['block_hashes'][payload['index']]):
                            raise ValueError('chunk does not match the completed submission')
                        if completed.get('pending') or 'error' in completed:
                            raise ValueError(completed.get('error', 'previous submission execution did not complete; action will not be repeated'))
                        if payload['index'] == payload['total'] - 1:
                            response = replay_reply(completed, context, fingerprint)
                            atomic_bytes(path, {'fingerprint': fingerprint, 'response': response,
                                                **({'response_token': completed['response_token']} if 'response_token' in completed else {})})
                            return response
                        interim = {'chunk': payload['index'], 'received': payload['total'], 'total': payload['total']}
                        response = identity.seal('coordinator', 'rpc-response', interim, context)
                        atomic_bytes(path, {'fingerprint': fingerprint, 'response': response})
                        return response
                    outcome = assembler.accept(sender, payload)
                    if not outcome['complete']:
                        # Intermediate acknowledgement; one per block index.
                        interim = {'chunk': payload['index'], 'received': outcome['received'],
                                   'total': outcome['total']}
                        response = identity.seal('coordinator', 'rpc-response', interim, context)
                        atomic_bytes(path, {'fingerprint': fingerprint, 'response': response, 'interim': True})
                        return response
                    block_hashes = outcome['block_hashes']
                    action, payload = _reassemble(outcome)
                elif action.startswith('_'):
                    raise ValueError('unknown reserved RPC action')
                if worker is None:
                    from .roles import RoleWorker
                    worker = RoleWorker(node_id, runtime, identity, telemetry)
                # Persist an execution guard before calling a stateful role.
                # A interrupted response cannot cause an exact retry to run it
                # twice, including failures while encoding a large reply.
                atomic_bytes(path, {'fingerprint': fingerprint, 'pending': True})
                if logical_path is not None:
                    atomic_bytes(logical_path, {'action': action, 'total': data['payload']['total'],
                                               'block_hashes': block_hashes, 'pending': True})
                try:
                    role_context = payload.get('context', {})
                    if isinstance(role_context, dict):
                        telemetry.set_context(role_context.get('task_id'), role_context.get('round_id'), action)
                    result = worker.execute(action, payload)
                except ImportError as exc:
                    if action == 'train' and payload.get('backend') == 'torch':
                        # Do not expose import internals, filesystem paths, or a
                        # generic 500 when the explicitly selected backend fails.
                        raise ValueError('PyTorch 训练依赖无法加载，请检查客户端的 PyTorch 安装') from exc
                    raise
                if action == 'train' and isinstance(result, dict):
                    # Epochs and observation overhead travel through the
                    # independent telemetry RPC. Keep the business wire and
                    # its exact byte counter free of monitoring additions;
                    # direct RoleWorker callers retain their local diagnostics.
                    result = {key: value for key, value in result.items() if key not in TRAIN_MONITOR_FIELDS}
                response, result_record = encode_reply(result, context, fingerprint)
                if logical_path is not None:
                    atomic_bytes(logical_path, {'action': action, 'total': data['payload']['total'],
                                               'block_hashes': block_hashes, **result_record})
                atomic_bytes(path, {'fingerprint': fingerprint, 'response': response,
                                    **({'response_token': result_record['response_token']} if 'response_token' in result_record else {})})
                return response
            except (ValueError, KeyError, TypeError, FileNotFoundError) as exc:
                if path.exists() and read_bytes(path).get('pending'):
                    atomic_bytes(path, {'fingerprint': fingerprint, 'error': str(exc)})
                if logical_path is not None and logical_path.exists() and read_bytes(logical_path).get('pending'):
                    atomic_bytes(logical_path, {'action': action, 'total': data['payload']['total'],
                                               'block_hashes': block_hashes, 'error': str(exc)})
                raise HTTPException(409, str(exc)) from exc

    @app.post('/rpc')
    async def rpc(request: Request):
        payload = bytearray()
        async for chunk in request.stream():
            if len(payload) + len(chunk) > MESSAGE_LIMIT:
                raise HTTPException(413, 'node message too large')
            payload.extend(chunk)
        body = await run_in_threadpool(execute, bytes(payload))
        return Response(content=body, media_type='application/octet-stream')

    return app
