"""Fresh full 650-coordinate proof benchmark, including ciphertext/key links.

An isolated native wheel can be selected with --native-dir. Development trusted
setup and public-key preparation are explicitly reported outside proof timing.
All samples use fresh cryptographic keys, ciphertext contexts and proof randomness.
Output contains only settings, public code hashes, timings and correctness results.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import statistics
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def timed(function):
    start = time.perf_counter(); cpu = time.process_time()
    value = function()
    return value, {'wall_ms': (time.perf_counter()-start)*1000,
                   'cpu_ms': (time.process_time()-cpu)*1000}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-dir', type=Path)
    parser.add_argument('--baseline-dir', type=Path, default=ROOT/'tmp/proof-ms-baseline')
    parser.add_argument('--dimension', type=int, default=650)
    parser.add_argument('--bits', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--skip-baseline', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT/'tmp/lego-proof-benchmark.json')
    args = parser.parse_args(argv)
    if sys.flags.optimize:
        parser.error('run without -O: benchmark acceptance assertions are required')
    if not 1 <= args.samples <= 20 or not 1 <= args.workers <= 4:
        parser.error('invalid sample or worker count')
    if args.native_dir:
        sys.path.insert(0, str(args.native_dir.resolve()))
    sys.path.insert(0, str(ROOT/'src'))
    import dgfl_native
    from benchmark_crypto import directory_sources, directory_wire_source, snapshot_modules

    from dgfl.crypto import backend as b
    from dgfl.crypto import lego
    from dgfl.crypto import protocol as p
    from dgfl.transport.binary import packb

    sources = ['src/dgfl/crypto/'+name+'.py' for name in ('backend', 'ipa', 'compact', 'protocol', 'lego')]
    sources += ['native/dgfl-native/src/'+name+'.rs' for name in ('lib', 'lego', 'sigma')]
    sources += ['native/dgfl-native/Cargo.toml', 'native/dgfl-native/Cargo.lock',
                'native/dgfl-native/pyproject.toml', 'scripts/benchmark_lego_proof.py',
                'tests/crypto/test_lego.py']
    hashes = {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in sources}
    native_path = Path(dgfl_native.__file__).resolve()
    artifacts = [native_path] if native_path.suffix == '.pyd' else list(native_path.parent.glob('*.pyd'))
    if not artifacts:
        raise RuntimeError('loaded native artifact is missing')
    baseline = None
    if not args.skip_baseline:
        old_b, old_p, baseline_hashes = snapshot_modules(directory_sources(args.baseline_dir),
            args.output.parent/(args.output.stem+'-baseline'), '_proof_ms_baseline',
            directory_wire_source(args.baseline_dir))
        old_compact = sys.modules[old_b.__package__+'.compact']
        for start in range(0, args.dimension, 128):
            old_compact._generators(1 << (args.bits*min(128,args.dimension-start)-1).bit_length())
        for point in (old_b.G,old_b.H): old_b.public_fixed_g1(point,1)
        baseline = old_p
    (prover, params), setup = timed(lambda: lego.development_setup(args.dimension,args.bits,workers=args.workers))
    print(json.dumps({'development_setup': setup, 'crs_hash': params.crs_hash}), flush=True)
    rows = []
    for sample in range(args.samples):
        cid = 'proof-ms-client'
        ctx = dict(task_id='proof-ms-benchmark', round_id=sample+1, key_epoch=uuid.uuid4().hex,
                   model_hash='ab'*32, bits=args.bits, scale=128, dimension=args.dimension)
        offset = 1 << (args.bits-1)
        values = [((i*17+sample*7+3) % (2*offset))-offset for i in range(args.dimension)]
        key = dict(client_id=cid, epoch=ctx['key_epoch'], s=[b.random_scalar() for _ in values],
                   r=[b.random_scalar() for _ in values])
        key['public'] = [b.g1_dump(b.G*b.scalar(s)+b.H*b.scalar(r)) for s,r in zip(key['s'],key['r'])]
        contexts = {'lego':lego.context(ctx,params),
                    'baseline':dict(ctx,proof_suite='compact_norm_v1',proof_block_size=128)}
        order = ['lego', 'baseline'] if sample % 2 else ['baseline', 'lego']
        generation = {}; validation = {}; sizes = {}; packets = {}; proofs = {}
        for name in order:
            if name == 'baseline' and baseline is None: continue
            packet = p.encrypt(contexts[name],cid,values,key)
            operation = (lambda: lego.prove(contexts[name],cid,values,key,packet['ciphertext'],
                                           prover,params,workers=args.workers)) if name == 'lego' else (
                         lambda: baseline.prove(contexts[name],cid,values,key,packet['ciphertext']))
            proof, generation[name] = timed(operation)
            sizes[name] = len(packb(proof)); packets[name] = packet; proofs[name] = proof
        for name in order:
            if name not in proofs: continue
            validation[name] = {}
            for mode in ('deterministic','randomized'):
                packet,proof=packets[name],proofs[name]
                verify = (lambda norm: lego.verify(contexts[name],cid,packet['ciphertext'],norm,proof,
                                                  key['public'],params,verification=mode)) if name == 'lego' else (
                          lambda norm: baseline.verify(contexts[name],cid,packet['ciphertext'],norm,proof,
                                                       key['public'],verification=mode))
                accepted, validation[name][mode] = timed(lambda: verify(packet['norm_squared']))
                assert accepted, f'{name}/{mode} rejected a fresh honest full proof'
                assert not verify(packet['norm_squared']+1), f'{name}/{mode} accepted incorrect norm'
        row = {'sample': sample+1, 'generation': generation, 'verification':validation,
               'proof_bytes': sizes, 'honest_full_proofs_accepted':True, 'false_norm_rejected':True}
        rows.append(row); print(json.dumps(row), flush=True)
    medians = {'generation':{name:statistics.median(row['generation'][name]['wall_ms'] for row in rows)
                            for name in rows[0]['generation']},
               'verification':{name:{mode:statistics.median(row['verification'][name][mode]['wall_ms'] for row in rows)
                                      for mode in ('deterministic','randomized')}
                               for name in rows[0]['verification']}}
    cpu_medians = {'generation':{name:statistics.median(row['generation'][name]['cpu_ms'] for row in rows)
                                for name in rows[0]['generation']},
                   'verification':{name:{mode:statistics.median(row['verification'][name][mode]['cpu_ms'] for row in rows)
                                          for mode in ('deterministic','randomized')}
                                   for name in rows[0]['verification']}}
    max_prove = max(row['generation']['lego']['wall_ms'] for row in rows)
    max_verify = max(row['verification']['lego']['randomized']['wall_ms'] for row in rows)
    max_deterministic = max(row['verification']['lego']['deterministic']['wall_ms'] for row in rows)
    changed = [path for path,digest in hashes.items() if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != digest]
    if changed: raise RuntimeError('source changed during benchmark: '+', '.join(changed))
    result = {'scope':'FULL proof generation/checked verification including range, squared norm, ciphertext and registered-key bindings',
        'settings':{'dimension':args.dimension,'bits':args.bits,'workers':args.workers,'samples':args.samples},
        'setup':setup,'setup_kind':'local development trusted setup; not a ceremony',
        'crs_hash':params.crs_hash,'samples':rows,'medians_wall_ms':medians,'medians_cpu_ms':cpu_medians,
        'target':{'generation_and_both_verification_modes_under_1000ms_all_samples':
                  max(max_prove,max_verify,max_deterministic)<1000,
                  'max_generation_ms':max_prove,'max_randomized_verification_ms':max_verify,
                  'max_deterministic_verification_ms':max_deterministic},
        'source_sha256':hashes,'baseline_sha256':baseline_hashes if baseline is not None else None,
        'loaded_native_artifacts':{path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name:
                                  hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts},
        'python':sys.version,'platform':platform.platform(),
        'curve_package_version':importlib.metadata.version('py-arkworks-bls12381'),
        'limitations':['local synthetic benchmark excludes DKG/training/RPC',
                      'CRS and public bases prepared offline and retained; every proof/context/key is fresh',
                      'experimental composition has not received an external soundness audit',
                      'trusted setup required; existing task policy does not deploy this backend']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'medians_wall_ms':medians,'target':result['target'],'output':str(args.output)}),flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
