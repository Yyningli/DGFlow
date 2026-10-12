"""Pair legacy and compact verification on fresh DKG/proofs with public bases warm.

Only public generator/fixed-base tables persist across samples. Every sample
has a new DKG epoch, context, ciphertext and proof; no GT results are replayed.
This measures proof verification only, without cloud pairings or training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
import uuid
from pathlib import Path

from benchmark_crypto import ROOT, directory_sources, directory_wire_source, make_authorities, snapshot_modules

from dgfl.experiments.runner import implementation_evidence
from dgfl.transport.binary import packb
from dgfl.transport.security import atomic_json


def timed(function):
    wall=time.perf_counter(); cpu=time.process_time()
    result=function()
    return result,{'wall_seconds':time.perf_counter()-wall,'cpu_seconds':time.process_time()-cpu}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir',type=Path,required=True)
    parser.add_argument('--dimension',type=int,default=650)
    parser.add_argument('--bits',type=int,default=8)
    parser.add_argument('--block-size',type=int,default=128)
    parser.add_argument('--samples',type=int,default=3)
    parser.add_argument('--verification',choices=('deterministic','randomized'),default='deterministic')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if sys.flags.optimize: parser.error('optimized Python disables correctness assertions; run without -O')
    if (not 1<=args.dimension<=20000 or not 2<=args.bits<=16
            or not 1<=args.block_size<=1024 or not 1<=args.samples<=20):
        parser.error('unsupported dimensions, bit width, block size or samples')
    output=args.output.resolve(); script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    old_b,old_p,old_hashes=snapshot_modules(directory_sources(args.baseline_dir),
        output.parent/(output.stem+'-baseline'),'_focused_legacy',directory_wire_source(args.baseline_dir))
    new_b,new_p,new_hashes=snapshot_modules(directory_sources(ROOT),
        output.parent/(output.stem+'-current'),'_focused_compact',directory_wire_source(ROOT))
    compact=sys.modules[new_b.__package__+'.compact']
    dimensions={1<<(args.bits*min(args.block_size,args.dimension-start)-1).bit_length()
                for start in range(0,args.dimension,args.block_size)}
    def precompute():
        for n in sorted(dimensions): compact._generators(n)
        for point in (new_b.G,new_b.H): new_b.public_fixed_g1(point,1)
    _,precomputation=timed(precompute)
    print(json.dumps({'public_precomputation':precomputation,'generator_dimensions':sorted(dimensions)}),flush=True)
    samples=[]
    for index in range(args.samples):
        epoch=uuid.uuid4().hex; cid='paired-client'
        ctx={'task_id':'fresh-verification-benchmark','round_id':1,'key_epoch':epoch,
             'model_hash':'ab'*32,'bits':args.bits,'scale':128,'dimension':args.dimension}
        offset=1<<(args.bits-1)
        values=[((j*17+index*7+3)%(2*offset))-offset for j in range(args.dimension)]
        norm=sum(v*v for v in values)
        nodes=make_authorities(old_p,[cid],args.dimension,epoch)
        shares=[node.client_share(cid) for node in nodes]
        key=old_p.recover_client_key(shares,2)
        assert new_p.recover_client_key(shares,2)==key
        contexts={'baseline':ctx,'current':dict(ctx,proof_suite='compact_norm_v1',proof_block_size=args.block_size)}
        protocols={'baseline':old_p,'current':new_p}
        fixtures={}; generation={}
        order=['baseline','current'] if index%2==0 else ['current','baseline']
        for name in order:
            protocol=protocols[name]; context=contexts[name]
            packet=protocol.encrypt(context,cid,values,key)
            proof,timing=timed(lambda:protocol.prove(context,cid,values,key,packet['ciphertext']))
            fixtures[name]=(packet,proof); generation[name]=timing
        verification={}
        for name in order:
            protocol=protocols[name]; context=contexts[name]; packet,proof=fixtures[name]
            def verify(claimed_norm=norm):
                extra={'verification':args.verification} if name=='current' else {}
                return protocol.verify(context,cid,packet['ciphertext'],claimed_norm,proof,key['public'],**extra)
            accepted,timing=timed(verify)
            assert accepted, f'{name} rejected a freshly generated honest proof'
            assert not verify(norm+1), f'{name} accepted a false norm'
            verification[name]=timing
        row={'sample':index+1,'fresh_epoch':epoch,'generation':generation,'verification':verification,
             'proof_bytes':{name:len(packb(proof)) for name,(_,proof) in fixtures.items()},
             'honest_proofs_accepted':True,'modified_norm_rejected':True}
        samples.append(row)
        print(json.dumps(row),flush=True)
    result={'schema_version':1,'scope':'fresh-DKG paired proof generation/verification; public bases warm',
            'settings':{'dimension':args.dimension,'bits':args.bits,'block_size':args.block_size,
                        'samples':args.samples,'current_verification':args.verification},
            'public_precomputation':precomputation,'samples':samples,
            'source_sha256':{'baseline':old_hashes,'current':new_hashes},
            'native_extension_enabled':{'baseline':bool(getattr(old_b,'NATIVE_EXTENSION',False)),
                                        'current':bool(getattr(new_b,'NATIVE_EXTENSION',False))},
            'native':implementation_evidence()['native'],'script_sha256':script_hash,
            'python':sys.version,'platform':platform.platform(),
            'limitations':['synthetic microbenchmark is not an end-to-end speedup',
                           'public precomputation is reported separately; all points are still checked on every verification'],
            'workspace_source_changes_after_snapshot':[path for path,digest in new_hashes.items()
                if not (ROOT/path).is_file() or hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest]}
    result['medians']={stage:{name:{unit:statistics.median(row[stage][name][unit] for row in samples)
                          for unit in ('wall_seconds','cpu_seconds')} for name in ('baseline','current')}
                       for stage in ('generation','verification')}
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=script_hash:
        raise RuntimeError('benchmark script changed during execution; result not published')
    atomic_json(output,result)
    print(json.dumps({'medians':result['medians'],'output':output.name}),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
