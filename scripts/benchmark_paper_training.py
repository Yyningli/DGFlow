"""Offline real-MNIST smoke run of the full paper-sized CNN; no encryption."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from dgfl.training import data as training_data
from dgfl.training import paper_models
from dgfl.training.data import MNIST_FILES, _checked_archive, _idx


def evaluate(model, x, y, device, batch_size):
    loss, correct = 0.0, 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(y), batch_size):
            bx = torch.from_numpy(x[start:start + batch_size]).to(device)
            by = torch.from_numpy(y[start:start + batch_size]).to(device)
            logits = model(bx)
            loss += functional.cross_entropy(logits, by, reduction='sum').item()
            correct += (logits.argmax(dim=1) == by).sum().item()
    return {'cross_entropy': loss / len(y), 'accuracy': correct / len(y),
            'correct': correct, 'total': len(y)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/mnist'))
    parser.add_argument('--train-limit', type=int, default=512)
    parser.add_argument('--test-limit', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    source_paths = [Path(p) for p in (paper_models.__file__, training_data.__file__, __file__)]
    source_hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    if not (1 <= args.train_limit <= 60000 and 1 <= args.test_limit <= 10000
            and args.epochs >= 1 and args.batch_size >= 1 and args.threads >= 1
            and np.isfinite(args.learning_rate) and args.learning_rate > 0):
        parser.error('invalid sample limits or training settings')
    torch.set_num_threads(args.threads)
    model = paper_models.build_model('paper_cnn_mnist', seed=args.seed, device=args.device)
    manifest = paper_models.model_manifest(model)
    raw = args.data_dir / 'raw'
    data_hashes = {}
    for name, expected in MNIST_FILES.items():
        data_hashes[name] = hashlib.sha256(_checked_archive(raw / name, expected)).hexdigest()
    # Original train/test splits, deterministic prefixes, no resize or download.
    x = _idx(raw / 'train-images-idx3-ubyte.gz', True)[:args.train_limit, None].astype(np.float32) / 255
    y = _idx(raw / 'train-labels-idx1-ubyte.gz', False)[:args.train_limit].astype(np.int64)
    tx = _idx(raw / 't10k-images-idx3-ubyte.gz', True)[:args.test_limit, None].astype(np.float32) / 255
    ty = _idx(raw / 't10k-labels-idx1-ubyte.gz', False)[:args.test_limit].astype(np.int64)
    if len(x) != args.train_limit or len(tx) != args.test_limit:
        raise ValueError('cached dataset does not contain requested sample counts')
    before = evaluate(model, tx, ty, args.device, args.batch_size)
    weights = paper_models.flatten_parameters(model)
    if next(model.parameters()).is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    updated = paper_models.train_local(
        'paper_cnn_mnist', weights, architecture_hash=manifest['architecture_hash'],
        x=x, y=y, epochs=args.epochs, learning_rate=args.learning_rate,
        batch_size=args.batch_size, seed=args.seed, device=args.device)
    if str(args.device).startswith('cuda'):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    paper_models.load_parameters(model, updated, architecture_hash=manifest['architecture_hash'])
    after = evaluate(model, tx, ty, args.device, args.batch_size)
    if any(hashlib.sha256(p.read_bytes()).hexdigest()!=source_hashes[p.name] for p in source_paths):
        raise RuntimeError('training sources changed during execution; result not published')
    first = manifest['parameters'][0]
    # The first tensor is the first convolution's weight, not the classifier head.
    first_size = int(np.prod(first['shape']))
    report = {
        'scope': 'single-client full-CNN training smoke; no FE/proof/network/federated run',
        'model': manifest,
        'settings': {k: v for k, v in vars(args).items() if k not in ('data_dir', 'output')},
        'preprocessing': 'original 28x28, NCHW float32, divide by 255; reconstruction assumption',
        'dataset_sha256': data_hashes,
        'before': before, 'after': after, 'training_wall_seconds': elapsed,
        'full_vector_dimension': len(updated),
        'first_convolution_changed': bool(np.any(weights[:first_size] != updated[:first_size])),
        'output_vector_sha256': hashlib.sha256(updated.astype('<f4').tobytes()).hexdigest(),
        'source_sha256': source_hashes,
        'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                        'cuda_build': torch.version.cuda, 'cuda_available': torch.cuda.is_available()},
    }
    if not report['first_convolution_changed'] or not np.isfinite(updated).all():
        raise ValueError('full-model training smoke failed')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('scope', 'full_vector_dimension', 'before', 'after',
                      'training_wall_seconds', 'first_convolution_changed')}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
