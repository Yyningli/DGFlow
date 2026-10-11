"""A linear softmax classifier with real mini-batch SGD.

The wire layout is W[features, 10] flattened in C order, followed by b[10].
``features=64`` (an 8x8 pooling grid) is the original 650-parameter model;
``features=784`` (the full 28x28 grid) gives 7,850 parameters. All public
vectors are float64. Neither backend changes caller-owned arrays.
The optional torch backend deliberately uses CPU/float64 for reproducibility.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from numbers import Integral

import numpy as np

FEATURES = 64
CLASSES = 10
PARAMETERS = FEATURES * CLASSES + CLASSES
_LOG = logging.getLogger(__name__)


def parameter_count(features: int = FEATURES) -> int:
    """Number of coordinates in the flattened model for a given feature count."""
    if type(features) is not int or features < 1:
        raise ValueError("features must be a positive integer")
    return features * CLASSES + CLASSES


def _weights(weights: np.ndarray, features: int = FEATURES) -> np.ndarray:
    total = parameter_count(features)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (total,) or not np.isfinite(w).all():
        raise ValueError(f"weights must be a finite vector of {total} parameters")
    return w.copy()


def _data(x: np.ndarray, y: np.ndarray, features: int = FEATURES) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    if x.ndim != 2 or x.shape[1] != features or len(x) == 0:
        raise ValueError(f"x must contain at least one row of {features} features")
    if y.shape != (len(x),) or y.dtype.kind not in "iu":
        raise ValueError("y must be an integer vector matching x")
    if not np.isfinite(x).all() or np.any(y < 0) or np.any(y >= CLASSES):
        raise ValueError("features must be finite and labels must be in [0, 9]")
    return x, y.astype(np.int64, copy=False)


def initial_model(seed: int = 42, features: int = FEATURES) -> np.ndarray:
    """Return a small nonzero initialization without mutating global RNG state."""
    return np.random.default_rng(seed).normal(0.0, 0.01, parameter_count(features)).astype(np.float64)


def train_local(
    weights: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    features: int = FEATURES,
    epochs: int = 1,
    learning_rate: float = 0.1,
    batch_size: int = 64,
    seed: int = 42,
    backend: str = "numpy",
    on_epoch: Callable[[dict], None] | None = None,
) -> np.ndarray:
    """Minimize average softmax cross-entropy with deterministic shuffled SGD.

Returns a complete local model, not a delta. No network, data downloading,
implicit backend fallback, momentum, weight decay, or hidden quantization occurs.
When requested, epoch diagnostics evaluate this client's actual training data.
These diagnostics never expose model coordinates or samples to the callback;
diagnostic failures do not change or repeat optimizer steps.
"""
    w = _weights(weights, features)
    x, y = _data(x, y, features)
    if not isinstance(epochs, Integral) or isinstance(epochs, bool) or epochs < 0:
        raise ValueError("epochs must be a nonnegative integer")
    if not isinstance(batch_size, Integral) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if backend not in ("numpy", "torch"):
        raise ValueError("backend must be 'numpy' or 'torch'")
    rng = np.random.default_rng(seed)
    if backend == "torch":
        return _train_torch(w, x, y, features, epochs, learning_rate, batch_size, rng, on_epoch)
    split = features * CLASSES
    matrix, bias = w[:split].reshape(features, CLASSES), w[split:]
    batch_rows = np.arange(min(batch_size, len(y)))
    for epoch in range(epochs):
        started = time.perf_counter() if on_epoch is not None else None
        order = rng.permutation(len(y))
        for start in range(0, len(y), batch_size):
            indices = order[start:start + batch_size]
            bx, by = x[indices], y[indices]
            logits = bx @ matrix + bias
            logits -= logits.max(axis=1, keepdims=True)
            # No later operation needs logits; keep the identical exp arithmetic
            # while reusing the batch buffer instead of allocating another one.
            error = np.exp(logits, out=logits)
            error /= error.sum(axis=1, keepdims=True)
            error[batch_rows[:len(by)], by] -= 1
            error /= len(by)
            matrix -= learning_rate * (bx.T @ error)
            bias -= learning_rate * error.sum(axis=0)
        if on_epoch is not None:
            elapsed = time.perf_counter() - started
            _epoch_metrics(on_epoch, lambda: w, x, y, features, epoch + 1, elapsed)
    if not np.isfinite(w).all():
        raise ValueError("training produced nonfinite parameters; reduce learning_rate or feature scale")
    return w


def _epoch_metrics(on_epoch, weights, x, y, features, epoch, elapsed):
    """Keep observational evaluation outside the optimizer's failure boundary."""
    started = time.perf_counter()
    try:
        metrics = evaluate(weights(), x, y, features=features)
    except Exception as exc:
        _LOG.warning("Local epoch evaluation unavailable (%s)", type(exc).__name__)
        metrics = {"samples": len(y), "metrics_error": "local epoch evaluation unavailable"}
    metrics.update(epoch=int(epoch), elapsed_s=float(elapsed),
                   metrics_eval_s=time.perf_counter() - started)
    try:
        on_epoch(metrics)
    except Exception as exc:
        _LOG.warning("Local epoch monitoring callback unavailable (%s)", type(exc).__name__)


def _train_torch(w, x, y, features, epochs, learning_rate, batch_size, rng, on_epoch=None):
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise ImportError("backend='torch' requires the optional torch dependency") from exc
    split = features * CLASSES
    tensor = torch.tensor(w, dtype=torch.float64, requires_grad=True, device="cpu")
    features_t = torch.tensor(x, dtype=torch.float64, device="cpu")
    labels = torch.tensor(y, dtype=torch.int64, device="cpu")
    for epoch in range(epochs):
        started = time.perf_counter() if on_epoch is not None else None
        # One conversion per epoch retains the NumPy shuffle and exact batches.
        order = torch.from_numpy(rng.permutation(len(y)))
        for start in range(0, len(y), batch_size):
            indices = order[start:start + batch_size]
            logits = features_t[indices] @ tensor[:split].reshape(features, CLASSES) + tensor[split:]
            loss = functional.cross_entropy(logits, labels[indices])
            loss.backward()
            with torch.no_grad():
                tensor -= learning_rate * tensor.grad
            tensor.grad = None
        if on_epoch is not None:
            elapsed = time.perf_counter() - started
            _epoch_metrics(on_epoch, lambda: tensor.detach().numpy().copy(),
                           x, y, features, epoch + 1, elapsed)
    result = tensor.detach().numpy().copy()
    if not np.isfinite(result).all():
        raise ValueError("training produced nonfinite parameters; reduce learning_rate or feature scale")
    return result


def evaluate(weights, x, y, *, features: int = FEATURES) -> dict:
    """Compute stable mean cross-entropy and top-1 accuracy on supplied data."""
    w = _weights(weights, features)
    x, y = _data(x, y, features)
    split = features * CLASSES
    logits = x @ w[:split].reshape(features, CLASSES) + w[split:]
    if not np.isfinite(logits).all():
        raise ValueError("evaluation produced nonfinite logits")
    shifted = logits - logits.max(axis=1, keepdims=True)
    loss = np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(y)), y]
    return {"accuracy": float(np.mean(logits.argmax(axis=1) == y)),
            "loss": float(loss.mean()), "samples": len(y)}


def average_models(models: Sequence[np.ndarray], sample_counts=None, *, features: int = FEATURES) -> np.ndarray:
    """Plaintext reference mean, optionally sample-weighted; no secure aggregation."""
    if len(models) == 0:
        raise ValueError("at least one model is required")
    stacked = np.stack([_weights(w, features) for w in models])
    if sample_counts is None:
        return stacked.mean(axis=0)
    counts = np.asarray(sample_counts, dtype=np.float64)
    if counts.shape != (len(models),) or not np.isfinite(counts).all() or np.any(counts <= 0):
        raise ValueError("sample_counts must be finite positive counts matching models")
    scaled = counts / counts.max()
    return np.sum(stacked * (scaled / scaled.sum())[:, None], axis=0)


def attack_weights(weights, attack: str, seed: int = 42, *, features: int = FEATURES) -> np.ndarray:
    """Explicit reproducible model-vector attacks for experiments.

For free_rider, pass the round's initial/global model: it is returned unchanged,
representing no local learning. Passing a trained model cannot undo training.
sign_flip negates the supplied whole model (not its delta); random draws an
independent N(0,1) vector. These are simple attacks, not adaptive/backdoor tests.
"""
    w = _weights(weights, features)
    if attack in ("none", "free_rider"):
        return w
    if attack == "sign_flip":
        return -w
    if attack == "random":
        return np.random.default_rng(seed).normal(size=parameter_count(features))
    raise ValueError("attack must be none, sign_flip, random, or free_rider")
