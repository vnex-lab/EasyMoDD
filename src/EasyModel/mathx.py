"""Efficient numerical helpers built on NumPy's native vectorized kernels."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class IntegrationEstimate:
    estimate: float
    standard_error: float
    samples: int
    confidence_95: tuple[float, float]


def stable_softmax(values, axis: int = -1):
    values = np.asarray(values)
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=axis, keepdims=True)


def pairwise_distances(left, right, *, squared: bool = False, row_block: int = 2048):
    """BLAS-backed distances with a bounded-size temporary matrix."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("left and right must be 2D arrays with matching feature counts")
    if row_block < 1:
        raise ValueError("row_block must be at least 1")
    right_norms = np.sum(right * right, axis=1)[None, :]
    output = np.empty((left.shape[0], right.shape[0]), dtype=np.float64)
    for start in range(0, left.shape[0], row_block):
        block = left[start:start + row_block]
        distances = np.sum(block * block, axis=1)[:, None] + right_norms - 2.0 * (block @ right.T)
        np.maximum(distances, 0.0, out=distances)
        if not squared:
            np.sqrt(distances, out=distances)
        output[start:start + len(block)] = distances
    return output


def solve_linear(matrix, targets):
    matrix = np.asarray(matrix)
    targets = np.asarray(targets)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    return np.linalg.solve(matrix, targets)


def integrate_simpson(samples, spacing: float = 1.0):
    """Composite Simpson integration for uniformly spaced samples."""
    samples = np.asarray(samples)
    if samples.ndim != 1 or len(samples) < 3 or len(samples) % 2 == 0:
        raise ValueError("Simpson integration requires an odd number of at least 3 one-dimensional samples")
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    return (spacing / 3 * (samples[0] + samples[-1] + 4 * np.sum(samples[1:-1:2]) + 2 * np.sum(samples[2:-1:2]))).item()


def finite_difference_gradient(function, point, *, epsilon: float = 1e-5):
    """Estimate a gradient with central differences and one coordinate at a time."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    point = np.asarray(point, dtype=np.float64)
    gradient = np.empty_like(point)
    for index in range(point.size):
        offset = np.zeros_like(point)
        offset.flat[index] = epsilon
        gradient.flat[index] = (function(point + offset) - function(point - offset)) / (2 * epsilon)
    return gradient


def monte_carlo_integrate(function, bounds, *, samples: int = 100_000,
                          batch_size: int = 8192, seed: int = 0) -> IntegrationEstimate:
    """Integrate a vectorized function with deterministic, bounded-memory sampling.

    `function` receives an array shaped `(batch, dimensions)` and returns one
    real value per point. The estimate includes a normal-approximation 95%
    confidence interval; for rare-event/highly skewed integrands use more
    specialized sampling strategies.
    """
    bounds = np.asarray(bounds, dtype=np.float64)
    if bounds.ndim != 2 or bounds.shape[1] != 2 or len(bounds) == 0:
        raise ValueError("bounds must be a non-empty sequence of (lower, upper) pairs")
    if not np.all(np.isfinite(bounds)) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("each integration bound must have finite lower < upper values")
    if samples < 2 or batch_size < 1:
        raise ValueError("samples must be >= 2 and batch_size must be >= 1")
    rng = np.random.default_rng(seed)
    volume = float(np.prod(bounds[:, 1] - bounds[:, 0]))
    count, mean, m2 = 0, 0.0, 0.0
    while count < samples:
        batch_count = min(batch_size, samples - count)
        points = rng.uniform(bounds[:, 0], bounds[:, 1], size=(batch_count, len(bounds)))
        values = np.asarray(function(points), dtype=np.float64)
        if values.shape != (batch_count,) or not np.all(np.isfinite(values)):
            raise ValueError("function must return one finite real value per input sample")
        batch_mean = float(np.mean(values))
        batch_m2 = float(np.sum((values - batch_mean) ** 2))
        merged_count = count + batch_count
        delta = batch_mean - mean
        m2 += batch_m2 + delta * delta * count * batch_count / merged_count
        mean += delta * batch_count / merged_count
        count = merged_count
    error = (m2 / (count - 1) / count) ** 0.5 * volume
    estimate = mean * volume
    return IntegrationEstimate(estimate, error, count, (estimate - 1.96 * error, estimate + 1.96 * error))