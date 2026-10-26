"""Training metrics that work with ordinary array-like predictions."""

from __future__ import annotations

import numpy as np


def accuracy(predictions, targets) -> float:
    predicted = np.argmax(predictions, axis=-1) if getattr(predictions, "ndim", 0) > 1 else np.rint(predictions)
    return float(np.mean(predicted == targets))


def mean_absolute_error(predictions, targets) -> float:
    return float(np.mean(np.abs(predictions - targets)))


def perplexity(cross_entropy: float) -> float:
    return float(np.exp(np.clip(cross_entropy, -50, 50)))