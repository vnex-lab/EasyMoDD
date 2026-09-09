"""Loss functions with explicit gradient helpers."""

from __future__ import annotations

import numpy as np


class MSELoss:
    """Mean squared error for regression and educational custom loops."""

    def __call__(self, predictions, targets) -> float:
        return float(np.mean((predictions - targets) ** 2))

    def gradient(self, predictions, targets):
        return 2.0 * (predictions - targets) / predictions.size


class CrossEntropyLoss:
    """Stable softmax cross-entropy for class-index targets."""

    def __call__(self, logits, targets) -> float:
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        logsumexp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
        rows = np.arange(len(targets))
        return float(np.mean(-shifted[rows, targets] + logsumexp[:, 0]))
