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

    def gradient(self, logits, targets):
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
        probabilities[np.arange(len(targets)), targets] -= 1
        return probabilities / len(targets)


class L1Loss:
    def __call__(self, predictions, targets) -> float:
        return float(np.mean(np.abs(predictions - targets)))

    def gradient(self, predictions, targets):
        return np.sign(predictions - targets) / predictions.size


class HuberLoss:
    def __init__(self, delta: float = 1.0):
        if delta <= 0:
            raise ValueError("delta must be greater than zero")
        self.delta = delta

    def __call__(self, predictions, targets) -> float:
        error = np.abs(predictions - targets)
        quadratic = np.minimum(error, self.delta)
        linear = error - quadratic
        return float(np.mean(0.5 * quadratic ** 2 + self.delta * linear))

    def gradient(self, predictions, targets):
        error = predictions - targets
        return np.where(np.abs(error) <= self.delta, error, self.delta * np.sign(error)) / predictions.size
