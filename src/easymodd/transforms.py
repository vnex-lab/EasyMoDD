"""Composable array transforms for datasets."""

from __future__ import annotations

import numpy as np


class Compose:
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, value):
        for transform in self.transforms:
            value = transform(value)
        return value


class MapTransform:
    def __init__(self, function):
        self.function = function

    def __call__(self, value):
        return self.function(value)


class ToFloat32:
    def __call__(self, value):
        return value.astype(np.float32, copy=False)


class Flatten:
    def __call__(self, value):
        return value.reshape(-1)


class Normalize:
    def __init__(self, mean, standard_deviation, epsilon: float = 1e-8):
        self.mean = mean
        self.standard_deviation = standard_deviation
        self.epsilon = epsilon

    def __call__(self, value):
        return (value - self.mean) / (self.standard_deviation + self.epsilon)


class Standardize(Normalize):
    """Alias with clearer tabular-data terminology."""
