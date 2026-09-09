"""Reusable training callbacks."""

from __future__ import annotations

import json
from pathlib import Path


class MetricsLogger:
    """Append one JSON object per training event."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, *, step: int, metrics: dict[str, float]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": step, **metrics}) + "\n")


class EarlyStopping:
    def __init__(self, metric: str = "loss", patience: int = 10, minimum_delta: float = 0.0):
        if patience < 1:
            raise ValueError("patience must be at least 1")
        self.metric = metric
        self.patience = patience
        self.minimum_delta = minimum_delta
        self.best = float("inf")
        self.bad_steps = 0

    def __call__(self, *, step: int, metrics: dict[str, float]) -> bool:
        value = metrics.get(self.metric)
        if value is None:
            return False
        if value < self.best - self.minimum_delta:
            self.best = value
            self.bad_steps = 0
        else:
            self.bad_steps += 1
        return self.bad_steps >= self.patience