"""Learning-rate schedules."""

from __future__ import annotations

import math


class CosineScheduler:
    def __init__(self, optimizer, total_steps: int, minimum_rate: float = 0.0):
        if total_steps < 1:
            raise ValueError("total_steps must be at least 1")
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.minimum_rate = minimum_rate
        self.base_rate = optimizer.learning_rate
        self.step_count = 0

    def step(self) -> float:
        self.step_count += 1
        progress = min(self.step_count, self.total_steps) / self.total_steps
        self.optimizer.learning_rate = self.minimum_rate + (self.base_rate - self.minimum_rate) * (1 + math.cos(math.pi * progress)) / 2
        return self.optimizer.learning_rate


class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps: int, target_rate: float | None = None):
        if warmup_steps < 1:
            raise ValueError("warmup_steps must be at least 1")
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.target_rate = target_rate or optimizer.learning_rate
        self.step_count = 0

    def step(self) -> float:
        self.step_count += 1
        self.optimizer.learning_rate = self.target_rate * min(1.0, self.step_count / self.warmup_steps)
        return self.optimizer.learning_rate