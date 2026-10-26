"""Reusable training loop contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Any
import numpy as np


@dataclass
class TrainingHistory:
    """Metrics collected by a training run."""

    values: dict[str, list[float]] = field(default_factory=dict)

    def add(self, name: str, value: float) -> None:
        self.values.setdefault(name, []).append(float(value))


class Trainer:
    """Small extensible trainer for models exposing `train_batch`.

    A model can implement its own efficient batch update while the trainer
    still owns iteration, metrics, validation, and callbacks.
    """

    def __init__(self, optimizer=None, loss=None, max_steps: int = 1,
                 batch_size: int = 1, shuffle: bool = True,
                 scheduler=None, callback: Callable | None = None,
                 seed: int = 42):
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.optimizer = optimizer
        self.loss = loss
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.scheduler = scheduler
        self.callback = callback
        self.rng = np.random.default_rng(seed)

    def fit(self, model, dataset: Iterable[Any], validation_data=None) -> TrainingHistory:
        if hasattr(model, "train_batch"):
            return self._fit_batch_model(model, dataset, validation_data)
        if self.optimizer is None or self.loss is None or not hasattr(model, "backward"):
            raise TypeError(
                "Use a model with train_batch(), or provide a model with backward(), "
                "an optimizer, and a loss to Trainer."
            )
        samples = list(dataset)
        if not samples:
            raise ValueError("Training dataset is empty")
        history = TrainingHistory()
        for step in range(1, self.max_steps + 1):
            indices = self.rng.choice(len(samples), size=min(self.batch_size, len(samples)), replace=False) if self.shuffle else np.arange(min(self.batch_size, len(samples)))
            batch = [samples[int(index)] for index in indices]
            if any(not isinstance(item, (tuple, list)) or len(item) != 2 for item in batch):
                raise ValueError("Dataset items must be (inputs, targets) pairs")
            inputs = np.concatenate([np.atleast_2d(item[0]) for item in batch], axis=0)
            targets = np.concatenate([np.atleast_2d(item[1]) for item in batch], axis=0)
            self.optimizer.zero_grad()
            predictions = model(inputs)
            loss_value = self.loss(predictions, targets)
            model.backward(self.loss.gradient(predictions, targets))
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            history.add("loss", float(loss_value))
            if self.callback:
                should_stop = self.callback(step=step, metrics={"loss": float(loss_value)})
                if should_stop:
                    break
        return history

    def _fit_batch_model(self, model, dataset, validation_data):
        samples = list(dataset)
        if not samples:
            raise ValueError("Training dataset is empty")
        history = TrainingHistory()
        for step in range(1, self.max_steps + 1):
            item = samples[(step - 1) % len(samples)]
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("Dataset items must be (inputs, targets) pairs")
            loss = model.train_batch([item[0]], [item[1]])
            if self.scheduler:
                self.scheduler.step()
            history.add("loss", float(loss))
            if self.callback:
                if self.callback(step=step, metrics={"loss": float(loss)}):
                    break
        return history
