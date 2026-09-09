"""Optimizers for EasyMoDD modules."""

from __future__ import annotations


class SGD:
    """Simple gradient descent optimizer over EasyMoDD Parameters."""

    def __init__(self, parameters, learning_rate: float = 1e-3, weight_decay: float = 0.0):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")
        self.parameters = list(parameters)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.grad[...] = 0

    def step(self) -> None:
        for parameter in self.parameters:
            gradient = parameter.grad
            if self.weight_decay:
                gradient = gradient + self.weight_decay * parameter.value
            parameter.value[...] -= self.learning_rate * gradient


class Adam:
    """Adam optimizer with bias correction and optional decoupled decay."""

    def __init__(self, parameters, learning_rate: float = 1e-3,
                 weight_decay: float = 0.0, decoupled_weight_decay: bool = False,
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")
        if not 0 < beta1 < 1 or not 0 < beta2 < 1:
            raise ValueError("beta1 and beta2 must be between zero and one")
        self.parameters = list(parameters)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.decoupled_weight_decay = decoupled_weight_decay
        self.beta1, self.beta2, self.epsilon = beta1, beta2, epsilon
        self.step_count = 0
        self.first_moments = {}
        self.second_moments = {}

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.grad[...] = 0

    def step(self) -> None:
        self.step_count += 1
        for parameter in self.parameters:
            key = id(parameter)
            if key not in self.first_moments:
                xp = type(parameter.value).__module__.startswith("cupy")
                backend = __import__("cupy") if xp else __import__("numpy")
                self.first_moments[key] = backend.zeros_like(parameter.value)
                self.second_moments[key] = backend.zeros_like(parameter.value)
            gradient = parameter.grad
            if self.weight_decay and not self.decoupled_weight_decay:
                gradient = gradient + self.weight_decay * parameter.value
            first = self.first_moments[key] = self.beta1 * self.first_moments[key] + (1 - self.beta1) * gradient
            second = self.second_moments[key] = self.beta2 * self.second_moments[key] + (1 - self.beta2) * gradient ** 2
            first_hat = first / (1 - self.beta1 ** self.step_count)
            second_hat = second / (1 - self.beta2 ** self.step_count)
            parameter.value[...] -= self.learning_rate * first_hat / (second_hat ** 0.5 + self.epsilon)
            if self.weight_decay and self.decoupled_weight_decay:
                parameter.value[...] -= self.learning_rate * self.weight_decay * parameter.value


class AdamW(Adam):
    """Adam with decoupled weight decay."""

    def __init__(self, parameters, learning_rate: float = 1e-3, weight_decay: float = 1e-2, **kwargs):
        super().__init__(parameters, learning_rate, weight_decay, decoupled_weight_decay=True, **kwargs)
