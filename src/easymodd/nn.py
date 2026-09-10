"""Composable model building blocks for EasyMoDD."""

from __future__ import annotations

from collections import OrderedDict
import numpy as np

from .tensor import backend_of, randn, zeros


class Parameter:
    """A trainable array owned by a module."""

    def __init__(self, value):
        self.value = value
        self.grad = backend_of(value).zeros_like(value)


class Module:
    """Base class for composable EasyMoDD models."""

    def __init__(self) -> None:
        self._modules: OrderedDict[str, Module] = OrderedDict()
        self._parameters: OrderedDict[str, Parameter] = OrderedDict()
        self.training = True

    def add_module(self, name: str, module: "Module") -> None:
        self._modules[name] = module
        setattr(self, name, module)

    def register_parameter(self, name: str, parameter: Parameter) -> None:
        self._parameters[name] = parameter
        setattr(self, name, parameter)

    def parameters(self):
        for parameter in self._parameters.values():
            yield parameter
        for module in self._modules.values():
            yield from module.parameters()

    def named_parameters(self, prefix: str = ""):
        for name, parameter in self._parameters.items():
            yield (f"{prefix}.{name}" if prefix else name), parameter
        for name, module in self._modules.items():
            child_prefix = f"{prefix}.{name}" if prefix else name
            yield from module.named_parameters(child_prefix)

    def state_dict(self) -> dict[str, object]:
        state = {}
        for name, parameter in self._parameters.items():
            state[name] = parameter.value.copy()
        for name, module in self._modules.items():
            for child_name, value in module.state_dict().items():
                state[f"{name}.{child_name}"] = value
        return state

    def load_state_dict(self, state: dict[str, object], *, strict: bool = True) -> None:
        expected = dict(self.named_parameters())
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        if strict and (missing or unexpected):
            raise ValueError(f"State mismatch: missing={missing}, unexpected={unexpected}")
        for name, value in state.items():
            if name not in expected:
                continue
            parameter = expected[name]
            if parameter.value.shape != value.shape:
                raise ValueError(f"Shape mismatch for '{name}': expected {parameter.value.shape}, got {value.shape}")
            parameter.value[...] = value

    def train(self, mode: bool = True):
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def __call__(self, inputs):
        return self.forward(inputs)

    def backward(self, gradient):
        raise NotImplementedError("This module does not implement backward()")

    def forward(self, inputs):
        raise NotImplementedError("Model modules must implement forward()")


class Linear(Module):
    def __init__(self, input_size: int, output_size: int, device: str = "cpu") -> None:
        super().__init__()
        scale = np.sqrt(2.0 / max(input_size, 1))
        self.register_parameter("weight", Parameter(randn((input_size, output_size), device) * scale))
        self.register_parameter("bias", Parameter(zeros((1, output_size), device)))

    def forward(self, inputs):
        self._inputs = inputs
        return inputs @ self.weight.value + self.bias.value

    def backward(self, gradient):
        self.weight.grad[...] += self._inputs.T @ gradient
        self.bias.grad[...] += backend_of(gradient).sum(gradient, axis=0, keepdims=True)
        return gradient @ self.weight.value.T


class ReLU(Module):
    def forward(self, inputs):
        self._mask = inputs > 0
        return backend_of(inputs).maximum(inputs, 0)

    def backward(self, gradient):
        return gradient * self._mask


class GELU(Module):
    def forward(self, inputs):
        return inputs * 0.5 * (1 + np.tanh(0.7978845608 * (inputs + 0.044715 * inputs ** 3)))


class Sigmoid(Module):
    def forward(self, inputs):
        self._output = 1 / (1 + np.exp(-np.clip(inputs, -60, 60)))
        return self._output


class Tanh(Module):
    def forward(self, inputs):
        self._output = np.tanh(inputs)
        return self._output


class Dropout(Module):
    def __init__(self, probability: float = 0.1):
        super().__init__()
        if not 0 <= probability < 1:
            raise ValueError("dropout probability must be in [0, 1)")
        self.probability = probability

    def forward(self, inputs):
        if not self.training or self.probability == 0:
            self._mask = None
            return inputs
        self._mask = (np.random.random(inputs.shape) >= self.probability) / (1 - self.probability)
        return inputs * self._mask

    def backward(self, gradient):
        return gradient if self._mask is None else gradient * self._mask


class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        for index, module in enumerate(modules):
            self.add_module(str(index), module)

    def forward(self, inputs):
        for module in self._modules.values():
            inputs = module(inputs)
        return inputs

    def backward(self, gradient):
        for module in reversed(list(self._modules.values())):
            gradient = module.backward(gradient)
        return gradient
