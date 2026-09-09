"""General-purpose multilayer perceptron."""

from __future__ import annotations

from ..nn import Linear, ReLU, Sequential


class MLP(Sequential):
    """Composable feed-forward model for regression and classification."""

    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int,
                 device: str = "cpu"):
        modules = []
        current = input_size
        for hidden_size in hidden_sizes:
            modules.extend([Linear(current, hidden_size, device=device), ReLU()])
            current = hidden_size
        modules.append(Linear(current, output_size, device=device))
        super().__init__(*modules)


def build_mlp(input_size: int, hidden_sizes: list[int], output_size: int,
              device: str = "cpu") -> MLP:
    return MLP(input_size, hidden_sizes, output_size, device=device)
