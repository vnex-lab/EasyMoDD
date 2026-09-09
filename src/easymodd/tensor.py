"""Small explicit NumPy/CUDA backend used by EasyMoDD modules."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Device:
    """Execution device: `cpu` or `cuda`."""

    name: str = "cpu"

    def __post_init__(self) -> None:
        if self.name not in {"cpu", "cuda"}:
            raise ValueError("Device must be 'cpu' or 'cuda'")


def get_backend(device: str | Device = "cpu"):
    selected = device.name if isinstance(device, Device) else device
    if selected == "cpu":
        return np
    if selected != "cuda":
        raise ValueError("Device must be 'cpu' or 'cuda'")
    try:
        import cupy
        cupy.cuda.runtime.getDeviceCount()
        return cupy
    except Exception as error:
        raise RuntimeError(
            "CUDA was requested but CuPy/CUDA could not initialize. "
            "Install a compatible CuPy package or set device = 'cpu'."
        ) from error


def seed(value: int, device: str | Device = "cpu") -> None:
    get_backend(device).random.seed(value)


def array(value, device: str | Device = "cpu", dtype=None):
    return get_backend(device).asarray(value, dtype=dtype)


def zeros(shape, device: str | Device = "cpu", dtype=np.float32):
    return get_backend(device).zeros(shape, dtype=dtype)


def ones(shape, device: str | Device = "cpu", dtype=np.float32):
    return get_backend(device).ones(shape, dtype=dtype)


def randn(shape, device: str | Device = "cpu", dtype=np.float32):
    return get_backend(device).random.randn(*shape).astype(dtype)


def matmul(left, right):
    return left @ right


def backend_of(value):
    """Return NumPy or CuPy for an existing array without forcing a copy."""
    if isinstance(value, np.ndarray):
        return np
    module = type(value).__module__
    if module.startswith("cupy"):
        import cupy
        return cupy
    return np
