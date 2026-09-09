"""EasyMoDD: configurable, training-first numerical machine learning tools."""

from .config import EasyMoDDConfig, load_config
from .tensor import Device, array, backend_of, get_backend, matmul, ones, randn, seed, zeros
from .nn import Linear, Module, Parameter, ReLU, Sequential
from .optim import Adam, AdamW, SGD
from .schedulers import CosineScheduler, WarmupScheduler
from .metrics import accuracy, mean_absolute_error, perplexity
from .callbacks import EarlyStopping, MetricsLogger
from .serialization import load_state, save_state
from .training import Trainer, TrainingHistory
from .data import ArrayDataset, BatchLoader, Dataset, Subset, TextDataset, split_dataset
from .losses import CrossEntropyLoss, MSELoss
from .checkpoints import CheckpointManager
from .models import MLP, TextTransformerConfig, VisionTransformerConfig, build_mlp

__all__ = [
    "Device",
    "ArrayDataset",
    "BatchLoader",
    "Adam",
    "AdamW",
    "CheckpointManager",
    "CrossEntropyLoss",
    "CosineScheduler",
    "Dataset",
    "EarlyStopping",
    "EasyMoDDConfig",
    "Linear",
    "MSELoss",
    "MetricsLogger",
    "Module",
    "Parameter",
    "ReLU",
    "Sequential",
    "SGD",
    "WarmupScheduler",
    "Trainer",
    "TrainingHistory",
    "TextDataset",
    "Subset",
    "array",
    "accuracy",
    "backend_of",
    "MLP",
    "TextTransformerConfig",
    "VisionTransformerConfig",
    "build_mlp",
    "get_backend",
    "load_config",
    "load_state",
    "matmul",
    "mean_absolute_error",
    "perplexity",
    "ones",
    "randn",
    "seed",
    "save_state",
    "zeros",
    "split_dataset",
]
