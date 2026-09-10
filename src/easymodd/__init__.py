"""EasyMoDD: configurable, training-first numerical machine learning tools."""

from .config import EasyMoDDConfig, load_config
from .tensor import Device, array, backend_of, get_backend, matmul, ones, randn, seed, zeros
from .nn import Dropout, GELU, Linear, Module, Parameter, ReLU, Sequential, Sigmoid, Tanh
from .optim import Adam, AdamW, SGD
from .schedulers import CosineScheduler, WarmupScheduler
from .metrics import accuracy, mean_absolute_error, perplexity
from .callbacks import EarlyStopping, MetricsLogger
from .serialization import load_state, save_state
from .training import Trainer, TrainingHistory
from .data import ArrayDataset, BatchLoader, Dataset, Subset, TextDataset, collate_batch, split_dataset
from .losses import CrossEntropyLoss, HuberLoss, L1Loss, MSELoss
from .checkpoints import CheckpointManager
from .experiments import Experiment
from .registry import ModelRegistry, models
from .secret_store import SecretStore
from .transforms import Compose, Flatten, MapTransform, Normalize, Standardize, ToFloat32
from .visualization import summarize_history, summarize_model
from .models import MLP, TextTransformerConfig, VisionTransformerConfig, build_mlp

__all__ = [
    "Device",
    "ArrayDataset",
    "BatchLoader",
    "Adam",
    "AdamW",
    "CheckpointManager",
    "Compose",
    "CrossEntropyLoss",
    "CosineScheduler",
    "Dataset",
    "Dropout",
    "EarlyStopping",
    "EasyMoDDConfig",
    "Experiment",
    "Flatten",
    "GELU",
    "HuberLoss",
    "L1Loss",
    "Linear",
    "MSELoss",
    "MapTransform",
    "ModelRegistry",
    "Normalize",
    "MetricsLogger",
    "Module",
    "Parameter",
    "ReLU",
    "Sequential",
    "SGD",
    "SecretStore",
    "Sigmoid",
    "Standardize",
    "Tanh",
    "ToFloat32",
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
    "collate_batch",
    "models",
    "ones",
    "randn",
    "seed",
    "save_state",
    "zeros",
    "split_dataset",
    "summarize_history",
    "summarize_model",
]
