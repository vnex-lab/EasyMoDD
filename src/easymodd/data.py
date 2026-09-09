"""Small dataset primitives shared by EasyMoDD trainers."""

from __future__ import annotations

from pathlib import Path
import numpy as np


class Dataset:
    """Minimal indexable dataset contract."""

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError


class BatchLoader:
    """Deterministic mini-batch iterator for any indexable Dataset."""

    def __init__(self, dataset: Dataset, batch_size: int = 32, shuffle: bool = True,
                 drop_last: bool = False, seed: int = 42):
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            rng.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue
            yield [self.dataset[int(index)] for index in batch_indices]

    def __len__(self):
        size = len(self.dataset) // self.batch_size
        return size if self.drop_last else (size + bool(len(self.dataset) % self.batch_size))


def split_dataset(dataset: Dataset, validation_fraction: float = 0.1, seed: int = 42):
    """Return disjoint train and validation Subset objects."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    indices = np.arange(len(dataset))
    np.random.default_rng(seed).shuffle(indices)
    cutoff = max(1, int(len(indices) * (1 - validation_fraction)))
    return Subset(dataset, indices[:cutoff]), Subset(dataset, indices[cutoff:])


class Subset(Dataset):
    def __init__(self, dataset: Dataset, indices):
        if len(indices) == 0:
            raise ValueError("subset cannot be empty")
        self.dataset = dataset
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[int(self.indices[index])]


class ArrayDataset(Dataset):
    """Dataset for aligned NumPy/CuPy-compatible input and target arrays."""

    def __init__(self, inputs, targets):
        if len(inputs) != len(targets):
            raise ValueError("inputs and targets must contain the same number of rows")
        if len(inputs) == 0:
            raise ValueError("dataset cannot be empty")
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]


class TextDataset(Dataset):
    """Packed next-token windows from a UTF-8 text file."""

    def __init__(self, source: str | Path, tokenizer, sequence_length: int):
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Text dataset was not found: {path}")
        tokens = tokenizer.encode(path.read_text(encoding="utf-8"))
        self.windows = []
        for start in range(0, len(tokens) - sequence_length, sequence_length):
            window = tokens[start : start + sequence_length + 1]
            if len(window) == sequence_length + 1:
                self.windows.append(np.asarray(window, dtype=np.int32))
        if len(self.windows) < 2:
            raise ValueError(
                f"Text dataset {path} is too small for sequence_length={sequence_length}; "
                "provide more text or reduce the sequence length."
            )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        window = self.windows[index]
        return window[:-1], window[1:]
