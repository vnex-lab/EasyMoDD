"""Small dataset primitives shared by EasyMoDD trainers."""

from __future__ import annotations

from pathlib import Path
import mmap
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

    def __init__(self, inputs, targets, transform=None, target_transform=None):
        if len(inputs) != len(targets):
            raise ValueError("inputs and targets must contain the same number of rows")
        if len(inputs) == 0:
            raise ValueError("dataset cannot be empty")
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        inputs, targets = self.inputs[index], self.targets[index]
        return (self.transform(inputs) if self.transform else inputs,
                self.target_transform(targets) if self.target_transform else targets)


def collate_batch(items):
    if not items:
        raise ValueError("Cannot collate an empty batch")
    inputs, targets = zip(*items)
    return np.stack(inputs), np.stack(targets)


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


class MMapTextDataset(Dataset):
    """Memory-mapped UTF-8 byte windows for corpora larger than RAM budgets.

    This dataset uses raw UTF-8 bytes as token IDs 0..255, matching the
    EasyMoDD byte-token convention. It stores no token list or window array;
    each sample copies only one small window from the mapped file.
    """

    def __init__(self, source: str | Path, sequence_length: int, stride: int | None = None):
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        self.path = Path(source).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Text dataset was not found: {self.path}")
        self.sequence_length = sequence_length
        self.stride = stride if stride is not None else sequence_length
        if self.stride < 1:
            raise ValueError("stride must be at least 1")
        self._file = self.path.open("rb")
        self._size = self.path.stat().st_size
        if self._size == 0:
            self._file.close()
            raise ValueError(f"Text dataset is empty: {self.path}")
        self._mapping = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self._valid_starts = max(0, self._size - sequence_length)
        if self._valid_starts == 0:
            self.close()
            raise ValueError(
                f"Text dataset has {self._size} bytes, but sequence_length={sequence_length} "
                "requires at least sequence_length + 1 bytes."
            )

    def __len__(self):
        return (self._valid_starts + self.stride - 1) // self.stride

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = index * self.stride
        raw = self._mapping[start : start + self.sequence_length + 1]
        tokens = np.frombuffer(raw, dtype=np.uint8).astype(np.int32)
        return tokens[:-1], tokens[1:]

    def close(self):
        mapping = getattr(self, "_mapping", None)
        if mapping is not None and not mapping.closed:
            mapping.close()
        handle = getattr(self, "_file", None)
        if handle is not None and not handle.closed:
            handle.close()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
