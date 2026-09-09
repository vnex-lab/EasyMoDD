"""Versioned, portable state serialization for EasyMoDD runs."""

from __future__ import annotations

import pickle
from pathlib import Path


FORMAT_VERSION = 1


def save_state(path: str | Path, *, model, optimizer=None, scheduler=None,
               config: dict | None = None, metrics: dict | None = None) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "model": model.state_dict() if hasattr(model, "state_dict") else None,
        "optimizer": optimizer.__dict__ if optimizer is not None else None,
        "scheduler": scheduler.__dict__ if scheduler is not None else None,
        "config": config or {},
        "metrics": metrics or {},
    }
    with destination.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return destination


def load_state(path: str | Path) -> dict:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint was not found: {source}")
    try:
        with source.open("rb") as handle:
            payload = pickle.load(handle)
    except (OSError, pickle.PickleError, EOFError) as error:
        raise ValueError(f"Checkpoint is corrupt or unreadable: {source}") from error
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format in {source}: "
            f"{payload.get('format_version')!r}; expected {FORMAT_VERSION}"
        )
    return payload