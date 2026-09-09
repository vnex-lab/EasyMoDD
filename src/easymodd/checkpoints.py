"""Config-controlled checkpoint management."""

from __future__ import annotations

import json
from pathlib import Path
import zipfile
from .serialization import save_state


class CheckpointManager:
    """Checkpoint policy that creates no files when disabled."""

    def __init__(self, directory: str | Path, enabled: bool = True,
                 keep_recent: int = 2, compress_old: bool = True):
        if keep_recent < 1:
            raise ValueError("keep_recent must be at least 1")
        self.directory = Path(directory)
        self.enabled = bool(enabled)
        self.keep_recent = keep_recent
        self.compress_old = bool(compress_old)

    def save(self, state: dict, step: int) -> Path | None:
        if not self.enabled:
            return None
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"step_{step:07d}.json"
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        self._compress_old()
        return path

    def save_training_state(self, step: int, *, model, optimizer=None,
                            scheduler=None, config=None, metrics=None) -> Path | None:
        if not self.enabled:
            return None
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"step_{step:07d}.pkl"
        save_state(path, model=model, optimizer=optimizer, scheduler=scheduler,
                   config=config, metrics=metrics)
        self._compress_artifacts()
        return path

    def _compress_old(self) -> None:
        files = sorted(self.directory.glob("step_*.json"), key=lambda item: item.stat().st_mtime)
        for path in files[:-self.keep_recent]:
            archive = path.with_suffix(".zip")
            if self.compress_old and not archive.exists():
                with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
                    handle.write(path, arcname=path.name)
                path.unlink()

    def _compress_artifacts(self) -> None:
        files = sorted(self.directory.glob("step_*.pkl"), key=lambda item: item.stat().st_mtime)
        for path in files[:-self.keep_recent]:
            archive = path.with_suffix(".zip")
            if self.compress_old and not archive.exists():
                with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
                    handle.write(path, arcname=path.name)
                path.unlink()
