"""Lightweight local experiment tracking with no external service required."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class Experiment:
    """Record configuration and metrics locally, optionally integrating later."""

    def __init__(self, name: str, directory: str | Path = "runs", config: dict | None = None):
        if not name.strip():
            raise ValueError("Experiment name cannot be empty")
        self.name = name
        self.directory = Path(directory) / name
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        self._write_json("config.json", self.config)

    def log(self, metrics: dict[str, float], step: int | None = None) -> None:
        event = {"time": datetime.now(timezone.utc).isoformat(), "step": step, **metrics}
        with (self.directory / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")

    def artifact(self, name: str, content: str | bytes) -> Path:
        path = self.directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
        return path

    def _write_json(self, name: str, value) -> None:
        (self.directory / name).write_text(json.dumps(value, indent=2), encoding="utf-8")
