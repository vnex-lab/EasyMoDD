"""Configuration loading and validation for EasyMoDD projects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any


@dataclass(frozen=True)
class EasyMoDDConfig:
    """Validated TOML configuration with paths resolved from its location."""

    values: dict[str, Any]
    source: Path

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self.values.get(section, {}).get(key, default)

    def path(self, section: str, key: str, default: str | None = None) -> Path | None:
        value = self.get(section, key, default)
        if value is None:
            return None
        path = Path(str(value))
        return path if path.is_absolute() else (self.source.parent / path).resolve()


def load_config(path: str | Path = "config.txt") -> EasyMoDDConfig:
    """Load TOML from a `.txt` config file and report actionable errors."""
    source = Path(path).resolve()
    try:
        raw = tomllib.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"EasyMoDD config was not found: {source}. Create config.txt first."
        ) from error
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"Invalid TOML in {source}: {error}") from error

    if not isinstance(raw, dict):
        raise ValueError("EasyMoDD config must contain TOML sections")
    allowed_sections = {
        "project", "paths", "data", "model", "training", "validation",
        "checkpoints", "device", "logging",
    }
    unknown = sorted(set(raw) - allowed_sections)
    if unknown:
        raise ValueError(f"Unknown config section(s): {', '.join(unknown)}")
    checkpoints = raw.get("checkpoints", {})
    if not isinstance(checkpoints, dict):
        raise ValueError("[checkpoints] must be a TOML table")
    retention = checkpoints.get("keep_recent", 2)
    if not isinstance(retention, int) or retention < 1:
        raise ValueError("[checkpoints] keep_recent must be an integer >= 1")
    return EasyMoDDConfig(values=raw, source=source)
