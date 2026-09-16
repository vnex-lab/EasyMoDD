"""Consumer-owned TOML configuration for EasyModel."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib

from .errors import ConfigurationError
from .security import SecurityPolicy


@dataclass(frozen=True)
class EasyModelConfig:
    values: dict
    source: Path

    def get(self, section: str, key: str, default=None):
        return self.values.get(section, {}).get(key, default)

    def path(self, section: str, key: str, default=None) -> Path | None:
        value = self.get(section, key, default)
        if value is None:
            return None
        path = Path(str(value))
        return path if path.is_absolute() else (self.source.parent / path).resolve()

    def security_policy(self) -> SecurityPolicy:
        values = self.values.get("security", {})
        return SecurityPolicy(**{key: values[key] for key in SecurityPolicy.__dataclass_fields__ if key in values})


def load_config(path: str | Path) -> EasyModelConfig:
    source = Path(path).expanduser().resolve()
    try:
        values = tomllib.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfigurationError(f"EasyModel config was not found: {source}") from error
    except tomllib.TOMLDecodeError as error:
        raise ConfigurationError(f"Invalid TOML in {source}: {error}") from error
    if not isinstance(values, dict):
        raise ConfigurationError("EasyModel config must contain TOML sections")
    return EasyModelConfig(values, source)
