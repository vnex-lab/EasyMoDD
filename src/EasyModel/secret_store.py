"""Optional environment-backed integration secrets."""

from __future__ import annotations

import os


class SecretStore:
    """Read secrets from environment variables without persisting them."""

    def __init__(self, prefix: str = ""):
        self.prefix = prefix

    def _key(self, name: str) -> str:
        return f"{self.prefix}{name}" if self.prefix else name

    def get(self, name: str, default: str | None = None, *, required: bool = False) -> str | None:
        value = os.environ.get(self._key(name), default)
        if required and not value:
            raise RuntimeError(f"Required integration secret '{self._key(name)}' is not set")
        return value

    def require(self, *names: str) -> dict[str, str]:
        return {name: self.get(name, required=True) for name in names}  # type: ignore[misc]

    def has(self, name: str) -> bool:
        return bool(self.get(name))

    def redact(self, name: str) -> str:
        return "<set>" if self.has(name) else "<missing>"
