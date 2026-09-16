"""Backend protocols and registries."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from .artifacts import Artifact
from .reports import ArtifactReport


class DecompilerBackend(Protocol):
    name: str

    def supports(self, artifact: Artifact) -> bool: ...

    def inspect(self, artifact: Artifact, **options) -> ArtifactReport: ...

    def decompile(self, artifact: Artifact, **options) -> ArtifactReport: ...


class BackendRegistry:
    def __init__(self):
        self._items: dict[str, object] = {}

    def register(self, name: str, backend=None):
        def attach(item):
            if name in self._items:
                raise ValueError(f"Backend '{name}' is already registered")
            self._items[name] = item
            return item
        return attach(backend) if backend is not None else attach

    def get(self, name: str):
        try:
            return self._items[name]
        except KeyError as error:
            raise KeyError(f"Unknown backend '{name}'. Available: {', '.join(self.names()) or 'none'}") from error

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))

    def select(self, artifact: Artifact, preferred: str | None = None):
        if preferred:
            backend = self.get(preferred)
            if not backend.supports(artifact):
                raise ValueError(f"Backend '{preferred}' does not support artifact kind '{artifact.kind.value}'")
            return backend
        for backend in self._items.values():
            if backend.supports(artifact):
                return backend
        raise ValueError(f"No registered backend supports '{artifact.kind.value}'")
