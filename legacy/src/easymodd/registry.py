"""Named factories for configurable EasyMoDD components."""

from __future__ import annotations

from collections.abc import Callable


class ModelRegistry:
    """Register and construct models without hard-coded filenames."""

    def __init__(self):
        self._factories: dict[str, Callable] = {}

    def register(self, name: str, factory: Callable | None = None):
        if not name or not name.strip():
            raise ValueError("Model name cannot be empty")

        def attach(item):
            if name in self._factories:
                raise ValueError(f"A model named '{name}' is already registered")
            self._factories[name] = item
            return item

        return attach(factory) if factory is not None else attach

    def create(self, name: str, **settings):
        try:
            factory = self._factories[name]
        except KeyError as error:
            available = ", ".join(self.names()) or "none"
            raise ValueError(f"Unknown model '{name}'. Available models: {available}") from error
        return factory(**settings)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


models = ModelRegistry()
