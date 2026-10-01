"""Typed extension points for custom models, tools, pages, and API routes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import importlib
import re

COMPONENT_GROUPS = frozenset({"models", "simulations", "compilers", "decompilers", "apps", "tools", "pages", "api_routes"})


class ComponentRegistry:
    """Named providers shared by model, simulator, and artifact systems."""

    def __init__(self):
        self._components: dict[str, dict[str, Callable]] = {group: {} for group in COMPONENT_GROUPS}

    def register(self, group: str, name: str, provider: Callable):
        if group not in self._components:
            raise ValueError(f"Unknown component group '{group}'. Choose from: {', '.join(sorted(COMPONENT_GROUPS))}")
        if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_.-]{0,63}", name):
            raise ValueError("Component name must start with a letter and contain only letters, digits, '.', '_' or '-'")
        if name in self._components[group]:
            raise ValueError(f"{group} component '{name}' is already registered")
        self._components[group][name] = provider
        return provider

    def get(self, group: str, name: str):
        try:
            return self._components[group][name]
        except KeyError as error:
            available = ", ".join(self.names(group)) or "none"
            raise KeyError(f"Unknown {group} component '{name}'. Available: {available}") from error

    def create(self, group: str, name: str, **options):
        return self.get(group, name)(**options)

    def names(self, group: str) -> tuple[str, ...]:
        if group not in self._components:
            raise ValueError(f"Unknown component group '{group}'")
        return tuple(sorted(self._components[group]))

    def inventory(self) -> dict[str, tuple[str, ...]]:
        return {group: self.names(group) for group in sorted(self._components)}


@dataclass(frozen=True)
class WebPage:
    path: str
    title: str
    html: str

    def __post_init__(self):
        if not self.path.startswith("/pages/") or ".." in self.path.split("/"):
            raise ValueError("Custom page paths must be under /pages/ and cannot traverse directories")
        if len(self.html.encode("utf-8")) > 2_000_000:
            raise ValueError("Custom page HTML cannot exceed 2 MB")


@dataclass(frozen=True)
class ApiRoute:
    path: str
    handler: Callable
    method: str = "POST"

    def __post_init__(self):
        if not self.path.startswith("/api/custom/") or ".." in self.path.split("/"):
            raise ValueError("Custom API paths must start with /api/custom/ and cannot traverse directories")
        if self.method.upper() != "POST":
            raise ValueError("Custom API routes support POST only")

class PluginLoader:
    """Load explicitly named Python plugins only after user opt-in."""

    @staticmethod
    def load(modules: list[str], workbench, *, enabled: bool = False) -> list[str]:
        if modules and not enabled:
            raise PermissionError("Plugin loading is disabled; enable trusted plugins in your consumer config")
        loaded = []
        for name in modules:
            module = importlib.import_module(name)
            register = getattr(module, "register", None)
            if not callable(register):
                raise TypeError(f"Plugin '{name}' must expose register(workbench)")
            register(workbench)
            loaded.append(name)
        return loaded