"""Configurable wrappers around installed compiler toolchains."""

from __future__ import annotations

from pathlib import Path

from .errors import CompilationError
from .process import Command, run_command
from .reports import BackendResult
from .security import SecurityPolicy


class CompilerBackend:
    def __init__(self, name: str, executable: str):
        self.name = name
        self.executable = executable

    def compile(self, source: str | Path, *, output: str | Path | None = None,
                arguments: tuple[str, ...] = (), policy: SecurityPolicy | None = None,
                dry_run: bool = False) -> BackendResult:
        source = Path(source).expanduser().resolve()
        if not source.is_file():
            raise CompilationError(f"Source file was not found: {source}")
        args = (*arguments, str(source))
        if output is not None:
            args = (*args, "-o", str(Path(output).expanduser().resolve()))
        return run_command(Command(self.executable, args), policy=policy or SecurityPolicy(),
                           dry_run=dry_run, backend=self.name)


class CompilerRegistry:
    def __init__(self):
        self._items: dict[str, CompilerBackend] = {}

    def register(self, backend: CompilerBackend) -> CompilerBackend:
        if backend.name in self._items:
            raise ValueError(f"Compiler '{backend.name}' is already registered")
        self._items[backend.name] = backend
        return backend

    def get(self, name: str) -> CompilerBackend:
        try:
            return self._items[name]
        except KeyError as error:
            raise KeyError(f"Unknown compiler '{name}'. Available: {', '.join(self.names()) or 'none'}") from error

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))


compilers = CompilerRegistry()
