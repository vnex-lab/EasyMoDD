"""Controlled subprocess execution for compiler/decompiler tools."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import time

from .errors import BackendUnavailableError, SecurityPolicyError
from .reports import BackendResult
from .security import SecurityPolicy


@dataclass(frozen=True)
class Command:
    executable: str
    arguments: tuple[str, ...] = ()

    def argv(self) -> list[str]:
        return [self.executable, *self.arguments]


def run_command(command: Command, *, policy: SecurityPolicy,
                cwd: str | Path | None = None, env: dict[str, str] | None = None,
                dry_run: bool = False, backend: str = "external") -> BackendResult:
    if not policy.allow_native_tools:
        raise SecurityPolicyError("Native tool execution is disabled by the security policy")
    executable = command.executable
    if not _find_executable(executable):
        raise BackendUnavailableError(
            f"Executable '{executable}' was not found. Install the tool or configure its full path."
        )
    argv = command.argv()
    if dry_run:
        return BackendResult(ok=True, backend=backend, command=argv, warnings=["dry_run: command was not executed"])
    policy.require_execution()
    started = time.perf_counter()
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    try:
        completed = subprocess.run(
            argv, cwd=cwd, env=merged_env, capture_output=True, text=True,
            timeout=policy.timeout_seconds, check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise SecurityPolicyError(
            f"Command exceeded timeout of {policy.timeout_seconds} seconds: {executable}"
        ) from error
    return BackendResult(
        ok=completed.returncode == 0,
        backend=backend,
        stdout=completed.stdout,
        stderr=completed.stderr,
        exit_code=completed.returncode,
        command=argv,
        duration_seconds=time.perf_counter() - started,
    )


def _find_executable(executable: str) -> str | None:
    path = Path(executable)
    if path.is_file():
        return str(path)
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(directory) / executable
        if candidate.is_file():
            return str(candidate)
        if os.name == "nt" and candidate.with_suffix(".exe").is_file():
            return str(candidate.with_suffix(".exe"))
    return None
