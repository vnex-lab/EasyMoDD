"""Execution and input safety policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .errors import SecurityPolicyError


@dataclass(frozen=True)
class SecurityPolicy:
    allow_execution: bool = False
    allow_network: bool = False
    allow_native_tools: bool = True
    max_input_size: int = 1_073_741_824
    timeout_seconds: float = 120.0

    def validate_input(self, path: str | Path) -> Path:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise SecurityPolicyError(f"Input artifact does not exist or is not a file: {source}")
        if source.stat().st_size > self.max_input_size:
            raise SecurityPolicyError(
                f"Input artifact is too large: {source.stat().st_size:,} bytes; "
                f"limit is {self.max_input_size:,} bytes"
            )
        return source

    def require_execution(self) -> None:
        if not self.allow_execution:
            raise SecurityPolicyError(
                "Execution is disabled by policy. Set allow_execution=true only "
                "for source/toolchain operations you explicitly trust."
            )
