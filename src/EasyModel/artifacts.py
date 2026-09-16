"""Artifact metadata and conservative format detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import zipfile

from .errors import ArtifactError


class ArtifactKind(str, Enum):
    JAR = "jar"
    CLASS = "class"
    PE_EXE = "pe_exe"
    PE_DLL = "pe_dll"
    DOTNET_ASSEMBLY = "dotnet_assembly"
    PYTHON_BYTECODE = "python_bytecode"
    WASM = "wasm"
    SOURCE = "source"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Artifact:
    path: Path
    kind: ArtifactKind
    size: int
    metadata: dict[str, object] = field(default_factory=dict)


def detect_artifact(path: str | Path) -> Artifact:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ArtifactError(f"Artifact was not found: {source}")
    data = source.read_bytes()[:4096]
    suffix = source.suffix.lower()
    if data.startswith(b"PK\x03\x04") and (suffix == ".jar" or _is_jar(source)):
        kind = ArtifactKind.JAR
    elif data.startswith(b"\xca\xfe\xba\xbe"):
        kind = ArtifactKind.CLASS
    elif data[:2] == b"MZ":
        kind = ArtifactKind.PE_DLL if suffix == ".dll" else ArtifactKind.PE_EXE
    elif data.startswith(b"\x7fELF"):
        kind = ArtifactKind.UNKNOWN
    elif data.startswith(b"\x00asm"):
        kind = ArtifactKind.WASM
    elif suffix in {".pyc", ".pyo"}:
        kind = ArtifactKind.PYTHON_BYTECODE
    elif suffix in {".py", ".java", ".cs", ".c", ".cpp", ".h", ".rs", ".wat"}:
        kind = ArtifactKind.SOURCE
    else:
        kind = ArtifactKind.UNKNOWN
    return Artifact(path=source, kind=kind, size=source.stat().st_size)


def _is_jar(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            return any(name.endswith(".class") for name in archive.namelist())
    except zipfile.BadZipFile:
        return False
