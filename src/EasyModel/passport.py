"""Dependency-free artifact fingerprint and structural safety summary."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from collections import Counter
from pathlib import Path
import struct
import zipfile

from .artifacts import ArtifactKind, detect_artifact


@dataclass(frozen=True)
class ArtifactPassport:
    path: str
    kind: str
    size_bytes: int
    sha256: str
    details: dict[str, object] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def create_passport(path: str | Path, *, max_archive_entries: int = 50_000) -> ArtifactPassport:
    """Fingerprint and inspect common binary formats without execution.

    Archive contents are inspected from their central directory only; members
    are never extracted. File hashing is streaming with a fixed-size buffer.
    """
    artifact = detect_artifact(path)
    digest = hashlib.sha256()
    with artifact.path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    details: dict[str, object] = {}
    warnings: list[str] = []
    kind = artifact.kind

    if kind == ArtifactKind.JAR:
        details, warnings = _inspect_zip(artifact.path, max_archive_entries)
    elif kind == ArtifactKind.CLASS:
        with artifact.path.open("rb") as handle:
            raw = handle.read(8)
        details = {
            "magic": raw[:4].hex(),
            "minor_version": struct.unpack(">H", raw[4:6])[0] if len(raw) >= 6 else None,
            "major_version": struct.unpack(">H", raw[6:8])[0] if len(raw) >= 8 else None,
        }
    elif kind in {ArtifactKind.PE_EXE, ArtifactKind.PE_DLL, ArtifactKind.DOTNET_ASSEMBLY}:
        details, warnings = _inspect_pe(artifact.path)
        if details.get("is_dotnet"):
            kind = ArtifactKind.DOTNET_ASSEMBLY
    elif kind == ArtifactKind.PYTHON_BYTECODE:
        with artifact.path.open("rb") as handle:
            header = handle.read(16)
        details = {
            "magic": header[:4].hex(),
            "flags": int.from_bytes(header[4:8], "little") if len(header) >= 8 else None,
            "header_bytes": len(header),
            "code_executed": False,
        }
        if len(header) < 16:
            warnings.append("File is shorter than the standard modern Python bytecode header")
    elif kind == ArtifactKind.WASM:
        details, warnings = _inspect_wasm(artifact.path)
    elif kind == ArtifactKind.SOURCE:
        details = _inspect_text(artifact.path)
    else:
        with artifact.path.open("rb") as handle:
            details = {"magic_prefix": handle.read(16).hex()}

    return ArtifactPassport(
        path=str(artifact.path), kind=kind.value, size_bytes=artifact.size,
        sha256=digest.hexdigest(), details=details, warnings=tuple(warnings),
    )


def compare_passports(left: ArtifactPassport, right: ArtifactPassport) -> dict[str, object]:
    """Compare identity and structural facts without loading binary contents."""
    keys = sorted(set(left.details) | set(right.details))
    return {
        "identical_content": left.sha256 == right.sha256,
        "same_kind": left.kind == right.kind,
        "left_sha256": left.sha256,
        "right_sha256": right.sha256,
        "detail_changes": {
            key: {"left": left.details.get(key), "right": right.details.get(key)}
            for key in keys if left.details.get(key) != right.details.get(key)
        },
    }


def _inspect_zip(path: Path, limit: int):
    warnings: list[str] = []
    details: dict[str, object] = {"entry_count": 0, "class_count": 0, "total_uncompressed_bytes": 0}
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            details["entry_count"] = len(infos)
            if len(infos) > limit:
                warnings.append(f"Archive has {len(infos):,} entries; listing limited to {limit:,}")
                infos = infos[:limit]
            names = [info.filename for info in infos]
            duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
            unsafe = [name for name in names if _unsafe_archive_name(name)]
            total_size = sum(info.file_size for info in infos)
            details.update({
                "entries": names,
                "class_count": sum(name.endswith(".class") for name in names),
                "total_uncompressed_bytes": total_size,
                "duplicate_entry_names": duplicates,
                "unsafe_entry_names": unsafe,
                "manifest_present": "META-INF/MANIFEST.MF" in names,
            })
            if unsafe:
                warnings.append("Archive contains absolute or parent-traversal paths; do not extract without sanitizing")
            if duplicates:
                warnings.append("Archive contains duplicate member names")
            if total_size > max(path.stat().st_size, 1) * 1000:
                warnings.append("Archive has an extreme declared compression ratio")
            if "META-INF/MANIFEST.MF" in names:
                manifest = archive.getinfo("META-INF/MANIFEST.MF")
                if manifest.file_size <= 1_000_000:
                    details["manifest"] = archive.read(manifest).decode("utf-8", errors="replace")
    except (zipfile.BadZipFile, OSError) as error:
        warnings.append(f"Archive metadata could not be read: {error}")
    return details, warnings


def _unsafe_archive_name(name: str) -> bool:
    normalized = name.replace("\\", "/")
    return normalized.startswith("/") or any(part == ".." for part in normalized.split("/"))


def _inspect_pe(path: Path):
    warnings: list[str] = []
    details: dict[str, object] = {"is_dotnet": False}
    try:
        with path.open("rb") as handle:
            dos = handle.read(64)
            if len(dos) < 64 or dos[:2] != b"MZ":
                return details, ["Invalid DOS/PE header"]
            pe_offset = struct.unpack_from("<I", dos, 0x3C)[0]
            if pe_offset > path.stat().st_size - 24:
                return details, ["PE header offset is outside file bounds"]
            handle.seek(pe_offset)
            header = handle.read(24)
            if header[:4] != b"PE\x00\x00":
                return details, ["PE signature is missing"]
            machine, section_count, timestamp, _, _, optional_size, characteristics = struct.unpack_from("<HHIIIHH", header, 4)
            optional = handle.read(min(optional_size, 4096))
            magic = struct.unpack_from("<H", optional, 0)[0] if len(optional) >= 2 else 0
            directory_offset = 112 if magic == 0x20B else 96 if magic == 0x10B else None
            clr_size = 0
            if directory_offset and len(optional) >= directory_offset + 15 * 8:
                _, clr_size = struct.unpack_from("<II", optional, directory_offset + 14 * 8)
            section_names = []
            handle.seek(pe_offset + 4 + 20 + optional_size)
            for _ in range(min(section_count, 4096)):
                section = handle.read(40)
                if len(section) < 40:
                    warnings.append("Truncated PE section table")
                    break
                section_names.append(section[:8].split(b"\0", 1)[0].decode("ascii", errors="replace"))
            details.update({
                "machine": f"0x{machine:04x}",
                "section_count": section_count,
                "sections": section_names,
                "timestamp": timestamp,
                "optional_header": "PE32+" if magic == 0x20B else "PE32" if magic == 0x10B else "unknown",
                "characteristics": f"0x{characteristics:04x}",
                "is_dotnet": clr_size > 0,
            })
    except (OSError, struct.error) as error:
        warnings.append(f"PE metadata could not be parsed: {error}")
    return details, warnings


def _inspect_wasm(path: Path):
    warnings: list[str] = []
    details: dict[str, object] = {"sections": []}
    try:
        with path.open("rb") as handle:
            header = handle.read(8)
            if len(header) < 8 or header[:4] != b"\0asm":
                return details, ["Invalid WebAssembly header"]
            details["version"] = int.from_bytes(header[4:8], "little")
            while True:
                section_id = handle.read(1)
                if not section_id:
                    break
                size = _read_uleb(handle)
                if size is None:
                    warnings.append("Malformed WebAssembly section size")
                    break
                details["sections"].append({"id": section_id[0], "size_bytes": size})
                handle.seek(size, 1)
                if handle.tell() > path.stat().st_size:
                    warnings.append("WebAssembly section extends beyond end of file")
                    break
    except OSError as error:
        warnings.append(f"WebAssembly metadata could not be parsed: {error}")
    return details, warnings


def _read_uleb(handle):
    value = 0
    shift = 0
    for _ in range(5):
        byte = handle.read(1)
        if not byte:
            return None
        number = byte[0]
        value |= (number & 0x7F) << shift
        if number < 0x80:
            return value
        shift += 7
    return None


def _inspect_text(path: Path):
    line_count = 0
    with path.open("rb") as handle:
        for _ in handle:
            line_count += 1
    return {"line_count": line_count, "encoding_assumption": "UTF-8"}