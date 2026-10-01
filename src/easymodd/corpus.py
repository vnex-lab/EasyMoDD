"""Streaming corpus health checks and bounded-memory deduplication."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time


@dataclass(frozen=True)
class CorpusAudit:
    """Compact, reproducible summary of a text or JSONL corpus."""

    source: str
    format: str
    sha256: str
    bytes: int
    records: int
    unique_records: int
    duplicate_records: int
    empty_records: int
    malformed_records: int
    characters: int
    words: int
    estimated_tokens: int
    deduplicated_path: str | None
    elapsed_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def audit_corpus(
    source: str | Path,
    *,
    format: str = "auto",
    text_field: str | None = None,
    deduplicated_output: str | Path | None = None,
    tokenizer=None,
) -> CorpusAudit:
    """Audit newline-delimited text/JSONL without loading the corpus in RAM.

    Duplicate identity is a case-folded, whitespace-normalized record hash.
    SQLite stores those hashes on disk, keeping working memory bounded for
    large corpora. JSONL output preserves original valid source lines.
    """
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Corpus file was not found: {path}")
    selected_format = _detect_format(path, format)
    output_path = Path(deduplicated_output).expanduser().resolve() if deduplicated_output else None
    if output_path is not None and output_path == path:
        raise ValueError("Deduplicated output must not overwrite the source corpus")
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    source_hash = hashlib.sha256()
    counts = {"records": 0, "duplicates": 0, "empty": 0, "malformed": 0,
              "characters": 0, "words": 0, "tokens": 0, "unique": 0}
    db_file = tempfile.NamedTemporaryFile(prefix="easymodd-audit-", suffix=".sqlite3", delete=False)
    db_file.close()
    connection = sqlite3.connect(db_file.name)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA cache_size=-2048")
        connection.execute("CREATE TABLE seen (digest BLOB PRIMARY KEY) WITHOUT ROWID")
        output_handle = output_path.open("w", encoding="utf-8", newline="") if output_path else None
        try:
            with path.open("rb") as source_handle:
                for raw_line in source_handle:
                    source_hash.update(raw_line)
                    counts["records"] += 1
                    try:
                        decoded = raw_line.decode("utf-8").rstrip("\r\n")
                    except UnicodeDecodeError:
                        decoded = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                        counts["malformed"] += 1

                    identity_text = decoded
                    valid_record = True
                    if selected_format == "jsonl":
                        try:
                            parsed = json.loads(decoded)
                            if text_field:
                                parsed_value = _get_field(parsed, text_field)
                                identity_text = str(parsed_value) if parsed_value is not None else ""
                            else:
                                identity_text = json.dumps(parsed, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                        except (json.JSONDecodeError, TypeError, ValueError):
                            counts["malformed"] += 1
                            valid_record = False

                    normalized = " ".join(identity_text.casefold().split())
                    if not normalized:
                        counts["empty"] += 1
                        is_unique = False
                    else:
                        digest = hashlib.sha256(normalized.encode("utf-8")).digest()
                        cursor = connection.execute("INSERT OR IGNORE INTO seen(digest) VALUES (?)", (digest,))
                        is_unique = cursor.rowcount == 1

                    if is_unique:
                        counts["unique"] += 1
                        counts["characters"] += len(identity_text)
                        counts["words"] += len(identity_text.split())
                        counts["tokens"] += len(tokenizer.encode(identity_text)) if tokenizer else len(identity_text.split())
                        if output_handle:
                            output_handle.write(decoded + "\n")
                    elif normalized:
                        counts["duplicates"] += 1
                    elif output_handle and valid_record and normalized:
                        output_handle.write(decoded + "\n")
            connection.commit()
        finally:
            if output_handle:
                output_handle.close()
    finally:
        connection.close()
        try:
            os.unlink(db_file.name)
        except OSError:
            pass

    return CorpusAudit(
        source=str(path), format=selected_format, sha256=source_hash.hexdigest(),
        bytes=path.stat().st_size, records=counts["records"], unique_records=counts["unique"],
        duplicate_records=counts["duplicates"], empty_records=counts["empty"],
        malformed_records=counts["malformed"], characters=counts["characters"],
        words=counts["words"], estimated_tokens=counts["tokens"],
        deduplicated_path=str(output_path) if output_path else None,
        elapsed_seconds=time.perf_counter() - started,
    )


def _detect_format(path: Path, requested: str) -> str:
    if requested in {"text", "jsonl"}:
        return requested
    if requested != "auto":
        raise ValueError("format must be 'auto', 'text', or 'jsonl'")
    return "jsonl" if path.suffix.lower() in {".jsonl", ".ndjson"} else "text"


def _get_field(value, dotted_path: str):
    for part in dotted_path.split("."):
        if isinstance(value, dict):
            value = value.get(part)
        else:
            return None
    return value
