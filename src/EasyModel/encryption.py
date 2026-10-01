"""Optional chunked authenticated encryption for large files."""

from __future__ import annotations

import os
from pathlib import Path
import secrets
import struct
import tempfile

from .errors import BackendUnavailableError, EasyModelError


_MAGIC = b"EMCRYPT1"
_MIN_CHUNK = 4096
_MAX_CHUNK = 16 * 1024 * 1024
_DEFAULT_ITERATIONS = 600_000


class EncryptionError(EasyModelError):
    """Encryption file format, key, or authentication failure."""


def encrypt_file(source: str | Path, destination: str | Path, passphrase: str | bytes,
                 *, chunk_size: int = 1024 * 1024, iterations: int = _DEFAULT_ITERATIONS) -> Path:
    """Encrypt a file incrementally using AES-256-GCM authenticated chunks.

    Passphrases are never stored. An authenticated terminal record protects
    against truncation; output is replaced atomically only after completion.
    Requires the optional `cryptography` package.
    """
    AESGCM, PBKDF2HMAC, hashes = _crypto()
    src, dst = _paths(source, destination)
    if not _MIN_CHUNK <= chunk_size <= _MAX_CHUNK:
        raise ValueError(f"chunk_size must be between {_MIN_CHUNK} and {_MAX_CHUNK} bytes")
    if not 100_000 <= iterations <= 5_000_000:
        raise ValueError("iterations must be between 100,000 and 5,000,000")
    phrase = _passphrase(passphrase)
    salt, nonce_prefix = secrets.token_bytes(16), secrets.token_bytes(8)
    header = _MAGIC + struct.pack(">II", chunk_size, iterations) + salt + nonce_prefix
    key = _derive_key(PBKDF2HMAC, hashes, phrase, salt, iterations)
    cipher = AESGCM(key)
    temporary = _temporary_path(dst)
    chunk_index = 0
    try:
        with src.open("rb") as input_handle, temporary.open("wb") as output_handle:
            output_handle.write(header)
            while True:
                block = input_handle.read(chunk_size)
                if not block:
                    break
                nonce = nonce_prefix + struct.pack(">I", chunk_index)
                encrypted = cipher.encrypt(nonce, block, header + b"D" + struct.pack(">I", chunk_index))
                output_handle.write(struct.pack(">I", len(encrypted)))
                output_handle.write(encrypted)
                chunk_index += 1
                if chunk_index >= 0xFFFFFFFF:
                    raise EncryptionError("File exceeds the format's maximum number of encrypted chunks")
            nonce = nonce_prefix + struct.pack(">I", chunk_index)
            final = cipher.encrypt(nonce, b"", header + b"E" + struct.pack(">I", chunk_index))
            output_handle.write(struct.pack(">I", len(final)))
            output_handle.write(final)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(temporary, dst)
        return dst
    finally:
        _remove_temporary(temporary)


def decrypt_file(source: str | Path, destination: str | Path, passphrase: str | bytes) -> Path:
    """Decrypt and authenticate a chunked EasyModel file without partial output."""
    AESGCM, PBKDF2HMAC, hashes = _crypto()
    src, dst = _paths(source, destination)
    phrase = _passphrase(passphrase)
    temporary = _temporary_path(dst)
    try:
        with src.open("rb") as input_handle:
            header = input_handle.read(40)
            if len(header) != 40 or header[:8] != _MAGIC:
                raise EncryptionError("Not a supported EasyModel encrypted file")
            chunk_size, iterations = struct.unpack_from(">II", header, 8)
            if not _MIN_CHUNK <= chunk_size <= _MAX_CHUNK or not 100_000 <= iterations <= 5_000_000:
                raise EncryptionError("Encrypted file contains invalid cryptographic parameters")
            salt, nonce_prefix = header[16:32], header[32:40]
            key = _derive_key(PBKDF2HMAC, hashes, phrase, salt, iterations)
            cipher = AESGCM(key)
            chunk_index = 0
            terminal_seen = False
            with temporary.open("wb") as output_handle:
                while True:
                    raw_length = input_handle.read(4)
                    if not raw_length:
                        break
                    if len(raw_length) != 4:
                        raise EncryptionError("Encrypted file ended inside a chunk header")
                    frame_size = struct.unpack(">I", raw_length)[0]
                    if frame_size < 16 or frame_size > chunk_size + 16:
                        raise EncryptionError("Encrypted file contains an invalid chunk size")
                    frame = input_handle.read(frame_size)
                    if len(frame) != frame_size:
                        raise EncryptionError("Encrypted file ended inside an authenticated chunk")
                    nonce = nonce_prefix + struct.pack(">I", chunk_index)
                    if frame_size == 16:
                        try:
                            final = cipher.decrypt(nonce, frame, header + b"E" + struct.pack(">I", chunk_index))
                        except Exception as error:
                            raise EncryptionError("Authentication failed; wrong passphrase or modified file") from error
                        if final:
                            raise EncryptionError("Invalid encrypted terminal record")
                        terminal_seen = True
                        if input_handle.read(1):
                            raise EncryptionError("Unexpected data after encrypted terminal record")
                        break
                    try:
                        block = cipher.decrypt(nonce, frame, header + b"D" + struct.pack(">I", chunk_index))
                    except Exception as error:
                        raise EncryptionError("Authentication failed; wrong passphrase or modified file") from error
                    if not block or len(block) > chunk_size:
                        raise EncryptionError("Invalid plaintext chunk length")
                    output_handle.write(block)
                    chunk_index += 1
                    if chunk_index >= 0xFFFFFFFF:
                        raise EncryptionError("Encrypted file exceeds the supported chunk count")
                if not terminal_seen:
                    raise EncryptionError("Encrypted file is truncated or missing its authentication trailer")
                output_handle.flush()
                os.fsync(output_handle.fileno())
        os.replace(temporary, dst)
        return dst
    finally:
        _remove_temporary(temporary)


def _crypto():
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        from cryptography.hazmat.primitives import hashes
        return AESGCM, PBKDF2HMAC, hashes
    except ImportError as error:
        raise BackendUnavailableError(
            "File encryption requires the optional dependency. Install EasyMoDD with `pip install easymodd[security]`."
        ) from error


def _derive_key(PBKDF2HMAC, hashes, passphrase: bytes, salt: bytes, iterations: int) -> bytes:
    return PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=iterations).derive(passphrase)


def _passphrase(value: str | bytes) -> bytes:
    result = value.encode("utf-8") if isinstance(value, str) else bytes(value)
    if len(result) < 12:
        raise ValueError("Passphrase must be at least 12 bytes")
    return result


def _paths(source, destination) -> tuple[Path, Path]:
    src = Path(source).expanduser().resolve()
    dst = Path(destination).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Input file was not found: {src}")
    if src == dst:
        raise ValueError("Source and destination must be different files")
    dst.parent.mkdir(parents=True, exist_ok=True)
    return src, dst


def _temporary_path(destination: Path) -> Path:
    handle = tempfile.NamedTemporaryFile(prefix=".easymodel-", suffix=".tmp", dir=destination.parent, delete=False)
    path = Path(handle.name)
    handle.close()
    return path


def _remove_temporary(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
