from __future__ import annotations

import struct
from pathlib import Path

import pytest


@pytest.fixture
def make_pe_file():
	def create(path: Path, *, dll: bool = True, dotnet: bool = False) -> Path:
		dos = bytearray(64)
		dos[:2] = b"MZ"
		struct.pack_into("<I", dos, 0x3C, 64)
		coff = struct.pack(
			"<HHIIIHH",
			0x14C,
			1,
			0x12345678,
			0,
			0,
			224,
			0x2102 if dll else 0x0102,
		)
		optional = bytearray(224)
		struct.pack_into("<H", optional, 0, 0x10B)
		struct.pack_into("<I", optional, 92, 16)
		if dotnet:
			struct.pack_into("<II", optional, 96 + 14 * 8, 0x2000, 72)
		section = bytearray(40)
		section[:8] = b".text\0\0\0"
		struct.pack_into("<IIII", section, 8, 0x100, 0x1000, 0x200, 0x200)
		path.write_bytes(bytes(dos) + b"PE\0\0" + coff + optional + section + bytes(512))
		return path

	return create
