from __future__ import annotations

from pathlib import Path
import shutil
import sys

import pytest

from EasyModel import ArtifactKind, Workbench, detect_artifact
from EasyModel.security import SecurityPolicy


PROJECT = Path(__file__).resolve().parents[1] / "examples" / "multi_file_project"


def test_multifile_plugin_connects_cpp_and_dll_tools(monkeypatch, tmp_path, make_pe_file):
    monkeypatch.syspath_prepend(str(PROJECT))
    workbench = Workbench.from_config(str(PROJECT / "config.toml"))

    inventory = workbench.run_component("tools", "native-source-inventory")
    cpp_file = PROJECT / "native" / "math.cpp"
    assert str(cpp_file) in inventory
    assert detect_artifact(cpp_file).kind is ArtifactKind.SOURCE

    dll_file = make_pe_file(tmp_path / "native.dll")
    dll = detect_artifact(dll_file)
    backend = workbench.decompiler_backends.select(dll, "native-metadata")
    report = backend.inspect(dll)
    assert report.backend == "native-metadata"
    assert report.metadata["dos_header_hex"] == "4d5a"

    assert "/pages/native-tools" in workbench.pages
    assert workbench.run_api_route("/api/custom/native-health", {})["ready"] is True


def test_cpp_backend_dry_run_when_toolchain_exists(monkeypatch, tmp_path):
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("No C++ compiler is installed; format interoperability is tested without executing a toolchain")
    monkeypatch.syspath_prepend(str(PROJECT))
    workbench = Workbench.from_config(str(PROJECT / "config.toml"))
    backend = workbench.compiler_backends.get("cxx")
    result = backend.compile(
        PROJECT / "native" / "math.cpp",
        output=tmp_path / "math-library",
        policy=SecurityPolicy(),
        dry_run=True,
    )
    assert result.ok
    assert result.warnings and "dry_run" in result.warnings[0]
    assert str(PROJECT / "native" / "math.cpp") in result.command
