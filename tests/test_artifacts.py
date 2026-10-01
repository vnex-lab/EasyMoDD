from __future__ import annotations

import zipfile

from EasyModel import ArtifactKind, compare_passports, create_passport, detect_artifact


def test_jar_and_class_passports(tmp_path):
    jar = tmp_path / "demo.jar"
    with zipfile.ZipFile(jar, "w") as archive:
        archive.writestr("META-INF/MANIFEST.MF", "Main-Class: demo.Main\n")
        archive.writestr("demo/Main.class", b"\xca\xfe\xba\xbe\x00\x00\x00\x41")
    artifact = detect_artifact(jar)
    passport = create_passport(jar)
    assert artifact.kind is ArtifactKind.JAR
    assert passport.details["class_count"] == 1
    assert passport.details["manifest_present"] is True
    assert compare_passports(passport, create_passport(jar))["identical_content"]

    standalone = tmp_path / "Main.class"
    standalone.write_bytes(b"\xca\xfe\xba\xbe\x00\x00\x00\x41")
    class_passport = create_passport(standalone)
    assert detect_artifact(standalone).kind is ArtifactKind.CLASS
    assert class_passport.details["major_version"] == 65


def test_pe_dll_and_dotnet_detection(tmp_path, make_pe_file):
    dll = make_pe_file(tmp_path / "native.dll")
    passport = create_passport(dll)
    assert detect_artifact(dll).kind is ArtifactKind.PE_DLL
    assert passport.details["sections"] == [".text"]

    executable = make_pe_file(tmp_path / "native.exe", dll=False)
    assert detect_artifact(executable).kind is ArtifactKind.PE_EXE

    assembly = make_pe_file(tmp_path / "managed.dll", dotnet=True)
    assert create_passport(assembly).kind == ArtifactKind.DOTNET_ASSEMBLY.value


def test_cpp_pyc_and_wasm_format_detection(tmp_path):
    source = tmp_path / "bridge.cpp"
    source.write_text("int add(int a, int b) { return a + b; }\n", encoding="utf-8")
    assert detect_artifact(source).kind is ArtifactKind.SOURCE

    bytecode = tmp_path / "module.pyc"
    bytecode.write_bytes(b"\x00\x00\x00\x00" + bytes(12))
    assert detect_artifact(bytecode).kind is ArtifactKind.PYTHON_BYTECODE
    assert create_passport(bytecode).details["code_executed"] is False

    wasm = tmp_path / "module.wasm"
    wasm.write_bytes(b"\x00asm\x01\x00\x00\x00\x01\x01\x00")
    report = create_passport(wasm)
    assert report.kind == ArtifactKind.WASM.value
    assert report.details["sections"] == [{"id": 1, "size_bytes": 1}]


def test_archive_traversal_is_reported_not_extracted(tmp_path):
    jar = tmp_path / "unsafe.jar"
    with zipfile.ZipFile(jar, "w") as archive:
        archive.writestr("../outside.txt", "not extracted")
        archive.writestr("Demo.class", b"\xca\xfe\xba\xbe")
    passport = create_passport(jar)
    assert passport.details["unsafe_entry_names"] == ["../outside.txt"]
    assert any("traversal" in warning for warning in passport.warnings)
