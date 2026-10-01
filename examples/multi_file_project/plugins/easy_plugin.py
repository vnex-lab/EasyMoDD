"""Example plugin joining native sources, compiler adapters, and DLL analysis."""

from pathlib import Path

from EasyModel import ApiRoute, ArtifactKind, ArtifactReport, BackendResult, CompilerBackend, WebPage


class NativeMetadataBackend:
    name = "native-metadata"

    def supports(self, artifact):
        return artifact.kind in {ArtifactKind.PE_EXE, ArtifactKind.PE_DLL}

    def inspect(self, artifact, **options):
        with artifact.path.open("rb") as handle:
            prefix = handle.read(64)
        return ArtifactReport(
            operation="inspect",
            artifact=str(artifact.path),
            kind=artifact.kind.value,
            backend=self.name,
            result=BackendResult(ok=True, backend=self.name),
            metadata={"size_bytes": artifact.size, "dos_header_hex": prefix[:2].hex()},
        )

    def decompile(self, artifact, **options):
        return ArtifactReport(
            operation="decompile",
            artifact=str(artifact.path),
            kind=artifact.kind.value,
            backend=self.name,
            result=BackendResult(
                ok=False,
                backend=self.name,
                warnings=["Metadata backend does not recover original native source; use an authorized disassembler backend."],
            ),
        )


def register(workbench):
    native_dir = Path(__file__).resolve().parents[1] / "native"

    def native_source_inventory():
        return sorted(str(path) for path in native_dir.rglob("*") if path.is_file())

    workbench.register_component("tools", "native-source-inventory", native_source_inventory)
    workbench.register_compiler_backend(CompilerBackend("cxx", "c++"))
    workbench.register_decompiler_backend(NativeMetadataBackend())
    workbench.register_page(WebPage(
        "/pages/native-tools",
        "Native tools",
        "<main><h1>Native sources</h1><p>Use the API to inspect registered native artifacts.</p></main>",
    ))
    workbench.register_api_route(ApiRoute(
        "/api/custom/native-health",
        lambda payload, hub: {"ready": True, "registered_tools": hub.components.names("tools")},
    ))
