"""Built-in safe inspectors and optional external decompiler adapters."""

from __future__ import annotations

from pathlib import Path
import zipfile

from .artifacts import Artifact, ArtifactKind
from .process import Command, run_command
from .reports import ArtifactReport, BackendResult
from .security import SecurityPolicy


class JarInspector:
    name = "jar-inspector"

    def supports(self, artifact: Artifact) -> bool:
        return artifact.kind in {ArtifactKind.JAR, ArtifactKind.CLASS}

    def inspect(self, artifact: Artifact, **options) -> ArtifactReport:
        if artifact.kind == ArtifactKind.CLASS:
            metadata = {"magic": "CAFEBABE", "message": "Java class file detected"}
        else:
            with zipfile.ZipFile(artifact.path) as archive:
                names = archive.namelist()
            metadata = {"entries": names, "class_count": sum(name.endswith(".class") for name in names)}
        return ArtifactReport("inspect", str(artifact.path), artifact.kind.value, self.name,
                              BackendResult(True, self.name), metadata)

    def decompile(self, artifact: Artifact, **options) -> ArtifactReport:
        return ArtifactReport(
            "decompile", str(artifact.path), artifact.kind.value, self.name,
            BackendResult(False, self.name, warnings=[
                "Built-in Java inspection does not recover source. Configure CFR or FernFlower for decompilation."
            ]),
        )


class ExternalDecompiler:
    def __init__(self, name: str, executable: str, supported: set[ArtifactKind]):
        self.name, self.executable, self.supported = name, executable, supported

    def supports(self, artifact: Artifact) -> bool:
        return artifact.kind in self.supported

    def inspect(self, artifact: Artifact, **options) -> ArtifactReport:
        return self.decompile(artifact, **options)

    def decompile(self, artifact: Artifact, **options) -> ArtifactReport:
        output = Path(options.get("output_directory", artifact.path.parent / "decompiled"))
        command = Command(self.executable, (str(artifact.path), "--outputdir", str(output)))
        result = run_command(command, policy=options.get("policy", SecurityPolicy()),
                             dry_run=options.get("dry_run", False), backend=self.name)
        return ArtifactReport("decompile", str(artifact.path), artifact.kind.value, self.name, result,
                              {"output_directory": str(output)})
