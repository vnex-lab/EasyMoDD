"""User-facing EasyModel artifact inspection and decompilation handler."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

_source_root = Path(__file__).resolve().parent / "src"
if str(_source_root) not in sys.path:
    sys.path.insert(0, str(_source_root))

from EasyModel.artifacts import detect_artifact
from EasyModel.backends import BackendRegistry
from EasyModel.decompilers import JarInspector
from EasyModel.reports import ArtifactReport
from EasyModel.security import SecurityPolicy
from EasyModel.events import EventBus, events


class EzDecompiler:
    """Facade for safe inspection and configurable decompiler backends."""

    def __init__(self, *, policy: SecurityPolicy | None = None,
                 registry: BackendRegistry | None = None, event_bus: EventBus | None = None):
        self.policy = policy or SecurityPolicy()
        self.event_bus = event_bus or events
        self.registry = registry or BackendRegistry()
        if not self.registry.names():
            self.registry.register("jar-inspector", JarInspector())

    def detect(self, path: str | Path):
        artifact = detect_artifact(path)
        self.event_bus.publish("easymodel", "artifact.detected", {"path": str(artifact.path), "kind": artifact.kind.value})
        return artifact

    def backends(self) -> tuple[str, ...]:
        return self.registry.names()

    def inspect(self, path: str | Path, *, backend: str | None = None) -> ArtifactReport:
        artifact = detect_artifact(path)
        self.event_bus.publish("easymodel", "artifact.inspect.started", {"path": str(artifact.path), "kind": artifact.kind.value})
        selected = self.registry.select(artifact, backend)
        report = selected.inspect(artifact, policy=self.policy)
        self.event_bus.publish("easymodel", "artifact.inspect.completed", {"path": str(artifact.path), "backend": report.backend})
        return report

    def decompile(self, path: str | Path, *, backend: str | None = None,
                  output_directory: str | Path | None = None, dry_run: bool = False) -> ArtifactReport:
        artifact = detect_artifact(path)
        self.event_bus.publish("easymodel", "artifact.decompile.started", {"path": str(artifact.path), "kind": artifact.kind.value})
        selected = self.registry.select(artifact, backend)
        report = selected.decompile(artifact, policy=self.policy,
                        output_directory=output_directory, dry_run=dry_run)
        self.event_bus.publish("easymodel", "artifact.decompile.completed", {"path": str(artifact.path), "backend": report.backend})
        return report


def main() -> None:
    parser = argparse.ArgumentParser(prog="EzDecompiler")
    parser.add_argument("operation", choices=["detect", "inspect", "decompile", "backends"])
    parser.add_argument("path", nargs="?")
    parser.add_argument("--backend")
    parser.add_argument("--output-directory")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    handler = EzDecompiler()
    if args.operation == "backends":
        result = list(handler.backends())
    elif args.operation == "detect":
        result = handler.detect(args.path).__dict__
        result["kind"] = result["kind"].value
    elif args.operation == "inspect":
        result = handler.inspect(args.path, backend=args.backend).to_dict()
    else:
        result = handler.decompile(args.path, backend=args.backend,
                                   output_directory=args.output_directory,
                                   dry_run=args.dry_run).to_dict()
    print(json.dumps(result, indent=2, default=str) if args.json else result)


if __name__ == "__main__":
    main()
