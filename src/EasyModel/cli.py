"""Installed EasyModel command line interface."""

from __future__ import annotations

import argparse
import json

from .artifacts import detect_artifact
from .decompilers import JarInspector
from .backends import BackendRegistry


def main() -> None:
    parser = argparse.ArgumentParser(prog="easymodel", description="Inspect and decompile software artifacts safely")
    parser.add_argument("operation", choices=["detect", "inspect", "backends"])
    parser.add_argument("path", nargs="?")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    registry = BackendRegistry()
    registry.register("jar-inspector", JarInspector())
    if args.operation == "backends":
        result = list(registry.names())
    else:
        if not args.path:
            parser.error("path is required for this operation")
        artifact = detect_artifact(args.path)
        if args.operation == "detect":
            result = {"path": str(artifact.path), "kind": artifact.kind.value, "size": artifact.size}
        else:
            result = registry.select(artifact).inspect(artifact).to_dict()
    print(json.dumps(result, indent=2, default=str) if args.json else result)


if __name__ == "__main__":
    main()
