"""Installed EasyModel command line interface."""

from __future__ import annotations

import argparse
import json
import os

from .artifacts import detect_artifact
from .passport import compare_passports, create_passport


def main() -> None:
    parser = argparse.ArgumentParser(prog="easymodel", description="Inspect and decompile software artifacts safely")
    parser.add_argument("operation", choices=["detect", "inspect", "passport", "compare", "backends", "encrypt", "decrypt", "compile", "web", "app", "lan", "forward", "simulate"])
    parser.add_argument("path", nargs="?")
    parser.add_argument("--against", help="Second artifact path for passport comparison")
    parser.add_argument("--output", help="Destination path for encryption/decryption")
    parser.add_argument("--passphrase-env", default="EASYMODEL_PASSPHRASE",
                        help="Environment variable containing the encryption passphrase")
    parser.add_argument("--host", help="Bind address (loopback by default)")
    parser.add_argument("--port", type=int, help="Web UI/forwarder port")
    parser.add_argument("--allow-remote", action="store_true", help="Allow non-loopback bind; use behind HTTPS only")
    parser.add_argument("--config", help="Consumer-owned TOML config with optional trusted plugins")
    parser.add_argument("--name", help="Generated app name")
    parser.add_argument("--directory", help="Generated app directory")
    parser.add_argument("--template", choices=["cli", "web", "simulation", "training", "toolbox"], default="cli")
    parser.add_argument("--target-host", help="TCP forward target host")
    parser.add_argument("--target-port", type=int, help="TCP forward target port")
    parser.add_argument("--component", help="Registered simulation component name")
    parser.add_argument("--backend", help="Registered compiler or decompiler backend")
    parser.add_argument("--allow-execution", action="store_true", help="Permit the selected compiler tool to execute")
    parser.add_argument("--dry-run", action="store_true", help="Show compiler command without executing it")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    from .workbench import Workbench
    workbench = Workbench.from_config(args.config) if args.config else Workbench()
    registry = workbench.decompiler_backends
    if args.operation == "backends":
        result = list(registry.names())
    elif args.operation == "web":
        from .webui import serve
        web_config = {}
        if args.config:
            from .config import load_config
            web_config = load_config(args.config).values.get("web", {})
        serve(args.host or web_config.get("host", "127.0.0.1"),
              args.port if args.port is not None else web_config.get("port", 8765),
              token=os.environ.get("EASYMODEL_WEB_TOKEN"),
              allow_remote=args.allow_remote or web_config.get("allow_remote", False),
              workbench=workbench)
        return
    elif args.operation == "app":
        if not args.name or not args.directory:
            parser.error("app requires --name and --directory")
        from .app_builder import create_app
        result = {"project": str(create_app(args.name, args.directory, template=args.template)),
                  "template": args.template}
    elif args.operation == "simulate":
        if not args.config or not args.component:
            parser.error("simulate requires --config and --component")
        if not args.config:
            parser.error("simulate requires --config and --component")
        if not args.component:
            parser.error("simulate requires --component")
        result = workbench.simulate(args.component)
    elif args.operation == "lan":
        from .networking import lan_addresses
        result = {"addresses": lan_addresses(), "note": "LAN reachability also depends on firewall/router policy"}
    elif args.operation == "forward":
        if args.port is None or not args.target_host or args.target_port is None:
            parser.error("forward requires --port, --target-host, and --target-port")
        from .networking import TcpPortForwarder
        forwarder = TcpPortForwarder(args.host or "127.0.0.1", args.port,
                                     args.target_host, args.target_port,
                                     allow_remote=args.allow_remote)
        forwarder.start()
        print(f"Forwarding {args.host or '127.0.0.1'}:{forwarder.bound_port} -> {args.target_host}:{args.target_port}")
        print("This does not configure router port forwarding. Press Ctrl+C to stop.")
        try:
            import threading
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
        finally:
            forwarder.close()
        return
    elif args.operation == "detect":
        if not args.path:
            parser.error("path is required for this operation")
        artifact = detect_artifact(args.path)
        result = {"path": str(artifact.path), "kind": artifact.kind.value, "size": artifact.size}
    elif args.operation == "inspect":
        if not args.path:
            parser.error("path is required for this operation")
        artifact = detect_artifact(args.path)
        result = registry.select(artifact).inspect(artifact).to_dict()
    elif args.operation == "passport":
        if not args.path:
            parser.error("path is required for this operation")
        result = create_passport(args.path).to_dict()
    elif args.operation == "compare":
        if not args.path or not args.against:
            parser.error("compare requires an input path and --against path")
        result = compare_passports(create_passport(args.path), create_passport(args.against))
    elif args.operation == "compile":
        if not args.path or not args.output or not args.backend:
            parser.error("compile requires a source path, --output, and --backend")
        from .security import SecurityPolicy
        policy = SecurityPolicy(allow_execution=args.allow_execution)
        result = workbench.compiler_backends.get(args.backend).compile(
            args.path, output=args.output, policy=policy,
            dry_run=args.dry_run,
        ).to_dict()
    elif args.operation in {"encrypt", "decrypt"}:
        if not args.path or not args.output:
            parser.error(f"{args.operation} requires an input path and --output")
        passphrase = os.environ.get(args.passphrase_env)
        if not passphrase:
            parser.error(f"Set the passphrase in environment variable {args.passphrase_env}; do not put it in command history")
        from .workbench import Workbench
        workbench = Workbench()
        operation = workbench.encrypt if args.operation == "encrypt" else workbench.decrypt
        result = {"path": str(operation(args.path, args.output, passphrase))}
    print(json.dumps(result, indent=2, default=str) if args.json else result)


if __name__ == "__main__":
    main()
