"""Installed EasyMoDD command-line entry point."""

from __future__ import annotations

import argparse
from importlib.metadata import version, PackageNotFoundError

from .config import load_config
from .corpus import audit_corpus


def main() -> None:
    parser = argparse.ArgumentParser(prog="easymodd", description="Configurable training tools")
    parser.add_argument("command", choices=["validate-config", "inspect-config", "version", "audit-corpus"], help="Operation to perform")
    parser.add_argument("--config", default=None, help="Consumer-project TOML config path")
    parser.add_argument("--source", help="Corpus file to audit")
    parser.add_argument("--format", choices=["auto", "text", "jsonl"], default="auto")
    parser.add_argument("--text-field", help="Text field path to hash in JSONL records")
    parser.add_argument("--deduplicated-output", help="Optional output path for unique records")
    parser.add_argument("--json", action="store_true", help="Print a JSON result")
    args = parser.parse_args()
    if args.command == "version":
        try:
            print(version("easymodd"))
        except PackageNotFoundError:
            print("0.1.0 (source checkout)")
        return
    if args.command == "audit-corpus":
        if not args.source:
            parser.error("--source is required for audit-corpus")
        report = audit_corpus(args.source, format=args.format, text_field=args.text_field,
                              deduplicated_output=args.deduplicated_output)
        print(report.to_json() if args.json else report.to_dict())
        return
    if not args.config:
        parser.error(f"--config is required for {args.command}; EasyMoDD does not ship a project config")
    if args.command == "validate-config":
        config = load_config(args.config)
        print(f"Valid EasyMoDD config: {config.source}")
    else:
        config = load_config(args.config)
        for section, values in config.values.items():
            print(f"[{section}]")
            for key, value in values.items():
                print(f"{key} = {value!r}")


if __name__ == "__main__":
    main()
