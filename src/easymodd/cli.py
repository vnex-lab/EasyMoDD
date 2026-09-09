"""Installed EasyMoDD command-line entry point."""

from __future__ import annotations

import argparse
from importlib.metadata import version, PackageNotFoundError

from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(prog="easymodd", description="Configurable training tools")
    parser.add_argument("command", choices=["validate-config", "inspect-config", "version"], help="Operation to perform")
    parser.add_argument("--config", default=None, help="Consumer-project TOML config path")
    args = parser.parse_args()
    if args.command == "version":
        try:
            print(version("easymodd"))
        except PackageNotFoundError:
            print("0.1.0 (source checkout)")
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
