"""Run the example's registered tools and optionally host its web page."""

from __future__ import annotations

import argparse

from EasyModel import Workbench, serve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--web", action="store_true")
    args = parser.parse_args()
    workbench = Workbench.from_config("config.toml")
    print("Registered components:", workbench.components.inventory())
    print("Native source files:", workbench.run_component("tools", "native-source-inventory"))
    if args.web:
        serve(workbench=workbench)


if __name__ == "__main__":
    main()
