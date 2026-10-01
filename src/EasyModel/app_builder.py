"""Safe starter-project generator for connected EasyModel workflows."""

from __future__ import annotations

from pathlib import Path
import re


_TEMPLATES = {
    "cli": '"""Application entry point."""\n\ndef main():\n    print("Application ready")\n\nif __name__ == "__main__":\n    main()\n',
    "web": '"""Local EasyModel web workbench."""\n\nfrom EasyModel.config import load_config\nfrom EasyModel.webui import serve\nfrom EasyModel.workbench import Workbench\n\nif __name__ == "__main__":\n    config = load_config("config.toml")\n    web = config.values.get("web", {})\n    serve(web.get("host", "127.0.0.1"), web.get("port", 8765),\n          allow_remote=web.get("allow_remote", False),\n          workbench=Workbench.from_config("config.toml"))\n',
    "simulation": '"""Deterministic simulation starter."""\n\nfrom EasyModel.simulation import DiscreteEventSimulator\n\nsim = DiscreteEventSimulator(seed=42)\ndef tick(simulator, event):\n    print(f"tick at {event.time}")\nsim.on("tick", tick)\nsim.schedule(1.0, "tick")\nprint(sim.run(until=10))\n',
    "training": '"""EasyMoDD training application starter."""\n\nimport numpy as np\nfrom easymodd import ArrayDataset, MSELoss, SGD, Trainer, build_mlp\n\nmodel = build_mlp(1, [8], 1)\ndata = ArrayDataset(np.array([[0.], [1.]], dtype=np.float32), np.array([[0.], [2.]], dtype=np.float32))\nhistory = Trainer(SGD(model.parameters(), 0.01), MSELoss(), max_steps=50).fit(model, data)\nprint(history.values["loss"][-1])\n',
    "toolbox": '"""Connected model and artifact workbench starter."""\n\nfrom EasyModel import Workbench\n\nworkbench = Workbench()\nprint(workbench.components.inventory())\nprint("Use the same Workbench for training, audits, simulations, and artifact tools.")\n',
}


def create_app(name: str, directory: str | Path, *, template: str = "cli") -> Path:
    """Generate a minimal consumer project without overwriting existing files."""
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", name):
        raise ValueError("Project name must start with a letter and use only letters, digits, '_' or '-'")
    if template not in _TEMPLATES:
        raise ValueError(f"Unknown template '{template}'. Available: {', '.join(sorted(_TEMPLATES))}")
    root = Path(directory).expanduser().resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError(f"Refusing to overwrite non-empty project location: {root}")
    root.mkdir(parents=True, exist_ok=True)
    normalized_name = name.lower().replace("_", "-")
    files = {
        "main.py": _TEMPLATES[template],
        "pyproject.toml": (
            f"[project]\nname = \"{normalized_name}\"\nversion = \"0.1.0\"\n"
            "requires-python = \">=3.11\"\ndependencies = [\"easymodd\"]\n"
        ),
        "README.md": f"# {name}\n\nGenerated EasyModel {template} project.\n",
        "config.toml": (
            f"[project]\nname = \"{normalized_name}\"\n\n"
            "[plugins]\nenabled = false\nmodules = []\n\n"
            "[web]\nhost = \"127.0.0.1\"\nport = 8765\nallow_remote = false\n"
        ),
    }
    for relative, content in files.items():
        destination = root / relative
        destination.write_text(content, encoding="utf-8")
    return root