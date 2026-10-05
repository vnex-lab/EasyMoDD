"""Cross-package facade joining corpus, artifact, and model-training tasks."""

from __future__ import annotations

from easymodd.corpus import audit_corpus
from easymodd.visualization import summarize_history

from .events import EventBus, events
from .extensions import ApiRoute, ComponentRegistry, PluginLoader, WebPage
from .backends import BackendRegistry
from .compilers import CompilerRegistry
from .decompilers import JarInspector
from .passport import create_passport


class Workbench:
    """Coordinate existing EasyMoDD and EasyModel APIs through shared events."""

    def __init__(self, event_bus: EventBus | None = None):
        self.event_bus = event_bus or events
        self.components = ComponentRegistry()
        self.pages: dict[str, WebPage] = {}
        self.api_routes: dict[str, ApiRoute] = {}
        self.decompiler_backends = BackendRegistry()
        self.decompiler_backends.register("jar-inspector", JarInspector())
        self.compiler_backends = CompilerRegistry()
        from easymodd.models import build_mlp
        from .app_builder import create_app
        from .simulation import DiscreteEventSimulator
        self.components.register("models", "mlp", build_mlp)
        self.components.register("simulations", "discrete-event", DiscreteEventSimulator)
        self.components.register("decompilers", "jar-inspector", JarInspector)
        self.components.register("apps", "scaffold", create_app)
        self.components.register("tools", "corpus-audit", audit_corpus)
        self.components.register("tools", "artifact-passport", create_passport)

    def register_decompiler_backend(self, backend):
        return self.decompiler_backends.register(backend.name, backend)

    def register_compiler_backend(self, backend):
        return self.compiler_backends.register(backend)

    def register_component(self, group: str, name: str, provider):
        self.event_bus.publish("workbench", "component.registered", {"group": group, "name": name})
        return self.components.register(group, name, provider)

    def run_component(self, group: str, name: str, **options):
        self.event_bus.publish("workbench", "component.started", {"group": group, "name": name})
        try:
            result = self.components.create(group, name, **options)
        except Exception as error:
            self.event_bus.publish("workbench", "component.failed", {"group": group, "name": name, "error": str(error)})
            raise
        self.event_bus.publish("workbench", "component.completed", {"group": group, "name": name})
        return result

    def simulate(self, name: str, *, until: float | None = None,
                 max_events: int = 1_000_000, **options):
        simulator = self.run_component("simulations", name, **options)
        if not callable(getattr(simulator, "run", None)):
            raise TypeError(f"Simulation component '{name}' must return an object with run()")
        self.event_bus.publish("workbench", "simulation.run.started", {"name": name})
        try:
            result = simulator.run(until=until, max_events=max_events)
        except Exception as error:
            self.event_bus.publish("workbench", "simulation.run.failed", {"name": name, "error": str(error)})
            raise
        self.event_bus.publish("workbench", "simulation.run.completed", {
            "name": name, "processed_events": result.processed_events,
            "final_time": result.final_time,
        })
        return result

    def register_page(self, page: WebPage):
        if page.path in self.pages:
            raise ValueError(f"Web page '{page.path}' is already registered")
        self.pages[page.path] = page
        self.event_bus.publish("workbench", "page.registered", {"path": page.path, "title": page.title})

    def register_api_route(self, route: ApiRoute):
        if route.path in self.api_routes:
            raise ValueError(f"API route '{route.path}' is already registered")
        self.api_routes[route.path] = route
        self.event_bus.publish("workbench", "api_route.registered", {"path": route.path})

    def run_api_route(self, path: str, payload: dict):
        route = self.api_routes[path]
        self.event_bus.publish("workbench", "api_route.started", {"path": path})
        try:
            result = route.handler(payload, self)
        except Exception as error:
            self.event_bus.publish("workbench", "api_route.failed", {"path": path, "error": str(error)})
            raise
        self.event_bus.publish("workbench", "api_route.completed", {"path": path})
        return result

    def load_plugins(self, modules: list[str], *, enabled: bool = False):
        return PluginLoader.load(modules, self, enabled=enabled)

    def run_workflow(self, workflow, *, inputs=None, fail_fast: bool = True):
        self.event_bus.publish("workbench", "workflow.dispatched", {"name": workflow.name})
        return workflow.run(inputs=inputs, fail_fast=fail_fast)

    @classmethod
    def from_config(cls, path: str):
        from .config import load_config
        config = load_config(path)
        workbench = cls()
        plugin_values = config.values.get("plugins", {})
        workbench.load_plugins(plugin_values.get("modules", []), enabled=plugin_values.get("enabled", False))
        return workbench

    def audit_corpus(self, source, **options):
        self.event_bus.publish("easymodd", "corpus.audit.started", {"source": str(source)})
        try:
            report = audit_corpus(source, **options)
        except Exception as error:
            self.event_bus.publish("easymodd", "corpus.audit.failed", {"error": str(error)})
            raise
        self.event_bus.publish("easymodd", "corpus.audit.completed", report.to_dict())
        return report

    def passport(self, source, **options):
        self.event_bus.publish("easymodel", "artifact.passport.started", {"source": str(source)})
        try:
            report = create_passport(source, **options)
        except Exception as error:
            self.event_bus.publish("easymodel", "artifact.passport.failed", {"error": str(error)})
            raise
        self.event_bus.publish("easymodel", "artifact.passport.completed", {
            "path": report.path, "kind": report.kind, "size_bytes": report.size_bytes,
            "sha256": report.sha256, "warning_count": len(report.warnings),
        })
        return report

    def encrypt(self, source, destination, passphrase, **options):
        from .encryption import encrypt_file
        self.event_bus.publish("easymodel", "artifact.encryption.started", {"source": str(source), "destination": str(destination)})
        path = encrypt_file(source, destination, passphrase, **options)
        self.event_bus.publish("easymodel", "artifact.encryption.completed", {"source": str(source), "destination": str(path)})
        return path

    def decrypt(self, source, destination, passphrase):
        from .encryption import decrypt_file
        self.event_bus.publish("easymodel", "artifact.decryption.started", {"source": str(source), "destination": str(destination)})
        path = decrypt_file(source, destination, passphrase)
        self.event_bus.publish("easymodel", "artifact.decryption.completed", {"source": str(source), "destination": str(path)})
        return path

    def train(self, model, trainer, dataset, validation_data=None):
        self.event_bus.publish("easymodd", "training.started", {"model": type(model).__name__})
        try:
            history = trainer.fit(model, dataset, validation_data=validation_data)
        except Exception as error:
            self.event_bus.publish("easymodd", "training.failed", {"error": str(error)})
            raise
        self.event_bus.publish("easymodd", "training.completed", {
            "model": type(model).__name__,
            "history": summarize_history(history),
        })
        return history
