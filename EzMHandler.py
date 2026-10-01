"""General EasyModel facade for artifact tooling and model-library access."""

from __future__ import annotations

from EzDecompiler import EzDecompiler
from EasyModel.workbench import Workbench
from EasyModel.app_builder import create_app
from EasyModel.networking import TcpPortForwarder, lan_addresses


class EzMHandler:
    """Single entry point that delegates artifact work to EzDecompiler."""

    def __init__(self, *, workbench=None, **kwargs):
        self.workbench = workbench or Workbench()
        kwargs.setdefault("event_bus", self.workbench.event_bus)
        kwargs.setdefault("registry", self.workbench.decompiler_backends)
        self.decompiler = EzDecompiler(**kwargs)

    def detect(self, path):
        return self.decompiler.detect(path)

    def inspect(self, path, **kwargs):
        return self.decompiler.inspect(path, **kwargs)

    def decompile(self, path, **kwargs):
        return self.decompiler.decompile(path, **kwargs)

    def compile(self, backend, source, **kwargs):
        self.workbench.event_bus.publish("easymodel", "artifact.compile.started", {"source": str(source), "backend": backend})
        result = self.workbench.compiler_backends.get(backend).compile(source, policy=self.decompiler.policy, **kwargs)
        self.workbench.event_bus.publish("easymodel", "artifact.compile.completed", {
            "source": str(source), "backend": backend, "ok": result.ok,
        })
        return result

    def train(self, model, trainer, dataset, validation_data=None):
        return self.workbench.train(model, trainer, dataset, validation_data)

    def audit_corpus(self, source, **options):
        return self.workbench.audit_corpus(source, **options)

    def simulate(self, name, **options):
        return self.workbench.simulate(name, **options)

    def register_component(self, group, name, provider):
        return self.workbench.register_component(group, name, provider)

    def register_page(self, page):
        return self.workbench.register_page(page)

    def register_api_route(self, route):
        return self.workbench.register_api_route(route)

    def create_app(self, name, directory, *, template="cli"):
        return create_app(name, directory, template=template)

    @staticmethod
    def lan_addresses():
        return lan_addresses()

    def load_plugins(self, modules, *, enabled=False):
        return self.workbench.load_plugins(modules, enabled=enabled)

    def serve_web(self, **options):
        from EasyModel.webui import serve
        return serve(workbench=self.workbench, **options)
