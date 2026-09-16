"""General EasyModel facade for artifact tooling and model-library access."""

from __future__ import annotations

from EzDecompiler import EzDecompiler


class EzMHandler:
    """Single entry point that delegates artifact work to EzDecompiler."""

    def __init__(self, **kwargs):
        self.decompiler = EzDecompiler(**kwargs)

    def detect(self, path):
        return self.decompiler.detect(path)

    def inspect(self, path, **kwargs):
        return self.decompiler.inspect(path, **kwargs)

    def decompile(self, path, **kwargs):
        return self.decompiler.decompile(path, **kwargs)

    def compile(self, backend, source, **kwargs):
        from EasyModel.compilers import compilers
        return compilers.get(backend).compile(source, policy=self.decompiler.policy, **kwargs)
