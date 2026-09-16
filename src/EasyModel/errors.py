"""Readable errors raised by EasyModel."""


class EasyModelError(Exception):
    """Base error for EasyModel operations."""


class ConfigurationError(EasyModelError):
    pass


class ArtifactError(EasyModelError):
    pass


class BackendUnavailableError(EasyModelError):
    pass


class CompilationError(EasyModelError):
    pass


class DecompilationError(EasyModelError):
    pass


class SecurityPolicyError(EasyModelError):
    pass
