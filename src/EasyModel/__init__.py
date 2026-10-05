"""Safe, configurable artifact inspection and toolchain adapters."""

from .artifacts import Artifact, ArtifactKind, detect_artifact
from .backends import BackendRegistry, DecompilerBackend
from .compilers import CompilerBackend, CompilerRegistry, compilers
from .config import EasyModelConfig, load_config
from .errors import (
    ArtifactError,
    BackendUnavailableError,
    CompilationError,
    ConfigurationError,
    DecompilationError,
    EasyModelError,
    SecurityPolicyError,
)
from .reports import ArtifactReport, BackendResult
from .passport import ArtifactPassport, compare_passports, create_passport
from .security import SecurityPolicy
from .events import EventBus, WorkbenchEvent, events
from .workbench import Workbench
from .encryption import EncryptionError, decrypt_file, encrypt_file
from .webui import create_server, serve
from .extensions import ApiRoute, ComponentRegistry, PluginLoader, WebPage
from .simulation import DiscreteEventSimulator, SimulationEvent, SimulationResult
from .mathx import IntegrationEstimate, finite_difference_gradient, integrate_simpson, monte_carlo_integrate, pairwise_distances, solve_linear, stable_softmax
from .app_builder import create_app
from .networking import TcpPortForwarder, lan_addresses
from .workflows import Workflow, WorkflowContext, WorkflowReport, WorkflowStep
from easymodd.secret_store import SecretStore

__all__ = [
    "Artifact",
    "ArtifactError",
    "ArtifactKind",
    "ApiRoute",
    "ArtifactReport",
    "ArtifactPassport",
    "BackendRegistry",
    "BackendResult",
    "BackendUnavailableError",
    "CompilerBackend",
    "CompilerRegistry",
    "ComponentRegistry",
    "CompilationError",
    "ConfigurationError",
    "DecompilerBackend",
    "DecompilationError",
    "EasyModelConfig",
    "EasyModelError",
    "EncryptionError",
    "EventBus",
    "IntegrationEstimate",
    "SecurityPolicy",
    "SecurityPolicyError",
    "SecretStore",
    "DiscreteEventSimulator",
    "PluginLoader",
    "SimulationEvent",
    "SimulationResult",
    "TcpPortForwarder",
    "WebPage",
    "Workbench",
    "WorkbenchEvent",
    "Workflow",
    "WorkflowContext",
    "WorkflowReport",
    "WorkflowStep",
    "detect_artifact",
    "decrypt_file",
    "encrypt_file",
    "compare_passports",
    "create_passport",
    "load_config",
    "create_server",
    "create_app",
    "integrate_simpson",
    "finite_difference_gradient",
    "lan_addresses",
    "pairwise_distances",
    "monte_carlo_integrate",
    "compilers",
    "events",
    "solve_linear",
    "stable_softmax",
    "serve",
]
