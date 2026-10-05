"""Dependency-aware workflows joining EasyModel and EasyMoDD operations."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable

from .events import EventBus, events


@dataclass(frozen=True)
class WorkflowStep:
    name: str
    action: Callable[["WorkflowContext"], Any]
    depends_on: tuple[str, ...] = ()
    retries: int = 0


@dataclass
class WorkflowContext:
    inputs: dict[str, Any] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)

    def get(self, name: str, default=None):
        if name in self.results:
            return self.results[name]
        return self.inputs.get(name, default)

    def require(self, name: str):
        value = self.get(name)
        if value is None:
            raise KeyError(f"Workflow value '{name}' is not available")
        return value


@dataclass(frozen=True)
class WorkflowReport:
    name: str
    statuses: dict[str, str]
    errors: dict[str, str]
    duration_seconds: float
    step_durations: dict[str, float]
    context: WorkflowContext = field(repr=False, compare=False)

    @property
    def ok(self) -> bool:
        return not self.errors and all(status == "succeeded" for status in self.statuses.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "statuses": dict(self.statuses),
            "errors": dict(self.errors),
            "duration_seconds": self.duration_seconds,
            "step_durations": dict(self.step_durations),
            "outputs": {name: type(value).__name__ for name, value in self.context.results.items()},
        }


class Workflow:
    """Run a deterministic DAG of registered Python actions.

    Actions receive one ``WorkflowContext`` and return a value under their step
    name. Retries are bounded and never hide a failed final attempt. A workflow
    can call training, data, simulation, compiler, or artifact APIs in the same
    graph without the library forcing those APIs into one implementation.
    """

    def __init__(self, name: str, *, event_bus: EventBus | None = None,
                 max_steps: int = 500):
        if not name.strip():
            raise ValueError("Workflow name cannot be empty")
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self.name = name
        self.event_bus = event_bus or events
        self.max_steps = max_steps
        self._steps: dict[str, WorkflowStep] = {}

    def add(self, name: str, action: Callable[[WorkflowContext], Any], *,
            depends_on: tuple[str, ...] | list[str] = (), retries: int = 0) -> "Workflow":
        if len(self._steps) >= self.max_steps:
            raise ValueError(f"Workflow exceeds configured step limit ({self.max_steps})")
        if not name or name in self._steps:
            raise ValueError(f"Workflow step name '{name}' is empty or already used")
        if not callable(action):
            raise TypeError("Workflow action must be callable")
        if retries < 0 or retries > 10:
            raise ValueError("retries must be between 0 and 10")
        self._steps[name] = WorkflowStep(name, action, tuple(depends_on), retries)
        return self

    def validate(self) -> tuple[str, ...]:
        names = set(self._steps)
        unknown = sorted({dep for step in self._steps.values() for dep in step.depends_on if dep not in names})
        if unknown:
            raise ValueError(f"Workflow references unknown dependency/dependencies: {', '.join(unknown)}")
        ordered: list[str] = []
        completed: set[str] = set()
        while len(ordered) < len(self._steps):
            ready = [step.name for step in self._steps.values()
                     if step.name not in completed and set(step.depends_on) <= completed]
            if not ready:
                raise ValueError("Workflow contains a dependency cycle")
            ordered.extend(ready)
            completed.update(ready)
        return tuple(ordered)

    def run(self, *, inputs: dict[str, Any] | None = None,
            fail_fast: bool = True) -> WorkflowReport:
        order = self.validate()
        started = time.perf_counter()
        context = WorkflowContext(inputs=dict(inputs or {}))
        statuses: dict[str, str] = {}
        errors: dict[str, str] = {}
        durations: dict[str, float] = {}
        self.event_bus.publish("workflow", "workflow.started", {"name": self.name, "steps": len(order)})

        for name in order:
            step = self._steps[name]
            failed_dependencies = [dep for dep in step.depends_on if statuses.get(dep) != "succeeded"]
            if failed_dependencies:
                statuses[name] = "skipped"
                self.event_bus.publish("workflow", "workflow.step.skipped", {
                    "workflow": self.name, "step": name, "dependencies": failed_dependencies,
                })
                continue

            step_started = time.perf_counter()
            self.event_bus.publish("workflow", "workflow.step.started", {"workflow": self.name, "step": name})
            for attempt in range(step.retries + 1):
                try:
                    context.results[name] = step.action(context)
                    statuses[name] = "succeeded"
                    durations[name] = time.perf_counter() - step_started
                    self.event_bus.publish("workflow", "workflow.step.completed", {
                        "workflow": self.name, "step": name, "attempts": attempt + 1,
                        "output_type": type(context.results[name]).__name__,
                    })
                    break
                except Exception as error:
                    if attempt < step.retries:
                        self.event_bus.publish("workflow", "workflow.step.retry", {
                            "workflow": self.name, "step": name, "attempt": attempt + 1,
                            "error_type": type(error).__name__,
                        })
                        continue
                    statuses[name] = "failed"
                    errors[name] = f"{type(error).__name__}: {error}"
                    durations[name] = time.perf_counter() - step_started
                    self.event_bus.publish("workflow", "workflow.step.failed", {
                        "workflow": self.name, "step": name, "error_type": type(error).__name__,
                    })
            if fail_fast and statuses[name] == "failed":
                for remaining in order[order.index(name) + 1:]:
                    statuses.setdefault(remaining, "skipped")
                break

        report = WorkflowReport(self.name, statuses, errors, time.perf_counter() - started, durations, context)
        self.event_bus.publish("workflow", "workflow.completed", report.to_dict())
        return report