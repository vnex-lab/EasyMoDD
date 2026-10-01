"""Deterministic discrete-event simulation primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import random
from typing import Any, Callable

from .events import EventBus, events


@dataclass(order=True, frozen=True)
class SimulationEvent:
    time: float
    priority: int
    sequence: int
    kind: str = field(compare=False)
    payload: dict[str, Any] = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class SimulationResult:
    processed_events: int
    final_time: float
    pending_events: int
    stopped_by_limit: bool


class DiscreteEventSimulator:
    """Stable priority-queue simulation with deterministic random state."""

    def __init__(self, *, seed: int = 0, event_bus: EventBus | None = None):
        self.now = 0.0
        self.random = random.Random(seed)
        self.event_bus = event_bus or events
        self._queue: list[SimulationEvent] = []
        self._handlers: dict[str, Callable] = {}
        self._sequence = 0
        self.processed_events = 0

    def on(self, kind: str, handler: Callable) -> None:
        if not kind or kind in self._handlers:
            raise ValueError(f"Simulation event handler '{kind}' is empty or already registered")
        self._handlers[kind] = handler

    def schedule(self, at: float, kind: str, payload: dict | None = None, *, priority: int = 0) -> SimulationEvent:
        if at < self.now:
            raise ValueError("Cannot schedule an event in the past")
        if kind not in self._handlers:
            raise ValueError(f"No handler is registered for event kind '{kind}'")
        self._sequence += 1
        event = SimulationEvent(float(at), int(priority), self._sequence, kind, payload or {})
        heapq.heappush(self._queue, event)
        return event

    def run(self, *, until: float | None = None, max_events: int = 1_000_000) -> SimulationResult:
        if max_events < 1:
            raise ValueError("max_events must be at least 1")
        processed = 0
        while self._queue and processed < max_events:
            event = self._queue[0]
            if until is not None and event.time > until:
                break
            heapq.heappop(self._queue)
            self.now = event.time
            self.event_bus.publish("simulation", "simulation.event.started", {
                "kind": event.kind, "time": event.time, "sequence": event.sequence,
            })
            self._handlers[event.kind](self, event)
            processed += 1
            self.processed_events += 1
        stopped = processed == max_events and bool(self._queue)
        if until is not None and (not self._queue or self._queue[0].time > until):
            self.now = max(self.now, float(until))
        return SimulationResult(processed, self.now, len(self._queue), stopped)