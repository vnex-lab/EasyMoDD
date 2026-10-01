"""Small thread-safe event bus shared by training and artifact workflows."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import threading
from typing import Callable


@dataclass(frozen=True)
class WorkbenchEvent:
    sequence: int
    timestamp: str
    source: str
    name: str
    payload: dict


class EventBus:
    """In-process event bus with bounded history and secret-value redaction."""

    def __init__(self, history_limit: int = 1000):
        if history_limit < 1:
            raise ValueError("history_limit must be at least 1")
        self._history = deque(maxlen=history_limit)
        self._subscribers: dict[int, Callable[[WorkbenchEvent], None]] = {}
        self._lock = threading.RLock()
        self._sequence = 0
        self._subscriber_id = 0

    def subscribe(self, callback: Callable[[WorkbenchEvent], None]) -> Callable[[], None]:
        with self._lock:
            self._subscriber_id += 1
            identifier = self._subscriber_id
            self._subscribers[identifier] = callback

        def unsubscribe() -> None:
            with self._lock:
                self._subscribers.pop(identifier, None)
        return unsubscribe

    def publish(self, source: str, name: str, payload: dict | None = None) -> WorkbenchEvent:
        with self._lock:
            self._sequence += 1
            event = WorkbenchEvent(
                sequence=self._sequence,
                timestamp=datetime.now(timezone.utc).isoformat(),
                source=source,
                name=name,
                payload=_redact(payload or {}),
            )
            self._history.append(event)
            subscribers = tuple(self._subscribers.values())
        for callback in subscribers:
            try:
                callback(event)
            except Exception:
                continue
        return event

    def recent(self, limit: int = 100) -> list[dict]:
        if limit < 1:
            return []
        with self._lock:
            events = list(self._history)[-limit:]
        return [asdict(event) for event in events]


def _redact(value):
    sensitive = ("password", "secret", "token", "api_key", "apikey", "passphrase")
    if isinstance(value, dict):
        entries = list(value.items())
        result = {
            str(key): "<redacted>" if any(item in str(key).lower() for item in sensitive)
            else _redact(item)
            for key, item in entries[:100]
        }
        if len(entries) > 100:
            result["_truncated_fields"] = len(entries) - 100
        return result
    if isinstance(value, (list, tuple)):
        items = [_redact(item) for item in value[:100]]
        if len(value) > 100:
            items.append(f"<truncated {len(value) - 100} items>")
        return items
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and len(value) > 4096:
            return value[:4096] + "<truncated>"
        return value
    rendered = repr(value)
    return rendered[:4096] + ("<truncated>" if len(rendered) > 4096 else "")


events = EventBus()
