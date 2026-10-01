"""Explicit LAN hosting helpers and bounded local TCP port forwarding."""

from __future__ import annotations

import ipaddress
import select
import socket
import threading

from .errors import SecurityPolicyError
from .events import EventBus, events


def lan_addresses() -> tuple[str, ...]:
    """Return private interface addresses visible to Python without probing hosts."""
    found = set()
    try:
        records = socket.getaddrinfo(socket.gethostname(), None, type=socket.SOCK_STREAM)
    except OSError:
        records = []
    for record in records:
        address = record[4][0].split("%", 1)[0]
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError:
            continue
        if parsed.is_private and not parsed.is_loopback and not parsed.is_link_local:
            found.add(address)
    return tuple(sorted(found))


class TcpPortForwarder:
    """Forward TCP connections with explicit bind exposure and connection caps.

    This forwards a local TCP port only. It does not modify router/firewall
    rules, configure UPnP, or create an Internet tunnel.
    """

    def __init__(self, listen_host: str, listen_port: int, target_host: str,
                 target_port: int, *, allow_remote: bool = False,
                 max_connections: int = 32, buffer_size: int = 65536,
                 event_bus: EventBus | None = None):
        try:
            address = ipaddress.ip_address(listen_host)
            loopback = address.is_loopback
        except ValueError:
            loopback = listen_host.lower() == "localhost"
        if not loopback and not allow_remote:
            raise SecurityPolicyError("Non-loopback port forwarding requires allow_remote=True")
        if not 0 <= listen_port <= 65535 or not 1 <= target_port <= 65535:
            raise ValueError("listen_port must be 0-65535 and target_port must be 1-65535")
        if max_connections < 1 or buffer_size < 1024:
            raise ValueError("max_connections must be positive and buffer_size at least 1024")
        self.listen_host, self.listen_port = listen_host, listen_port
        self.target_host, self.target_port = target_host, target_port
        self.max_connections, self.buffer_size = max_connections, buffer_size
        self.event_bus = event_bus or events
        self._stop = threading.Event()
        self._slots = threading.BoundedSemaphore(max_connections)
        self._clients: set[socket.socket] = set()
        self._lock = threading.Lock()
        self._listener: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self.bound_port: int | None = None

    def start(self) -> int:
        if self._thread is not None:
            raise RuntimeError("Port forwarder is already started")
        listener = socket.socket(socket.AF_INET6 if ":" in self.listen_host else socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind((self.listen_host, self.listen_port))
            listener.listen(self.max_connections)
            listener.settimeout(0.5)
        except Exception:
            listener.close()
            raise
        self._listener = listener
        self.bound_port = listener.getsockname()[1]
        self._thread = threading.Thread(target=self._accept_loop, name="easymodel-port-forward", daemon=True)
        self._thread.start()
        self.event_bus.publish("network", "port_forward.started", {
            "listen_host": self.listen_host, "listen_port": self.bound_port,
            "target_host": self.target_host, "target_port": self.target_port,
            "max_connections": self.max_connections,
        })
        return self.bound_port

    def close(self) -> None:
        self._stop.set()
        if self._listener is not None:
            self._listener.close()
            self._listener = None
        with self._lock:
            clients = tuple(self._clients)
        for client in clients:
            try:
                client.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        self.event_bus.publish("network", "port_forward.stopped", {
            "listen_host": self.listen_host, "listen_port": self.bound_port,
        })

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _accept_loop(self):
        while not self._stop.is_set() and self._listener is not None:
            try:
                client, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            if not self._slots.acquire(blocking=False):
                client.close()
                continue
            with self._lock:
                self._clients.add(client)
            threading.Thread(target=self._relay, args=(client,), daemon=True).start()

    def _relay(self, client: socket.socket):
        upstream = None
        try:
            upstream = socket.create_connection((self.target_host, self.target_port), timeout=5)
            client.settimeout(30)
            upstream.settimeout(30)
            active = {client, upstream}
            while active and not self._stop.is_set():
                readable, _, _ = select.select(active, [], [], 0.5)
                for source in readable:
                    target = upstream if source is client else client
                    try:
                        data = source.recv(self.buffer_size)
                    except OSError:
                        active.clear()
                        break
                    if not data:
                        active.discard(source)
                        try:
                            target.shutdown(socket.SHUT_WR)
                        except OSError:
                            pass
                    else:
                        target.sendall(data)
        except OSError:
            pass
        finally:
            for item in (client, upstream):
                if item is not None:
                    try:
                        item.close()
                    except OSError:
                        pass
            with self._lock:
                self._clients.discard(client)
            self._slots.release()