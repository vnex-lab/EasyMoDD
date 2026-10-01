from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import json
import socket
from socketserver import BaseRequestHandler, ThreadingTCPServer

import pytest

from EasyModel import ApiRoute, EventBus, SecurityPolicyError, TcpPortForwarder, WebPage, Workbench, create_server
from EasyModel.encryption import EncryptionError, decrypt_file, encrypt_file


@pytest.mark.skipif(find_spec("cryptography") is None, reason="install easymodd[security] to run encryption tests")
def test_encryption_roundtrip_and_tamper_failure(tmp_path):
    source = tmp_path / "source.bin"
    encrypted = tmp_path / "source.emenc"
    restored = tmp_path / "restored.bin"
    original = bytes(range(256)) * 100
    source.write_bytes(original)
    encrypt_file(source, encrypted, "unit-test-passphrase-long", chunk_size=4096, iterations=100_000)
    decrypt_file(encrypted, restored, "unit-test-passphrase-long")
    assert restored.read_bytes() == original

    tampered = tmp_path / "tampered.emenc"
    data = bytearray(encrypted.read_bytes())
    data[-25] ^= 1
    tampered.write_bytes(data)
    with pytest.raises(EncryptionError, match="Authentication failed"):
        decrypt_file(tampered, tmp_path / "partial.bin", "unit-test-passphrase-long")
    assert not (tmp_path / "partial.bin").exists()


def test_event_bus_redacts_secrets_and_bounds_history():
    bus = EventBus(history_limit=2)
    bus.publish("test", "one", {"api_token": "sensitive", "label": "visible"})
    bus.publish("test", "two")
    bus.publish("test", "three")
    events = bus.recent()
    assert len(events) == 2
    assert events[0]["name"] == "two"
    assert events[1]["name"] == "three"
    bus = EventBus()
    event = bus.publish("test", "redact", {"api_token": "sensitive"})
    assert event.payload["api_token"] == "<redacted>"


def test_authenticated_custom_page_and_api():
    workbench = Workbench()
    workbench.register_page(WebPage("/pages/test", "Test page", "<h1>Test</h1>"))
    workbench.register_api_route(ApiRoute("/api/custom/double", lambda body, hub: {"value": body["value"] * 2}))
    server, token = create_server(port=0, token="test-web-session", workbench=workbench)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        with pytest.raises(HTTPError) as error:
            urlopen(base + "/api/events")
        assert error.value.code == 401
        pages = json.loads(urlopen(Request(base + "/api/pages", headers=headers)).read())
        assert pages[0]["path"] == "/pages/test"
        body = json.dumps({"value": 3}).encode()
        request = Request(base + "/api/custom/double", data=body,
                          headers={**headers, "Content-Type": "application/json"})
        assert json.loads(urlopen(request).read())["value"] == 6
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class _EchoHandler(BaseRequestHandler):
    def handle(self):
        self.request.sendall(self.request.recv(4096))


def test_port_forwarder_local_and_public_bind_guard():
    with pytest.raises(SecurityPolicyError):
        TcpPortForwarder("0.0.0.0", 0, "127.0.0.1", 12345)

    upstream = ThreadingTCPServer(("127.0.0.1", 0), _EchoHandler)
    upstream_thread = Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    forwarder = TcpPortForwarder("127.0.0.1", 0, "127.0.0.1", upstream.server_address[1])
    try:
        port = forwarder.start()
        with socket.create_connection(("127.0.0.1", port), timeout=3) as client:
            client.sendall(b"forwarded")
            assert client.recv(64) == b"forwarded"
    finally:
        forwarder.close()
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join()
