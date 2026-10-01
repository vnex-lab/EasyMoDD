"""Local-only browser workbench with page-scoped keyboard shortcuts."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import ipaddress
import json
import os
import secrets
from urllib.parse import parse_qs, urlparse

from .errors import SecurityPolicyError
from .workbench import Workbench


_PAGE = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EasyModel Workbench</title>
<style>
:root{color-scheme:dark;--bg:#10151b;--panel:#19212a;--line:#344453;--ink:#e8f0f5;--muted:#9db0bf;--accent:#75d6bd;--warn:#f1c27b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}
header{padding:22px max(24px,calc((100vw - 1080px)/2));border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:center}
header strong{font-size:20px}main{max-width:1080px;margin:28px auto;padding:0 24px;display:grid;grid-template-columns:minmax(0,1.1fr) minmax(280px,.9fr);gap:18px}
section{background:var(--panel);border:1px solid var(--line);padding:20px;border-radius:8px}h2{font-size:17px;margin:0 0 14px}label{display:block;color:var(--muted);margin:12px 0 5px}
input,select,button,textarea{font:inherit;color:var(--ink);background:#10171e;border:1px solid var(--line);border-radius:5px;padding:10px;width:100%}button{cursor:pointer;background:#213b3b;border-color:#3b736a;color:#d8fff3;font-weight:650;margin-top:10px}button:hover{background:#2a504c}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#0d1217;padding:14px;border-radius:5px;max-height:55vh;overflow:auto;color:#c9d8e2}.row{display:flex;gap:8px;align-items:end}.row>*{flex:1}.hint{color:var(--muted);font-size:13px}.status{color:var(--accent)}@media(max-width:760px){main{grid-template-columns:1fr;margin-top:14px;padding:0 12px}header{padding:16px}}
</style>
<header><strong>EasyModel Workbench</strong><span id="status" class="status">Local session</span></header>
<main>
<section><h2>Artifact and corpus tools</h2>
<label for="token">Session token</label><input id="token" type="password" autocomplete="off" placeholder="Paste token printed by the server">
<label for="path">Input path</label><input id="path" autocomplete="off" placeholder="Path to an artifact or corpus">
<label for="operation">Operation</label><div class="row"><select id="operation"><option value="passport">Artifact Passport</option><option value="audit">Audit text/JSONL corpus</option><option value="encrypt">Encrypt file</option><option value="decrypt">Decrypt file</option></select>
<select id="format"><option value="auto">Auto format</option><option value="text">Text</option><option value="jsonl">JSONL</option></select></div>
<label for="output">Output path (optional)</label><input id="output" autocomplete="off" placeholder="Deduplicated or encrypted/decrypted output">
<label for="passphrase">Encryption passphrase (never saved)</label><input id="passphrase" type="password" autocomplete="new-password">
<button id="run">Run selected operation</button><p class="hint">Ctrl+Enter runs. Escape clears output. Shortcuts only work while this page is open.</p></section>
<section><h2>Result</h2><pre id="result">Waiting for an operation.</pre><h2>Custom pages</h2><div id="pages"></div><iframe id="customPage" sandbox="allow-scripts" title="Custom workbench page" style="width:100%;min-height:240px;border:1px solid var(--line);border-radius:5px"></iframe><h2>Recent shared events</h2><pre id="events">Loading...</pre></section>
</main>
<script>
const $=id=>document.getElementById(id), result=$('result');
if(sessionStorage.getItem('easymodelToken')) $('token').value=sessionStorage.getItem('easymodelToken');
async function api(path, body){const token=$('token').value.trim();sessionStorage.setItem('easymodelToken',token);const response=await fetch(path,{method:body?'POST':'GET',headers:{'Authorization':'Bearer '+token,'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});const data=await response.json();if(!response.ok)throw Error(data.error||response.statusText);return data;}
async function run(){ $('status').textContent='Working...';try{const operation=$('operation').value;const payload={path:$('path').value,format:$('format').value,output:$('output').value,passphrase:$('passphrase').value};const data=await api('/api/'+operation,payload);result.textContent=JSON.stringify(data,null,2);$('passphrase').value='';$('status').textContent='Complete';await refreshEvents();}catch(error){result.textContent=error.message;$('status').textContent='Failed';}}
async function refreshEvents(){try{$('events').textContent=JSON.stringify(await api('/api/events'),null,2);}catch(error){$('events').textContent=error.message;}}
async function refreshPages(){try{const pages=await api('/api/pages');$('pages').replaceChildren(...pages.map(page=>{const button=document.createElement('button');button.textContent=page.title;button.addEventListener('click',async()=>{const content=await api('/api/page?path='+encodeURIComponent(page.path));$('customPage').srcdoc=content.html;});return button;}));}catch(error){$('pages').textContent=error.message;}}
window.addEventListener('message',async event=>{const frame=$('customPage');if(event.source!==frame.contentWindow||!event.data||event.data.type!=='easymodel-api'||typeof event.data.path!=='string'||!event.data.path.startsWith('/api/custom/'))return;try{const result=await api(event.data.path,event.data.payload||{});frame.contentWindow.postMessage({type:'easymodel-api-result',id:event.data.id,result},'*');}catch(error){frame.contentWindow.postMessage({type:'easymodel-api-result',id:event.data.id,error:error.message},'*');}});
$('run').addEventListener('click',run);document.addEventListener('keydown',event=>{if(event.ctrlKey&&event.key==='Enter'){event.preventDefault();run();}else if(event.key==='Escape'){result.textContent='';$('passphrase').value='';}else if(event.key==='/'&&!['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName)){event.preventDefault();$('path').focus();}});refreshEvents();refreshPages();setInterval(refreshEvents,5000);
</script></html>'''


def create_server(host: str = "127.0.0.1", port: int = 8765, *, token: str | None = None,
                  allow_remote: bool = False, workbench: Workbench | None = None) -> tuple[ThreadingHTTPServer, str]:
    """Create a server for embedding in an application or starting in a thread."""
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host.lower() == "localhost"
    token = token or os.environ.get("EASYMODEL_WEB_TOKEN")
    if not loopback and not allow_remote:
        raise SecurityPolicyError("Web workbench binds to localhost only; remote binding requires allow_remote=True")
    if not token:
        if not loopback:
            raise SecurityPolicyError("Remote web workbench requires EASYMODEL_WEB_TOKEN")
        token = secrets.token_urlsafe(32)
    workbench = workbench or Workbench()

    class Handler(BaseHTTPRequestHandler):
        server_version = "EasyModelWorkbench/1"

        def log_message(self, format, *args):
            return

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send(200, _PAGE.encode("utf-8"), "text/html; charset=utf-8")
                return
            if not self._authorized():
                self._json(401, {"error": "Unauthorized. Enter the local session token."})
                return
            if parsed.path == "/api/status":
                self._json(200, {"status": "ready", "bind": host, "port": self.server.server_port})
            elif parsed.path == "/api/events":
                self._json(200, workbench.event_bus.recent(100))
            elif parsed.path == "/api/pages":
                self._json(200, [{"path": page.path, "title": page.title} for page in workbench.pages.values()])
            elif parsed.path == "/api/page":
                page = workbench.pages.get(parse_qs(parsed.query).get("path", [""])[0])
                if page is None:
                    self._json(404, {"error": "Custom page not found"})
                else:
                    self._json(200, {"title": page.title, "html": page.html})
            else:
                self._json(404, {"error": "Route not found"})

        def do_POST(self):
            if not self._authorized():
                self._json(401, {"error": "Unauthorized. Enter the local session token."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 1024 * 1024:
                    raise ValueError("Request body exceeds 1 MiB")
                payload = json.loads(self.rfile.read(length) or b"{}")
                path = payload.get("path", "")
                route = urlparse(self.path).path
                if route == "/api/passport":
                    output = workbench.passport(path).to_dict()
                elif route == "/api/audit":
                    output = workbench.audit_corpus(
                        path, format=payload.get("format", "auto"),
                        deduplicated_output=payload.get("output") or None,
                    ).to_dict()
                elif route in {"/api/encrypt", "/api/decrypt"}:
                    phrase = payload.get("passphrase", "")
                    destination = payload.get("output", "")
                    if not destination:
                        raise ValueError("Specify an output path")
                    if route.endswith("encrypt"):
                        output = {"path": str(workbench.encrypt(path, destination, phrase))}
                    else:
                        output = {"path": str(workbench.decrypt(path, destination, phrase))}
                elif route in workbench.api_routes:
                    output = workbench.run_api_route(route, payload)
                else:
                    self._json(404, {"error": "Route not found"})
                    return
                self._json(200, output)
            except Exception as error:
                self._json(400, {"error": str(error)})

        def _authorized(self) -> bool:
            supplied = self.headers.get("Authorization", "")
            return supplied.startswith("Bearer ") and hmac.compare_digest(supplied[7:], token)

        def _json(self, code: int, value) -> None:
            body = json.dumps(value, default=str).encode("utf-8")
            if len(body) > 5 * 1024 * 1024:
                code = 413
                body = b'{"error":"Response exceeds the 5 MiB workbench response limit"}'
            self._send(code, body, "application/json; charset=utf-8")

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'")
            self.end_headers()
            self.wfile.write(body)

    return ThreadingHTTPServer((host, port), Handler), token


def serve(host: str = "127.0.0.1", port: int = 8765, *, token: str | None = None,
          allow_remote: bool = False, workbench: Workbench | None = None) -> None:
    """Run the web workbench; remote binding requires an explicit token/opt-in."""
    server, token = create_server(host, port, token=token, allow_remote=allow_remote, workbench=workbench)
    print(f"EasyModel Workbench: http://{host}:{server.server_port}/")
    print(f"Session token: {token}")
    print("Keyboard shortcuts are scoped to this browser page; no system keystrokes are captured.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping EasyModel Workbench")
    finally:
        server.server_close()
