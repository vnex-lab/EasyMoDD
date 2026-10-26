"""Custom API engine (stdlib http.server based)."""
import json, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

class ApiEngine:
    def __init__(self, host="127.0.0.1", port=8080):
        self.host=host; self.port=port; self.routes={}; self._server=None
    def route(self, method, path, handler):
        self.routes.setdefault(path, {})[method.upper()]=handler
    def get(self,path,handler): self.route("GET",path,handler)
    def post(self,path,handler): self.route("POST",path,handler)
    def _make_handler(self):
        eng=self
        class H(BaseHTTPRequestHandler):
            def do_GET(self): self._dispatch("GET")
            def do_POST(self): self._dispatch("POST")
            def _dispatch(self,method):
                p=urlparse(self.path).path
                if p in eng.routes and method in eng.routes[p]:
                    try:
                        length=int(self.headers.get("Content-Length",0))
                        body=self.rfile.read(length) if length else b""
                        data=json.loads(body) if body else {}
                        res=eng.routes[p][method](data, dict(self.headers))
                        self.send_response(200); self.send_header("Content-Type","application/json"); self.end_headers()
                        self.wfile.write(json.dumps(res).encode())
                    except Exception as e:
                        self.send_response(500); self.end_headers(); self.wfile.write(str(e).encode())
                else:
                    self.send_response(404); self.end_headers()
            def log_message(self,*a): pass
        return H
    def start(self, blocking=False):
        self._server=HTTPServer((self.host,self.port),self._make_handler())
        if blocking: self._server.serve_forever()
        else: threading.Thread(target=self._server.serve_forever,daemon=True).start()
    def stop(self):
        if self._server: self._server.shutdown()
