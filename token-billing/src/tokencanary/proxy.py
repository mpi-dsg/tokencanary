from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx

from .auditor import Auditor
from .transport import AuditTransport

_SKIP_REQUEST = {"host", "content-length", "accept-encoding", "connection", "transfer-encoding"}
_SKIP_RESPONSE = {"content-length", "content-encoding", "transfer-encoding", "connection"}


def make_server(upstream: str, auditor: Auditor, host: str = "127.0.0.1", port: int = 8787,
                inner: httpx.BaseTransport | None = None, inject_logprobs: bool = True) -> ThreadingHTTPServer:
    """Reverse proxy to `upstream` that routes every request through an AuditTransport."""
    upstream = upstream.rstrip("/")
    client = httpx.Client(transport=AuditTransport(auditor, inner, inject_logprobs), timeout=httpx.Timeout(600, connect=30))

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def forward(self):
            body = self.rfile.read(int(self.headers.get("content-length") or 0))
            headers = [(k, v) for k, v in self.headers.items() if k.lower() not in _SKIP_REQUEST]
            try:
                resp = client.send(client.build_request(self.command, upstream + self.path, headers=headers, content=body), stream=True)
            except httpx.HTTPError as exc:
                self.send_error(502, f"upstream error: {type(exc).__name__}")
                return
            try:
                self.send_response(resp.status_code)
                for k, v in resp.headers.multi_items():
                    if k.lower() not in _SKIP_RESPONSE:
                        self.send_header(k, v)
                if resp.headers.get("content-type", "").startswith("text/event-stream"):
                    self.send_header("transfer-encoding", "chunked")
                    self.end_headers()
                    for chunk in resp.iter_bytes():
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                        self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")
                else:
                    data = resp.read()
                    self.send_header("content-length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
            finally:
                resp.close()

        do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = forward

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server
