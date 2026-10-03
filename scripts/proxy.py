#!/usr/bin/env python3
"""Transparent logging proxy: 127.0.0.1:1235 -> 127.0.0.1:1234.

Captures the request bodies opencode actually sends (does it pass `stop`? what
max_tokens? what stop sequences?) and streams the response back untouched.
Bodies land in /home/davidcanar/proxy_capture/<n>.json, one file per request.
"""
from __future__ import annotations

import http.server
import json
import pathlib
import socketserver
import threading
import urllib.request

UP = "http://127.0.0.1:1234"
OUT = pathlib.Path.home() / "proxy_capture"
OUT.mkdir(exist_ok=True)
_lock = threading.Lock()
_n = [0]


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # quiet
        pass

    def _proxy(self, method: str) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        if body:
            with _lock:
                _n[0] += 1
                idx = _n[0]
            try:
                (OUT / f"{idx:03d}.json").write_bytes(body)
                d = json.loads(body)
                # a compact index so the interesting fields are greppable
                with (OUT / "index.txt").open("a") as fh:
                    fh.write(
                        f"{idx:03d} stop={d.get('stop')!r} max_tokens={d.get('max_tokens')} "
                        f"stream={d.get('stream')} tools={len(d.get('tools') or [])} "
                        f"msgs={len(d.get('messages') or [])} "
                        f"ctk={d.get('chat_template_kwargs')}\n"
                    )
            except Exception as e:  # noqa: BLE001
                (OUT / "index.txt").open("a").write(f"{_n[0]:03d} unparsed: {e}\n")

        req = urllib.request.Request(
            UP + self.path, data=body if body else None, method=method,
            headers={k: v for k, v in self.headers.items() if k.lower() != "host"},
        )
        try:
            with urllib.request.urlopen(req, timeout=3600) as up:
                self.send_response(up.status)
                for k, v in up.headers.items():
                    if k.lower() in ("transfer-encoding", "content-length", "connection"):
                        continue
                    self.send_header(k, v)
                self.send_header("Connection", "close")
                self.end_headers()
                while True:
                    chunk = up.read(1)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except urllib.error.HTTPError as e:
            payload = e.read()
            self.send_response(e.code)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)
        except Exception:  # noqa: BLE001
            pass

    def do_POST(self):
        self._proxy("POST")

    def do_GET(self):
        self._proxy("GET")


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    print("proxy listening on 1235 -> 1234; captures in", OUT, flush=True)
    Server(("127.0.0.1", 1235), Handler).serve_forever()
