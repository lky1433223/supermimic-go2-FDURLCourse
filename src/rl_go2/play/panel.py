"""Loopback-only browser panel for portable press/release keyboard input."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Thread
from collections.abc import Iterator

from .controls import Controls


@contextmanager
def serve_panel(controls: Controls) -> Iterator[str]:
    """Serve controls on an ephemeral local port and release all server resources."""
    page = Path(__file__).with_name('panel.html').read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == '/status':
                self.respond(json.dumps(controls.status()).encode(), 'application/json')
            elif self.path == '/':
                self.respond(page, 'text/html; charset=utf-8')
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            if self.path != '/input':
                self.send_error(404)
                return
            # JSON and same-origin fetch prevent cross-site form submissions.
            if self.headers.get('Content-Type') != 'application/json':
                self.send_error(415)
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 4096:
                    raise ValueError('Invalid input length')
                data = json.loads(self.rfile.read(length))
                keys, action = data['keys'], data.get('action', '')
                if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
                    raise ValueError('Invalid keys')
                if not isinstance(action, str):
                    raise ValueError('Invalid action')
            except (ValueError, KeyError, TypeError):
                self.send_error(400)
                return
            controls.receive(keys, action)
            self.respond(b'{}', 'application/json')

        def respond(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            """Keep polling requests out of the simulator console."""

    with ThreadingHTTPServer(('127.0.0.1', 0), Handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f'http://127.0.0.1:{server.server_port}'
        finally:
            server.shutdown()
            thread.join()
