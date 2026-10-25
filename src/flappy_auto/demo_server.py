"""Serve the bundled local demo for offline tests and demos."""
from __future__ import annotations

from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread


@contextmanager
def demo_server():
    demo_dir = Path(__file__).resolve().parents[2] / 'demo'
    if not demo_dir.is_dir():
        raise FileNotFoundError(f'Local demo missing at {demo_dir}; use the source checkout')
    handler = partial(SimpleHTTPRequestHandler, directory=str(demo_dir))
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/'
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
