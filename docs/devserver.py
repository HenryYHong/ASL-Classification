"""Static server for the demo that also accepts POSTed diagnostics.

The browser equivalent of live_demo.py --log. Every hard bug in this project was found by
recording what the running system actually did and reading it back, never by reasoning about the
code, so the browser port gets the same instrument. Localhost only; the deployed page posts
nothing.

    ../.venv/bin/python devserver.py [port]    default 8799
"""
import json
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "browser_log.jsonl")


class Handler(SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/log":
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        with open(LOG, "a") as fh:
            for line in body.decode().splitlines():
                if line.strip():
                    fh.write(line + "\n")
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.end_headers()

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"serving {os.path.dirname(os.path.abspath(__file__))}")
    print(f"diagnostics -> {LOG}")
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8799
    print(f"http://localhost:{port}/")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
