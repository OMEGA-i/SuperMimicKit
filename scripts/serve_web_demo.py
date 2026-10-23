#!/usr/bin/env python3
"""Serve the exported browser demo over TLS without exposing certificate files."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import ssl


class DemoHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def list_directory(self, path):
        self.send_error(403, "Directory listing disabled")
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--cert", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    root = args.directory.resolve(strict=True)
    if not (root / "index.html").is_file():
        parser.error("directory must contain index.html")
    for path in [args.cert, args.key]:
        if path.resolve(strict=True).is_relative_to(root):
            parser.error("certificates and keys must be outside the served directory")
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(args.cert, args.key)
    handler = partial(DemoHandler, directory=str(root))
    with ThreadingHTTPServer((args.bind, args.port), handler) as server:
        server.socket = context.wrap_socket(server.socket, server_side=True)
        print(f"HTTPS listening on {args.bind}:{args.port}; root={root}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
