"""Chạy thử Court Vision trên máy tính / điện thoại cùng mạng Wi-Fi.

    python serve.py            # http://localhost:8000
Điện thoại mở http://<IP-máy-tính>:8000 — chế độ "Phân tích video" chạy được;
camera trực tiếp cần https (xem README.md).
"""
import http.server
import os
import socket
import sys
from functools import partial
from pathlib import Path

PORT = int(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PORT", 8000))
ROOT = Path(__file__).parent / "app"


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".webmanifest": "application/manifest+json",
        ".svg": "image/svg+xml",
    }

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()


def lan_ip():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # console Windows mặc định cp1252
    server = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), partial(Handler, directory=str(ROOT)))
    print(f"Máy tính:  http://localhost:{PORT}")
    print(f"Điện thoại (cùng Wi-Fi): http://{lan_ip()}:{PORT}")
    server.serve_forever()
