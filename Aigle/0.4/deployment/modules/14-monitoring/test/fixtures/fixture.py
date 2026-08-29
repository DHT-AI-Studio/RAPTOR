"""Test fixture for the Module 14 alert smoke test.

Two servers in one container:
  :8080/metrics  — serves a static Prometheus exposition file (synthetic GPU / Kafka series)
  :9095/alert    — Alertmanager webhook sink; appends each payload to /out/received.jsonl

Only used by test/alert_smoke_test.sh. Not part of the deployed stack.
"""

import http.server
import json
import os
import socketserver
import threading

METRICS_FILE = os.environ.get("METRICS_FILE", "/fixtures/metrics")
OUT_FILE = os.environ.get("SINK_OUT", "/out/received.jsonl")


class Metrics(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?")[0] != "/metrics":
            self.send_response(404)
            self.end_headers()
            return
        with open(METRICS_FILE, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class Sink(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length).decode("utf-8", "replace")

        with open(OUT_FILE, "a", encoding="utf-8") as f:
            f.write(body.strip().replace("\n", " ") + "\n")

        try:
            seen = [
                "%s=%s" % (a.get("labels", {}).get("alertname"), a.get("status"))
                for a in json.loads(body).get("alerts", [])
            ]
        except Exception:
            seen = ["<unparsed>"]
        print("[sink] POST %s -> %s" % (self.path, ",".join(seen) or "<empty>"), flush=True)

        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"ok")

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args):
        pass


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve(port, handler):
    Server(("0.0.0.0", port), handler).serve_forever()


if __name__ == "__main__":
    threading.Thread(target=serve, args=(9095, Sink), daemon=True).start()
    print("[fixture] metrics on :8080/metrics, webhook sink on :9095", flush=True)
    serve(8080, Metrics)
