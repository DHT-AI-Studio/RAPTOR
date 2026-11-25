"""TMP-7 AC: a forced `parse` timeout on "what happened after 2011" is UNRESOLVED -- the query never runs
without its time condition, and the interim regex's window ([2012-01-01, unbounded)) is not presented in
its place. A real HTTP server stands in for a slow Module 25; Module 18's client calls it over the network."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from app.core.config import get_settings
from app.routers.query import ClassifyRequest, classify_endpoint
from app.services.temporal_parse import parse_client

PARSE_DELAY = 2.0          # seconds Module 25 takes to answer `parse`
TIMEOUT_MS = 200


class _SlowModule25(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):                   # capabilities: v2 on, parse served -> Module 18 relies on it
        self._send({"temporal_v2": True, "operations": {"parse": "/personal/temporal/parse"}})

    def do_POST(self):                  # parse: answers, but too late
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        time.sleep(PARSE_DELAY)
        self._send({"status": "RESOLVED", "match": "inside",
                    "window": {"start": "2012-01-01T00:00:00.000+08:00", "end": None}})


@pytest.fixture
def slow_module25(monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SlowModule25)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    settings = get_settings()
    monkeypatch.setattr(settings, "personal_db_url", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setattr(settings, "parse_timeout_ms", TIMEOUT_MS)
    monkeypatch.setattr(parse_client, "_checked_at", 0.0)            # check the capabilities afresh
    yield
    server.shutdown()
    server.server_close()
    parse_client._checked_at = 0.0


def test_forced_parse_timeout_is_unresolved_and_never_the_unfiltered_question(slow_module25):
    classifier = SimpleNamespace(
        people_intent=lambda q, temporal: None,
        classify=lambda q, k=3: SimpleNamespace(intent_type=["TKG"], confidence=1.0,
                                                routing_strategy="rule_based", matched_rules=[]))
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(intent_classifier=classifier)))
    started = time.monotonic()
    plan = classify_endpoint(ClassifyRequest(query="what happened after 2011"), request)
    elapsed = time.monotonic() - started

    assert (plan.temporal_source, plan.temporal_status, plan.temporal_reason) == ("parse", "UNRESOLVED", "parse_failure")
    assert plan.answer is None and plan.temporal_constraint_applied is False
    # no window at all: neither unfiltered (no bound) presented as an answer, nor the regex's reading
    assert (plan.time_from, plan.time_to, plan.match, plan.reference, plan.keywords) == (None, None, None, None, [])
    assert elapsed < PARSE_DELAY / 2                                   # gave up at the timeout, not at the reply
