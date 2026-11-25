"""
Unit tests for TracingMiddleware (app/middlewares/tracing.py).

Covers the bug fixed in "fix(04,07,13): 5 real bugs found while triaging
BLE001": call_next() must be invoked exactly once per request, even when a
downstream handler raises -- the old code's single try/except wrapped both
the span setup *and* the call_next() call, so a handler's own exception
(re-raised to record it on the span) was caught by the same outer except and
triggered a second call_next(), re-running the request's side effects.

Also covers the narrower regression introduced while fixing that: moving
`with span_cm as span:` out of the try/except meant a failure entering the
span (or setting its attributes) was no longer caught, unlike every other
tracing failure here.

Run from the 13-api-services/ directory:
    python -m pytest tests/unit/test_tracing_middleware.py -v
"""
import os
import sys

os.environ.setdefault("GATEWAY_AUTH_SERVICE_URL", "http://localhost:8800")
os.environ.setdefault("GATEWAY_KEYCLOAK_URL", "http://localhost:8080")
os.environ.setdefault("GATEWAY_KAFKA_BOOTSTRAP_SERVERS", '["localhost:9092"]')
os.environ.setdefault("GATEWAY_KAFKA_TOPICS", '{"document": "document-processing-requests"}')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
from app.middlewares import tracing
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

pytestmark = pytest.mark.skipif(
    not tracing.TELEMETRY_AVAILABLE, reason="opentelemetry not installed in this env"
)


def _app(handler_calls: list) -> FastAPI:
    app = FastAPI()
    app.add_middleware(tracing.TracingMiddleware)

    @app.get("/ok")
    def ok():
        handler_calls.append(1)
        return {"ok": True}

    @app.get("/boom")
    def boom():
        handler_calls.append(1)
        raise HTTPException(status_code=500, detail="handler blew up")

    return app


def test_happy_path_calls_handler_once_and_sets_span_status():
    calls: list = []
    client = TestClient(_app(calls))
    resp = client.get("/ok")
    assert resp.status_code == 200
    assert len(calls) == 1


def test_handler_exception_propagates_and_handler_called_exactly_once():
    """The bug this commit fixed: a handler exception used to be caught by
    the same except that guards span setup, triggering a second call_next()
    (and a second handler invocation)."""
    calls: list = []
    client = TestClient(_app(calls), raise_server_exceptions=False)
    resp = client.get("/boom")
    assert resp.status_code == 500
    assert len(calls) == 1


def test_tracer_setup_failure_falls_back_to_single_call_next(monkeypatch):
    """If tracer.get_tracer() (or start_as_current_span's own __enter__)
    throws, the request must still be served exactly once, without tracing."""
    calls: list = []

    def _broken_get_tracer(*a, **kw):
        raise RuntimeError("tracer backend unavailable")

    monkeypatch.setattr(tracing.trace, "get_tracer", _broken_get_tracer)
    client = TestClient(_app(calls))
    resp = client.get("/ok")
    assert resp.status_code == 200
    assert len(calls) == 1


def test_span_attribute_failure_does_not_block_the_request(monkeypatch):
    """A failure setting span attributes (after the span is already open)
    must not take the request down with it -- this is the regression: moving
    `with span_cm as span:` out of the try/except left this path unguarded."""
    calls: list = []
    real_get_tracer = tracing.trace.get_tracer

    class _BrokenAttrSpan:
        def __init__(self, real_span):
            self._real = real_span

        def set_attribute(self, *a, **kw):
            raise RuntimeError("attribute sink unavailable")

        def __getattr__(self, name):
            return getattr(self._real, name)

    class _WrappingTracer:
        def __init__(self, real_tracer):
            self._real = real_tracer

        def start_as_current_span(self, *a, **kw):
            import contextlib

            @contextlib.contextmanager
            def _cm():
                with self._real.start_as_current_span(*a, **kw) as real_span:
                    yield _BrokenAttrSpan(real_span)

            return _cm()

    monkeypatch.setattr(
        tracing.trace, "get_tracer", lambda name: _WrappingTracer(real_get_tracer(name))
    )
    client = TestClient(_app(calls))
    resp = client.get("/ok")
    assert resp.status_code == 200
    assert len(calls) == 1


def test_span_entry_failure_falls_back_to_single_call_next(monkeypatch):
    """A failure inside start_as_current_span's own __enter__ (the actual
    span-creation call, lazily triggered by `with`/`__enter__`, not by
    merely calling start_as_current_span()) must be caught the same as a
    tracer-setup failure, and call_next() still runs exactly once."""
    calls: list = []
    real_get_tracer = tracing.trace.get_tracer

    class _BrokenEnterCM:
        def __enter__(self):
            raise RuntimeError("span creation failed")

        def __exit__(self, *exc):
            return False

    class _WrappingTracer:
        def __init__(self, real_tracer):
            self._real = real_tracer

        def start_as_current_span(self, *a, **kw):
            return _BrokenEnterCM()

    monkeypatch.setattr(
        tracing.trace, "get_tracer", lambda name: _WrappingTracer(real_get_tracer(name))
    )
    client = TestClient(_app(calls))
    resp = client.get("/ok")
    assert resp.status_code == 200
    assert len(calls) == 1
