"""
OpenTelemetry tracing middleware for distributed tracing support.
Enables end-to-end tracing across Gateway → downstream services → Kafka.
"""
from __future__ import annotations

import logging
import sys

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

try:
    from opentelemetry import trace
    from opentelemetry.baggage import set_baggage
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import Status, StatusCode
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False

_logger = logging.getLogger(__name__)


class TracingConfig:
    """OpenTelemetry tracing configuration."""

    _initialized = False

    @classmethod
    def initialize(cls, service_name: str, jaeger_host: str = "localhost", jaeger_port: int = 6831):
        """
        Initialize OpenTelemetry tracing with Jaeger exporter.

        Args:
            service_name: Name of the service for tracing
            jaeger_host: Jaeger agent hostname
            jaeger_port: Jaeger agent port
        """
        if not TELEMETRY_AVAILABLE:
            _logger.warning("OpenTelemetry packages not installed; tracing disabled")
            return

        if cls._initialized:
            return

        try:
            # Configure Jaeger exporter
            jaeger_exporter = JaegerExporter(
                agent_host_name=jaeger_host,
                agent_port=jaeger_port,
            )

            # Set up trace provider
            trace_provider = TracerProvider()
            trace_provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
            trace.set_tracer_provider(trace_provider)

            # Instrument FastAPI and httpx
            FastAPIInstrumentor.instrument_app(None)  # Will be attached to app instance later
            HTTPXClientInstrumentor().instrument()

            _logger.info(
                "OpenTelemetry tracing initialized",
                extra={
                    "service_name": service_name,
                    "jaeger_host": jaeger_host,
                    "jaeger_port": jaeger_port,
                }
            )
            cls._initialized = True
        except Exception as e:
            _logger.error(f"Failed to initialize tracing: {e}", exc_info=e)


class TracingMiddleware(BaseHTTPMiddleware):
    """Middleware that adds distributed tracing context to requests."""

    async def dispatch(self, request: Request, call_next) -> Response:
        if not TELEMETRY_AVAILABLE:
            return await call_next(request)

        # Get or create correlation ID from X-Request-ID header
        correlation_id = request.headers.get("X-Request-ID")

        # Set up the span itself defensively -- if tracing setup fails, fall back
        # to no tracing. call_next() must only ever be invoked once per request;
        # it used to sit inside this same try, so a downstream handler's own
        # exception (re-raised below to record it on the span) was caught by
        # this outer except too and triggered a *second* call_next(), re-running
        # the request (and any side effects) a second time.
        try:
            if correlation_id:
                set_baggage("correlation_id", correlation_id)
            tracer = trace.get_tracer(__name__)
            span_cm = tracer.start_as_current_span(f"{request.method} {request.url.path}")
            span = span_cm.__enter__()
        except Exception as e:  # noqa: BLE001 -- spans baggage + tracer + span setup (start_as_current_span's own __enter__ is where the span actually gets created); falls back to no tracing for this request, logged
            _logger.debug(f"Tracing middleware error: {e}")
            # Fallback: just call next middleware. call_next() hasn't been
            # invoked yet on this path, so calling it here is still the only call.
            return await call_next(request)

        # span is now open -- from here on, cleanup must go through
        # span_cm.__exit__() (via finally below), not a bare `return`/`raise`,
        # same guarantee the `with` statement gives, but split so a failure
        # setting attributes doesn't also need to avoid double-calling
        # call_next() below.
        try:
            try:
                span.set_attribute("http.method", request.method)
                span.set_attribute("http.url", str(request.url))
                span.set_attribute("http.client_ip", request.client.host if request.client else "unknown")
                if correlation_id:
                    span.set_attribute("correlation_id", correlation_id)
            except Exception as e:  # noqa: BLE001 -- attribute-setting failure shouldn't take the request down with it; span itself is still fine
                _logger.debug(f"Tracing attribute error: {e}")

            try:
                response = await call_next(request)
            except Exception as e:
                try:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                except Exception as span_err:  # noqa: BLE001 -- recording the real failure on the span must not shadow or replace it
                    _logger.debug(f"Tracing span error-recording failed: {span_err}")
                raise

            try:
                span.set_attribute("http.status_code", response.status_code)
                span.set_status(Status(StatusCode.ERROR) if response.status_code >= 400 else Status(StatusCode.OK))
            except Exception as e:  # noqa: BLE001 -- the request already succeeded; a tracing failure here must not turn it into one
                _logger.debug(f"Tracing response-attribute error: {e}")
            return response
        finally:
            span_cm.__exit__(*sys.exc_info())
