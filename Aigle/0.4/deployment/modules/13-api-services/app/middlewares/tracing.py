"""
OpenTelemetry tracing middleware for distributed tracing support.
Enables end-to-end tracing across Gateway → downstream services → Kafka.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

try:
    from opentelemetry import trace, metrics, context
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter as JaegerMetricsExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.baggage import get_baggage, set_baggage
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

        try:
            # Set correlation ID in baggage for downstream propagation
            if correlation_id:
                set_baggage("correlation_id", correlation_id)

            # Get tracer and create span
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(f"{request.method} {request.url.path}") as span:
                # Add attributes to span
                span.set_attribute("http.method", request.method)
                span.set_attribute("http.url", str(request.url))
                span.set_attribute("http.client_ip", request.client.host if request.client else "unknown")
                if correlation_id:
                    span.set_attribute("correlation_id", correlation_id)

                try:
                    response = await call_next(request)
                    span.set_attribute("http.status_code", response.status_code)
                    if response.status_code >= 400:
                        span.set_status(Status(StatusCode.ERROR))
                    else:
                        span.set_status(Status(StatusCode.OK))
                    return response
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        except Exception as e:
            _logger.debug(f"Tracing middleware error: {e}")
            # Fallback: just call next middleware
            return await call_next(request)
