"""MetricsInterceptor — Prometheus counters/histograms/gauges per RPC call.

The HTTP /metrics endpoint itself is started once in server.py via
prometheus_client.start_http_server(); this module only registers/updates
the metric objects.
"""
from __future__ import annotations

import asyncio
import time

import grpc
from prometheus_client import Counter, Gauge, Histogram

from interceptors._base import call_unary, iter_streaming, wrap_handler

REQUEST_COUNTER = Counter(
    "grpc_server_requests_total",
    "Total gRPC requests handled, by method and status code.",
    ["grpc_method", "grpc_code"],
)
# Prometheus' default buckets stop at 10 s. A download, upload or video stream routinely runs longer,
# and everything above the last bucket lands in +Inf, where histogram_quantile() can only answer
# "10 s": p95/p99 of the streaming RPCs were unreadable. The extra buckets go up to the 1 h that
# the proxy examples allow a stream to live. Unary RPCs are unaffected (they finish in the first ones).
LATENCY_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0,
    30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0,
)
LATENCY_HISTOGRAM = Histogram(
    "grpc_server_handling_seconds",
    "gRPC request handling duration in seconds, by method (the whole call, so a long stream counts in full).",
    ["grpc_method"],
    buckets=LATENCY_BUCKETS,
)
ACTIVE_STREAMS_GAUGE = Gauge(
    "grpc_server_active_streams",
    "Number of in-flight streaming calls (client-streaming, server-streaming, "
    "or both), by method. Unary RPCs are never counted here.",
    ["grpc_method"],
)
STREAM_BYTES_COUNTER = Counter(
    "grpc_server_stream_bytes_total",
    "Bytes transferred on streaming RPCs, by method and direction.",
    ["grpc_method", "direction"],  # direction: sent (server->client) | received (client->server)
)
STREAM_CANCELLATIONS_COUNTER = Counter(
    "grpc_server_stream_cancellations_total",
    "Streaming calls that ended because the client cancelled them, by method.",
    ["grpc_method"],
)

_STREAMING_SHAPES = ("unary_stream", "stream_unary", "stream_stream")


class MetricsInterceptor(grpc.aio.ServerInterceptor):
    async def intercept_service(self, continuation, handler_call_details):
        handler = await continuation(handler_call_details)
        if handler is None:
            return None

        method_name = handler_call_details.method

        def make_wrapper(behavior, shape):
            is_streaming_response = shape in ("unary_stream", "stream_stream")
            is_streaming = shape in _STREAMING_SHAPES

            if is_streaming_response:
                async def wrapped(request_or_iterator, context):
                    start = time.perf_counter()
                    if is_streaming:
                        ACTIVE_STREAMS_GAUGE.labels(grpc_method=method_name).inc()
                    cancelled = False
                    try:
                        async for response in iter_streaming(behavior, request_or_iterator, context):
                            STREAM_BYTES_COUNTER.labels(grpc_method=method_name, direction="sent").inc(
                                response.ByteSize()
                            )
                            yield response
                    except (asyncio.CancelledError, GeneratorExit):
                        # A client cancel reaches a server-streaming handler in one of two ways.
                        # If the handler is awaiting inside the generator, the task is cancelled
                        # (CancelledError). If grpc.aio is blocked writing the previous response
                        # (HTTP/2 flow control, the common case for a big transfer), the generator is
                        # simply closed at its `yield` (GeneratorExit). Measured against a real client:
                        # 9 of 10 cancels were GeneratorExit, which used to be recorded as OK.
                        cancelled = True
                        raise
                    finally:
                        if is_streaming:
                            ACTIVE_STREAMS_GAUGE.labels(grpc_method=method_name).dec()
                        _record(method_name, context, start, is_streaming, cancelled)
            else:
                async def wrapped(request_or_iterator, context):
                    start = time.perf_counter()
                    if is_streaming:
                        ACTIVE_STREAMS_GAUGE.labels(grpc_method=method_name).inc()
                        request_or_iterator = _counted_request_iterator(request_or_iterator, method_name)
                    cancelled = False
                    try:
                        return await call_unary(behavior, request_or_iterator, context)
                    except asyncio.CancelledError:
                        cancelled = True
                        raise
                    finally:
                        if is_streaming:
                            ACTIVE_STREAMS_GAUGE.labels(grpc_method=method_name).dec()
                        _record(method_name, context, start, is_streaming, cancelled)

            return wrapped

        return wrap_handler(handler, make_wrapper)


async def _counted_request_iterator(request_iterator, method_name: str):
    """Wraps a client-streaming request iterator to count received bytes
    per message, without changing what the wrapped handler observes."""
    async for request in request_iterator:
        STREAM_BYTES_COUNTER.labels(grpc_method=method_name, direction="received").inc(request.ByteSize())
        yield request


def _record(
    method_name: str, context: grpc.aio.ServicerContext, start: float, is_streaming: bool, cancelled: bool
) -> None:
    # context.cancelled()/context.code() are NOT reliable at this point for a
    # call that just unwound via asyncio.CancelledError -- grpc.aio's own
    # bookkeeping for the context hasn't caught up yet when the handler's
    # `finally` runs (confirmed empirically: both return their "nothing
    # happened yet" defaults -- False / None -- even though the RPC was, in
    # fact, cancelled). The `except (CancelledError, GeneratorExit)` in the wrapper
    # above is the only trustworthy signal; when it fired, we already know
    # the real status without asking the context.
    duration = time.perf_counter() - start
    code = grpc.StatusCode.CANCELLED if cancelled else (context.code() or grpc.StatusCode.OK)
    REQUEST_COUNTER.labels(grpc_method=method_name, grpc_code=code.name).inc()
    LATENCY_HISTOGRAM.labels(grpc_method=method_name).observe(duration)
    if is_streaming and cancelled:
        STREAM_CANCELLATIONS_COUNTER.labels(grpc_method=method_name).inc()
