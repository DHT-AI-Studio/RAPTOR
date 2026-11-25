"""LoggingInterceptor — structured JSON log line per RPC call."""
from __future__ import annotations

import logging
import time

import grpc

from interceptors._base import call_unary, iter_streaming, wrap_handler
from interceptors.context import request_id_var

_logger = logging.getLogger("grpc_gateway.request")


class LoggingInterceptor(grpc.aio.ServerInterceptor):
    async def intercept_service(self, continuation, handler_call_details):
        handler = await continuation(handler_call_details)
        if handler is None:
            return None

        method_name = handler_call_details.method

        def make_wrapper(behavior, shape):
            is_streaming_response = shape in ("unary_stream", "stream_stream")

            if is_streaming_response:
                async def wrapped(request_or_iterator, context):
                    start = time.perf_counter()
                    try:
                        async for response in iter_streaming(behavior, request_or_iterator, context):
                            yield response
                    finally:
                        _log_call(method_name, start)
            else:
                async def wrapped(request_or_iterator, context):
                    start = time.perf_counter()
                    try:
                        return await call_unary(behavior, request_or_iterator, context)
                    finally:
                        _log_call(method_name, start)

            return wrapped

        return wrap_handler(handler, make_wrapper)


def _log_call(method_name: str, start: float) -> None:
    duration_ms = (time.perf_counter() - start) * 1000
    _logger.info(
        "grpc_call",
        extra={
            "grpc_method": method_name,
            "request_id": request_id_var.get(),
            "duration_ms": round(duration_ms, 4),
        },
    )
