"""Propagates or generates x-request-id for every incoming call."""
from __future__ import annotations

import uuid

import grpc

from interceptors._base import call_unary, iter_streaming, wrap_handler
from interceptors.context import request_id_var


def _extract_request_id(handler_call_details) -> str:
    for key, value in handler_call_details.invocation_metadata or []:
        if key.lower() == "x-request-id" and value:
            return value
    return str(uuid.uuid4())


class RequestIdInterceptor(grpc.aio.ServerInterceptor):
    """Sets a request id (from metadata or freshly generated) for the call's duration."""

    async def intercept_service(self, continuation, handler_call_details):
        handler = await continuation(handler_call_details)
        if handler is None:
            return None

        request_id = _extract_request_id(handler_call_details)

        def make_wrapper(behavior, shape):
            is_streaming_response = shape in ("unary_stream", "stream_stream")

            if is_streaming_response:
                async def wrapped(request_or_iterator, context):
                    request_id_var.set(request_id)
                    context.set_trailing_metadata((("x-request-id", request_id),))
                    async for response in iter_streaming(behavior, request_or_iterator, context):
                        yield response
            else:
                async def wrapped(request_or_iterator, context):
                    request_id_var.set(request_id)
                    context.set_trailing_metadata((("x-request-id", request_id),))
                    return await call_unary(behavior, request_or_iterator, context)

            return wrapped

        return wrap_handler(handler, make_wrapper)
