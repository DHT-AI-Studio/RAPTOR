"""Shared helper for wrapping grpc.aio RpcMethodHandler behaviors.

grpc.RpcMethodHandler is a NamedTuple with exactly one of unary_unary /
unary_stream / stream_unary / stream_stream set, depending on the RPC shape.
Interceptors that want to run code around the actual handler (regardless of
shape) would otherwise need to duplicate the same four branches; this module
does it once so every interceptor stays a few lines long.
"""
from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable

import grpc

_BEHAVIOR_FIELDS = ("unary_unary", "unary_stream", "stream_unary", "stream_stream")

Behavior = Callable[..., Awaitable[object]]
BehaviorWrapper = Callable[[Behavior, str], Behavior]


def wrap_handler(handler: grpc.RpcMethodHandler, wrap_behavior: BehaviorWrapper) -> grpc.RpcMethodHandler:
    """Return a copy of `handler` with its single active behavior wrapped.

    `wrap_behavior(original_behavior, shape)` must return a replacement
    behavior with the same calling convention as the original.
    """
    for field in _BEHAVIOR_FIELDS:
        original = getattr(handler, field)
        if original is not None:
            return handler._replace(**{field: wrap_behavior(original, field)})
    return handler


async def call_unary(behavior, request_or_iterator, context):
    """Invokes a unary-response behavior and awaits it only if it's actually awaitable.

    Handlers we write ourselves are always coroutine functions, but built-in
    services (e.g. grpc_health.v1.health.HealthServicer.Check) can be plain
    sync methods even when registered on a grpc.aio.Server — awaiting their
    return value directly would raise TypeError.
    """
    result = behavior(request_or_iterator, context)
    if inspect.isawaitable(result):
        result = await result
    return result


async def iter_streaming(behavior, request_or_iterator, context):
    """Invokes a streaming-response behavior and yields from it uniformly.

    Normalizes async-generator, plain-generator and coroutine-that-writes-via-
    `context.write` behaviors (see call_unary's docstring for why the shapes can
    differ) to a single async-iterable interface.
    """
    result = behavior(request_or_iterator, context)
    if hasattr(result, "__anext__"):
        async for item in result:
            yield item
    elif inspect.isawaitable(result):
        # A streaming handler written as a coroutine that writes its responses itself with
        # `await context.write(...)` -- grpc.aio's grpc_health.v1 Watch is one. It has nothing
        # to yield; awaiting it runs the stream (and a cancel propagates out of the await).
        await result
    else:
        for item in result:
            yield item
