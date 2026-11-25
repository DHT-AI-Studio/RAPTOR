"""StreamLimitInterceptor — per-user concurrent stream cap, shared across
every streaming RPC (client-streaming, server-streaming, or both).

One Redis counter per user_id (`grpc:streams:{user_id}`), not one per RPC
method — a user's total concurrent streams across UploadAsset +
DownloadAsset + StreamSearchByPerson + WatchJob etc. are all capped
together. Incremented when a stream is accepted, decremented in a `finally`
so a normal finish, an error, *and* a client cancellation all release the
slot -- there is no separate cancellation-specific code path to keep in
sync with the happy path.

Runs between MetricsInterceptor and ExceptionMappingInterceptor: it needs
user_var (set by IdentityInterceptor, so it must be inside that).

Rejection calls `context.abort()` directly rather than raising an AppError
(ValidationError, ResourceExhaustedError, etc. from interceptors/errors.py).
That hierarchy only works for exceptions raised *from inside* a call into
the next inner behavior, because ExceptionMappingInterceptor's try/except
only wraps its own call into the raw handler -- an exception raised by an
*outer* interceptor's own code, before it ever calls into its inner
behavior, unwinds straight past ExceptionMapping without ever entering its
try block, and reaches grpc.aio as an unmapped UNKNOWN. `context.abort()`
sidesteps this entirely: it works from any layer, which is exactly why the
still-UNIMPLEMENTED RPC stubs in server.py call it directly instead of
raising an AppError too.

Like Logging and Metrics, this interceptor does not itself check
auth_error_var, so a stream slot is acquired-and-immediately-released even
for a call that AuthenticationInterceptor already flagged for rejection --
the same "observability/bookkeeping runs regardless of the eventual abort"
tradeoff this chain already makes for Logging/Metrics (see
AuthenticationInterceptor's docstring). Only unary_unary RPCs (Ping, GetJob,
SubmitJob, ...) are exempt -- they never hold a long-lived stream to limit.
"""
from __future__ import annotations

import logging

import grpc

from interceptors._base import call_unary, iter_streaming, wrap_handler
from interceptors.auth import _is_exempt as _is_infrastructure_call
from interceptors.context import user_var

_logger = logging.getLogger("grpc_gateway.stream_limit")

_STREAMING_SHAPES = ("unary_stream", "stream_unary", "stream_stream")


def _stream_key(user_id: str) -> str:
    return f"grpc:streams:{user_id}"


class StreamLimitInterceptor(grpc.aio.ServerInterceptor):
    def __init__(self, redis_client, max_streams_per_user: int):
        self._redis_client = redis_client
        self._max_streams = max_streams_per_user

    async def intercept_service(self, continuation, handler_call_details):
        handler = await continuation(handler_call_details)
        if handler is None:
            return None
        # Health checks and reflection are infrastructure: exempt from authentication, and so
        # also from the per-user cap -- a probe has no user, and a Watch is long-lived, so
        # counting it would eat into whoever the anonymous fallback identity is.
        if _is_infrastructure_call(handler_call_details):
            return handler

        def make_wrapper(behavior, shape):
            if shape not in _STREAMING_SHAPES:
                return behavior

            is_streaming_response = shape in ("unary_stream", "stream_stream")

            if is_streaming_response:
                async def wrapped(request_or_iterator, context):
                    user_id = user_var.get()["user_id"]
                    if not await self._acquire(user_id):
                        await context.abort(
                            grpc.StatusCode.RESOURCE_EXHAUSTED,
                            f"Too many concurrent streams for this user (limit {self._max_streams}).",
                        )
                        return
                    try:
                        async for item in iter_streaming(behavior, request_or_iterator, context):
                            yield item
                    finally:
                        await self._release(user_id)
            else:
                async def wrapped(request_or_iterator, context):
                    user_id = user_var.get()["user_id"]
                    if not await self._acquire(user_id):
                        await context.abort(
                            grpc.StatusCode.RESOURCE_EXHAUSTED,
                            f"Too many concurrent streams for this user (limit {self._max_streams}).",
                        )
                        return
                    try:
                        return await call_unary(behavior, request_or_iterator, context)
                    finally:
                        await self._release(user_id)

            return wrapped

        return wrap_handler(handler, make_wrapper)

    async def _acquire(self, user_id: str) -> bool:
        key = _stream_key(user_id)
        try:
            current = await self._redis_client.incr(key)
            if current == 1:
                # Crash-safety net only -- normal completion/cancellation
                # always decrements in `finally` well before this would fire.
                await self._redis_client.expire(key, 3600)
            if current > self._max_streams:
                await self._redis_client.decr(key)
                return False
            return True
        except Exception:
            _logger.warning("Redis unavailable for stream limit check -- failing open", exc_info=True)
            return True

    async def _release(self, user_id: str) -> None:
        try:
            await self._redis_client.decr(_stream_key(user_id))
        except Exception:
            _logger.warning("Redis unavailable while releasing a stream slot", exc_info=True)
