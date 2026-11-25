"""Application error hierarchy and the interceptor that maps them to gRPC status codes.

This is the innermost interceptor in the chain (closest to the actual RPC
handler) so it catches exceptions raised by business logic before they'd
otherwise become an uncaught-exception UNKNOWN status.
"""
from __future__ import annotations

import json
import logging
import socket

import grpc
import httpx
from fastapi import HTTPException

from interceptors._base import call_unary, iter_streaming, wrap_handler
from interceptors.context import auth_error_var

_logger = logging.getLogger("grpc_gateway.errors")


class AppError(Exception):
    """Base class for application errors with a fixed gRPC status code."""

    status_code = grpc.StatusCode.INTERNAL

    def __init__(self, message: str = ""):
        super().__init__(message)
        self.message = message or self.__class__.__name__


class ValidationError(AppError):
    status_code = grpc.StatusCode.INVALID_ARGUMENT


class NotFoundError(AppError):
    status_code = grpc.StatusCode.NOT_FOUND


class AlreadyExistsError(AppError):
    status_code = grpc.StatusCode.ALREADY_EXISTS


class UnauthenticatedError(AppError):
    status_code = grpc.StatusCode.UNAUTHENTICATED


class PermissionDeniedError(AppError):
    status_code = grpc.StatusCode.PERMISSION_DENIED


class FailedPreconditionError(AppError):
    status_code = grpc.StatusCode.FAILED_PRECONDITION


class ResourceExhaustedError(AppError):
    status_code = grpc.StatusCode.RESOURCE_EXHAUSTED


class UnavailableError(AppError):
    status_code = grpc.StatusCode.UNAVAILABLE


class DeadlineExceededError(AppError):
    status_code = grpc.StatusCode.DEADLINE_EXCEEDED


class InternalError(AppError):
    status_code = grpc.StatusCode.INTERNAL


class UnimplementedError(AppError):
    status_code = grpc.StatusCode.UNIMPLEMENTED


class DataLossError(AppError):
    status_code = grpc.StatusCode.DATA_LOSS


# What module 13's StorageService / PersonalDBService raise (HTTPException carrying the
# downstream status). Without this they all became INTERNAL, e.g. a missing asset.
_HTTP_STATUS_TO_GRPC = {
    400: grpc.StatusCode.INVALID_ARGUMENT,
    401: grpc.StatusCode.UNAUTHENTICATED,
    403: grpc.StatusCode.PERMISSION_DENIED,
    404: grpc.StatusCode.NOT_FOUND,
    409: grpc.StatusCode.ALREADY_EXISTS,
    412: grpc.StatusCode.FAILED_PRECONDITION,
    422: grpc.StatusCode.INVALID_ARGUMENT,
    429: grpc.StatusCode.RESOURCE_EXHAUSTED,
    501: grpc.StatusCode.UNIMPLEMENTED,
    502: grpc.StatusCode.UNAVAILABLE,
    503: grpc.StatusCode.UNAVAILABLE,
    504: grpc.StatusCode.DEADLINE_EXCEEDED,
}
_ASSET_SERVICE_PREFIX = "Asset service error: "
_MAX_CLIENT_MESSAGE = 300


def _http_client_message(detail) -> str:
    """The part of a 4xx answer that is meant for the caller. StorageService wraps the asset
    service's JSON body as `Asset service error: {"detail": "..."}`; unwrap it."""
    text = detail if isinstance(detail, str) else json.dumps(detail, ensure_ascii=False)
    if text.startswith(_ASSET_SERVICE_PREFIX):
        text = text[len(_ASSET_SERVICE_PREFIX):]
        try:
            inner = json.loads(text).get("detail")
            if isinstance(inner, str):
                text = inner
        except (json.JSONDecodeError, AttributeError):
            pass
    return text[:_MAX_CLIENT_MESSAGE]


def _map_http_exception(exc: HTTPException) -> tuple[grpc.StatusCode, str]:
    status = exc.status_code
    if status >= 500:
        # A downstream fault is not the caller's to read: no raw body, no internal names.
        code = _HTTP_STATUS_TO_GRPC.get(status, grpc.StatusCode.INTERNAL)
        if code == grpc.StatusCode.DEADLINE_EXCEEDED:
            return code, "Downstream service timed out."
        if code == grpc.StatusCode.UNAVAILABLE:
            return code, "Downstream service unavailable."
        return code, "Downstream service error."
    message = _http_client_message(exc.detail).strip()
    if not message:
        # module 13 builds `Asset service error: {response.text}`, which is only the prefix when the
        # asset service answers with an empty body. That must not become an empty gRPC message:
        # the caller would get a bare status code and no hint what to change.
        message = f"The request was rejected by the downstream service (HTTP {status})."
    return _HTTP_STATUS_TO_GRPC.get(status, grpc.StatusCode.INVALID_ARGUMENT), message


def _map_unexpected(exc: Exception) -> tuple[grpc.StatusCode, str]:
    """Status for a non-AppError exception. A downstream service that can't be
    reached or doesn't answer is a transient condition the caller can retry,
    not an INTERNAL bug -- and its raw transport message (resolver errors,
    hostnames) shouldn't be echoed back to clients."""
    if isinstance(exc, HTTPException):
        return _map_http_exception(exc)
    if isinstance(exc, httpx.TimeoutException):
        return grpc.StatusCode.DEADLINE_EXCEEDED, "Downstream service timed out."
    if isinstance(exc, (httpx.TransportError, ConnectionError, socket.gaierror)):
        return grpc.StatusCode.UNAVAILABLE, "Downstream service unavailable."
    return grpc.StatusCode.INTERNAL, str(exc)


def _abort_args(method_name: str, exc: Exception) -> tuple[grpc.StatusCode, str]:
    """Status and message for an unexpected exception, logged at the right level: a downstream
    4xx (e.g. a missing asset) is the caller's request being wrong, not a gateway fault."""
    code, message = _map_unexpected(exc)
    if isinstance(exc, HTTPException) and exc.status_code < 500:
        _logger.warning("%s -> %s: %s", method_name, code, message)
    else:
        _logger.exception("%s -> unhandled exception", method_name)
    return code, message


class ExceptionMappingInterceptor(grpc.aio.ServerInterceptor):
    """Maps AppError subclasses (and any other exception) to a gRPC status code."""

    async def intercept_service(self, continuation, handler_call_details):
        handler = await continuation(handler_call_details)
        if handler is None:
            return None

        method_name = handler_call_details.method

        def make_wrapper(behavior, shape):
            is_streaming_response = shape in ("unary_stream", "stream_stream")

            if is_streaming_response:
                async def wrapped(request_or_iterator, context):
                    try:
                        pending_auth_error = auth_error_var.get()
                        if pending_auth_error is not None:
                            code, message = pending_auth_error
                            await context.abort(code, message)
                        async for response in iter_streaming(behavior, request_or_iterator, context):
                            yield response
                    except AppError as exc:
                        _logger.warning("%s -> %s: %s", method_name, exc.status_code, exc.message)
                        await context.abort(exc.status_code, exc.message)
                    except grpc.aio.AbortError:
                        raise
                    except Exception as exc:  # noqa: BLE001 - last-resort catch-all; _abort_args logs it and maps it
                        await context.abort(*_abort_args(method_name, exc))
            else:
                async def wrapped(request_or_iterator, context):
                    try:
                        pending_auth_error = auth_error_var.get()
                        if pending_auth_error is not None:
                            code, message = pending_auth_error
                            await context.abort(code, message)
                        return await call_unary(behavior, request_or_iterator, context)
                    except AppError as exc:
                        _logger.warning("%s -> %s: %s", method_name, exc.status_code, exc.message)
                        await context.abort(exc.status_code, exc.message)
                    except grpc.aio.AbortError:
                        raise
                    except Exception as exc:  # noqa: BLE001 - last-resort catch-all; _abort_args logs it and maps it
                        await context.abort(*_abort_args(method_name, exc))

            return wrapped

        return wrap_handler(handler, make_wrapper)
