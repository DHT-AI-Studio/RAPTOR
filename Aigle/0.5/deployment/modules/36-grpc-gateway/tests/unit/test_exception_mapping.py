"""Unit tests for ExceptionMappingInterceptor's AppError -> gRPC status code table.

Exercises the interceptor directly against a minimal fake handler/context,
rather than a real RaptorService RPC — none of the nine AppError subclasses
are reachable through any real RPC yet (everything but Ping is UNIMPLEMENTED),
so this is the only way to cover the mapping table before business logic lands.
"""
from __future__ import annotations

import socket

import grpc
import httpx
import pytest
from interceptors.errors import (
    AlreadyExistsError,
    ExceptionMappingInterceptor,
    FailedPreconditionError,
    InternalError,
    NotFoundError,
    PermissionDeniedError,
    ResourceExhaustedError,
    UnauthenticatedError,
    UnavailableError,
    ValidationError,
)


class _FakeHandlerCallDetails:
    method = "/test.Service/Method"
    invocation_metadata = ()


class _FakeContext:
    def __init__(self):
        self.aborted_with = None

    async def abort(self, code, message=""):
        self.aborted_with = (code, message)
        raise grpc.aio.AbortError()


async def _run_and_capture_abort(raising_behavior):
    handler = grpc.unary_unary_rpc_method_handler(raising_behavior)

    async def continuation(_):
        return handler

    interceptor = ExceptionMappingInterceptor()
    wrapped_handler = await interceptor.intercept_service(continuation, _FakeHandlerCallDetails())

    context = _FakeContext()
    with pytest.raises(grpc.aio.AbortError):
        await wrapped_handler.unary_unary(None, context)
    return context.aborted_with


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc_cls,expected_code",
    [
        (ValidationError, grpc.StatusCode.INVALID_ARGUMENT),
        (NotFoundError, grpc.StatusCode.NOT_FOUND),
        (AlreadyExistsError, grpc.StatusCode.ALREADY_EXISTS),
        (UnauthenticatedError, grpc.StatusCode.UNAUTHENTICATED),
        (PermissionDeniedError, grpc.StatusCode.PERMISSION_DENIED),
        (FailedPreconditionError, grpc.StatusCode.FAILED_PRECONDITION),
        (ResourceExhaustedError, grpc.StatusCode.RESOURCE_EXHAUSTED),
        (UnavailableError, grpc.StatusCode.UNAVAILABLE),
        (InternalError, grpc.StatusCode.INTERNAL),
    ],
)
async def test_app_error_maps_to_expected_status_code(exc_cls, expected_code):
    async def raising_behavior(request, context):
        raise exc_cls("boom")

    code, message = await _run_and_capture_abort(raising_behavior)
    assert code == expected_code
    assert message == "boom"


@pytest.mark.asyncio
async def test_unhandled_exception_maps_to_internal():
    async def raising_behavior(request, context):
        raise ValueError("unexpected")

    code, _ = await _run_and_capture_abort(raising_behavior)
    assert code == grpc.StatusCode.INTERNAL


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc,expected_code",
    [
        (httpx.ConnectError("boom"), grpc.StatusCode.UNAVAILABLE),
        (httpx.ReadTimeout("slow"), grpc.StatusCode.DEADLINE_EXCEEDED),
        (socket.gaierror(-3, "Temporary failure in name resolution"), grpc.StatusCode.UNAVAILABLE),
        (ConnectionRefusedError("refused"), grpc.StatusCode.UNAVAILABLE),
        (RuntimeError("a real bug"), grpc.StatusCode.INTERNAL),
    ],
)
async def test_unexpected_exceptions_map_by_kind(exc, expected_code):
    async def behavior(request, context):
        raise exc

    code, message = await _run_and_capture_abort(behavior)
    assert code == expected_code
    if expected_code != grpc.StatusCode.INTERNAL:
        assert "Errno" not in message and "resolution" not in message  # raw transport text not echoed
