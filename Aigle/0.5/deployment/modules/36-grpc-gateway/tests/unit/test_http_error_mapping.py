"""Downstream HTTP errors (module 13's StorageService / PersonalDBService raise
HTTPException(status_code=<downstream status>)) must become the matching gRPC status. They used
to fall through to INTERNAL with the raw downstream JSON as the message."""
from __future__ import annotations

import grpc
import pytest
from fastapi import HTTPException
from interceptors.errors import _map_unexpected


@pytest.mark.parametrize(
    "status,expected",
    [
        (400, grpc.StatusCode.INVALID_ARGUMENT),
        (401, grpc.StatusCode.UNAUTHENTICATED),
        (403, grpc.StatusCode.PERMISSION_DENIED),
        (404, grpc.StatusCode.NOT_FOUND),
        (409, grpc.StatusCode.ALREADY_EXISTS),
        (412, grpc.StatusCode.FAILED_PRECONDITION),
        (422, grpc.StatusCode.INVALID_ARGUMENT),
        (429, grpc.StatusCode.RESOURCE_EXHAUSTED),
        (418, grpc.StatusCode.INVALID_ARGUMENT),  # any other 4xx is the caller's request
        (501, grpc.StatusCode.UNIMPLEMENTED),
        (502, grpc.StatusCode.UNAVAILABLE),
        (503, grpc.StatusCode.UNAVAILABLE),
        (504, grpc.StatusCode.DEADLINE_EXCEEDED),
        (500, grpc.StatusCode.INTERNAL),
        (599, grpc.StatusCode.INTERNAL),
    ],
)
def test_http_status_maps_to_the_matching_grpc_status(status, expected):
    code, _ = _map_unexpected(HTTPException(status_code=status, detail="x"))
    assert code == expected


def test_a_4xx_message_unwraps_the_asset_service_body():
    detail = 'Asset service error: {"detail":"File video/mp4/x or commit v1 not found"}'
    code, message = _map_unexpected(HTTPException(status_code=404, detail=detail))
    assert code == grpc.StatusCode.NOT_FOUND
    assert message == "File video/mp4/x or commit v1 not found"


def test_a_4xx_message_that_is_not_wrapped_is_passed_through_truncated():
    _, message = _map_unexpected(HTTPException(status_code=400, detail="y" * 1000))
    assert message == "y" * 300


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_a_5xx_message_never_carries_the_downstream_body(status):
    _, message = _map_unexpected(HTTPException(status_code=status, detail='Asset service error: {"detail":"internal-host-7 stack trace"}'))
    assert "internal-host" not in message and "Asset service error" not in message


@pytest.mark.parametrize("detail", ["", 'Asset service error: {"detail": ""}', "   ", "Asset service error: "])
def test_a_4xx_with_an_empty_body_still_tells_the_caller_something(detail):
    code, message = _map_unexpected(HTTPException(status_code=400, detail=detail))
    assert code == grpc.StatusCode.INVALID_ARGUMENT
    assert message == "The request was rejected by the downstream service (HTTP 400)."
