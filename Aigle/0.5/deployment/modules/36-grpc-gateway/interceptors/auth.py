"""AuthenticationInterceptor — JWT presence / API-key allow-list check.

A Bearer token's *format* is checked here (non-empty); its *signature* is
verified by IdentityInterceptor, which runs next and has the JWKS-fetch
logic -- this interceptor only needs to agree a credential of that shape was
supplied. An x-api-key credential has no further interceptor to check it, so
it's verified here, against the configured allow-list.

This interceptor never aborts on the spot — it only flags a rejection via
auth_error_var and always delegates. Aborting here would skip
RequestId/Logging/Metrics entirely for every rejected call, since Auth is the
outermost interceptor and those three sit deeper in the chain.
ExceptionMappingInterceptor (innermost) reads the flag and performs the actual
abort, so the layers in between still see rejected calls.
"""
from __future__ import annotations

import hmac

import grpc
from config import get_settings

from interceptors._base import call_unary, iter_streaming, wrap_handler
from interceptors.context import auth_error_var

# Infra-level services that must stay reachable without credentials — health
# probes (grpc_health_probe, k8s liveness/readiness) and reflection
# (grpcurl introspection) never send an authorization header.
_UNAUTHENTICATED_SERVICE_PREFIXES = (
    "/grpc.health.v1.Health/",
    "/grpc.reflection.v1alpha.ServerReflection/",
    "/grpc.reflection.v1.ServerReflection/",
)

_MISSING_CREDENTIALS_MESSAGE = "Missing credentials: expected 'authorization: Bearer <jwt>' or 'x-api-key: <key>'."
_INVALID_API_KEY_MESSAGE = "Invalid x-api-key."


def _is_exempt(handler_call_details) -> bool:
    return handler_call_details.method.startswith(_UNAUTHENTICATED_SERVICE_PREFIXES)


def _parse_api_keys(raw: str) -> frozenset[str]:
    return frozenset(k.strip() for k in raw.split(",") if k.strip())


def _valid_api_key(candidate: str, api_keys: frozenset[str]) -> bool:
    # Constant-time per candidate so a wrong-length guess doesn't exit faster
    # than a right-length one -- a timing side channel is cheap to close here.
    return any(hmac.compare_digest(candidate, key) for key in api_keys)


class AuthenticationInterceptor(grpc.aio.ServerInterceptor):
    """Flags a call with no credential, or an x-api-key not on the allow-list, with
    UNAUTHENTICATED; ExceptionMappingInterceptor aborts."""

    def __init__(self, api_keys: frozenset[str] | None = None):
        self._api_keys = api_keys if api_keys is not None else _parse_api_keys(get_settings().api_keys)

    def _auth_error(self, handler_call_details) -> tuple[grpc.StatusCode, str] | None:
        # Scan every metadata entry before deciding -- a client sending both a
        # valid Bearer token and a stale x-api-key must still pass; checking
        # x-api-key first and returning on an invalid one would reject that
        # call despite its Bearer token never having been looked at.
        bearer_present = False
        api_key_seen = False
        api_key_valid = False
        for key, value in handler_call_details.invocation_metadata or []:
            key = key.lower()
            # "Bearer " followed by nothing is not a credential.
            if key == "authorization" and value.lower().startswith("bearer ") and value[len("bearer "):].strip():
                bearer_present = True
            elif key == "x-api-key" and value:
                api_key_seen = True
                if _valid_api_key(value, self._api_keys):
                    api_key_valid = True
        if bearer_present or api_key_valid:
            return None
        if api_key_seen:
            return grpc.StatusCode.UNAUTHENTICATED, _INVALID_API_KEY_MESSAGE
        return grpc.StatusCode.UNAUTHENTICATED, _MISSING_CREDENTIALS_MESSAGE

    async def intercept_service(self, continuation, handler_call_details):
        handler = await continuation(handler_call_details)
        if handler is None or _is_exempt(handler_call_details):
            return handler
        error = self._auth_error(handler_call_details)
        if error is None:
            return handler

        def make_wrapper(behavior, shape):
            is_streaming_response = shape in ("unary_stream", "stream_stream")

            if is_streaming_response:
                async def wrapped(request_or_iterator, context):
                    auth_error_var.set(error)
                    async for item in iter_streaming(behavior, request_or_iterator, context):
                        yield item
            else:
                async def wrapped(request_or_iterator, context):
                    auth_error_var.set(error)
                    return await call_unary(behavior, request_or_iterator, context)

            return wrapped

        return wrap_handler(handler, make_wrapper)
