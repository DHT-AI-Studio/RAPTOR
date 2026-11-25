"""AuthorizationInterceptor — per-RPC permission via module 06's UMA check.

Reuses exactly what REST already does (check_uma_permission / /auth/permission):
the RPC's full method name (e.g. /raptor.v1.RaptorService/SubmitJob) is the UMA
resource URI, the same shape request.url.path is on the REST side. Keycloak's
raptor:inference:resource / raptor:fine-tuning:resource cover it via
KEYCLOAK_INFERENCE_URIS / KEYCLOAK_FINETUNING_URIS (06-authentication/app/
sync_resources.py) -- no gRPC-specific policy model, same client/policies REST uses.

The permission check runs inside the wrapper, not intercept_service(), so it can
read auth_error_var first and skip the call entirely when an earlier interceptor
(Authentication/Identity) already rejected the call -- no point asking "can this
caller do X" when the caller isn't established at all.
"""
from __future__ import annotations

import logging

import grpc
import httpx
from config import get_settings

from interceptors._base import call_unary, iter_streaming, wrap_handler
from interceptors.auth import _is_exempt
from interceptors.context import auth_error_var

_logger = logging.getLogger("grpc_gateway.authz")

_PERMISSION_DENIED_MESSAGE = "Not permitted to call this RPC."
_INVALID_TOKEN_MESSAGE = "Invalid or expired token."
_AUTH_SERVICE_UNAVAILABLE_MESSAGE = "Cannot reach the authorization service."


def _bearer_token(handler_call_details) -> str | None:
    for key, value in handler_call_details.invocation_metadata or []:
        if key.lower() == "authorization" and value.lower().startswith("bearer "):
            return value[len("Bearer "):]
    return None


class AuthorizationInterceptor(grpc.aio.ServerInterceptor):
    def __init__(self, auth_service_url: str | None = None, http_client: httpx.AsyncClient | None = None):
        settings = get_settings()
        self._auth_service_url = (auth_service_url if auth_service_url is not None else settings.auth_service_url).rstrip("/")
        self._http = http_client or httpx.AsyncClient(timeout=5.0)

    async def _check(self, handler_call_details) -> tuple[grpc.StatusCode, str] | None:
        token = _bearer_token(handler_call_details)
        if token is None:
            return None  # x-api-key call: no UMA token to check, same as IdentityInterceptor's dev-user path
        try:
            resp = await self._http.get(
                f"{self._auth_service_url}/auth/permission",
                params={"request_path": handler_call_details.method},
                headers={"Authorization": f"Bearer {token}"},
            )
        except httpx.HTTPError as exc:
            _logger.warning("Cannot reach auth service for permission check: %s", exc)
            return grpc.StatusCode.UNAVAILABLE, _AUTH_SERVICE_UNAVAILABLE_MESSAGE
        if resp.status_code == 200:
            return None
        if resp.status_code == 403:
            return grpc.StatusCode.PERMISSION_DENIED, _PERMISSION_DENIED_MESSAGE
        return grpc.StatusCode.UNAUTHENTICATED, _INVALID_TOKEN_MESSAGE

    async def intercept_service(self, continuation, handler_call_details):
        handler = await continuation(handler_call_details)
        if handler is None or _is_exempt(handler_call_details):
            return handler

        def make_wrapper(behavior, shape):
            is_streaming_response = shape in ("unary_stream", "stream_stream")

            if is_streaming_response:
                async def wrapped(request_or_iterator, context):
                    if auth_error_var.get() is None:
                        error = await self._check(handler_call_details)
                        if error is not None:
                            auth_error_var.set(error)
                    async for item in iter_streaming(behavior, request_or_iterator, context):
                        yield item
            else:
                async def wrapped(request_or_iterator, context):
                    if auth_error_var.get() is None:
                        error = await self._check(handler_call_details)
                        if error is not None:
                            auth_error_var.set(error)
                    return await call_unary(behavior, request_or_iterator, context)

            return wrapped

        return wrap_handler(handler, make_wrapper)
