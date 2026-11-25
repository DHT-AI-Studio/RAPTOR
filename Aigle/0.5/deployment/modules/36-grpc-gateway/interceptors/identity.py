"""IdentityInterceptor — verified user identity from the Bearer JWT.

Runs after AuthenticationInterceptor, so it only sees calls that already
passed the presence check -- but "present" is not "valid": a Bearer token's
signature is verified here against Keycloak's JWKS, matched by the token's
own `kid` (so key rotation -- two signing keys briefly valid at once -- is
handled correctly, unlike grabbing "the first RS256 key" the JWKS lists).
`iss` must name the configured realm (not just any realm the same Keycloak
instance happens to host) and `exp` is checked by jwt.decode() itself;
`aud` is checked only if GRPC_GATEWAY_JWT_AUDIENCE is set (see config.py).

An invalid, expired or unverifiable token is flagged with auth_error_var the
same way AuthenticationInterceptor flags a missing one -- this interceptor
never aborts on the spot, ExceptionMappingInterceptor (innermost) does, so
the layers in between still observe rejected calls. GRPC_GATEWAY_ALLOW_
INSECURE_DEV_FALLBACK (default false) turns that rejection back into the
pre-VIE01-383 dev-user fallback -- see config.py, never true in production.

A call authenticated via x-api-key instead of a Bearer token has no `sub` to
verify an identity against -- those calls get the fixed service identity
(_DEV_USER), which is intentional: an API key authenticates the caller as a
trusted service, not as a specific end user. The same fallback applies when
Keycloak itself can't be reached to fetch JWKS, but only for an x-api-key
call -- a Bearer-token call in that situation is UNAVAILABLE, not waved
through as dev-user.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import grpc
import httpx
import jwt
from config import get_settings

from interceptors._base import call_unary, iter_streaming, wrap_handler
from interceptors.context import auth_error_var, user_var

_logger = logging.getLogger("grpc_gateway.identity")

_DEV_USER = {"user_id": "dev-user", "branch_id": "dev-user"}
_INVALID_TOKEN_MESSAGE = "Invalid or expired token."
_KEYCLOAK_UNAVAILABLE_MESSAGE = "Cannot reach Keycloak to verify the token."


class _TokenInvalid(Exception):
    """A Bearer token that is present but must not be trusted (bad signature,
    wrong realm, expired, malformed) -- UNAUTHENTICATED, not a retry condition."""


class _KeycloakUnavailable(Exception):
    """JWKS could not be fetched -- UNAVAILABLE, the caller may retry."""


class IdentityInterceptor(grpc.aio.ServerInterceptor):
    def __init__(
        self,
        keycloak_url: str | None = None,
        expected_realm: str | None = None,
        audience: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        dev_fallback: bool | None = None,
    ):
        settings = get_settings()
        self._keycloak_url = (keycloak_url if keycloak_url is not None else settings.keycloak_url).rstrip("/")
        self._expected_realm = expected_realm if expected_realm is not None else settings.jwt_expected_realm
        self._audience = audience if audience is not None else (settings.jwt_audience or None)
        self._http = http_client or httpx.AsyncClient(timeout=5.0)
        self._dev_fallback = dev_fallback if dev_fallback is not None else settings.allow_insecure_dev_fallback
        self._jwks: dict[str, Any] = {}  # kid -> RSAAlgorithm public key, from the last successful fetch

    async def _refresh_jwks(self) -> None:
        jwks_url = f"{self._keycloak_url}/realms/{self._expected_realm}/protocol/openid-connect/certs"
        try:
            resp = await self._http.get(jwks_url)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise _KeycloakUnavailable(str(exc)) from exc
        self._jwks = {
            key["kid"]: jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(key))
            for key in resp.json().get("keys", [])
            if key.get("use") == "sig" and key.get("kty") == "RSA" and "kid" in key
        }

    async def _signing_key(self, kid: str) -> Any:
        if kid not in self._jwks:
            await self._refresh_jwks()  # first call, or kid rotated in since the last fetch
        key = self._jwks.get(kid)
        if key is None:
            raise _TokenInvalid(f"No signing key for kid={kid!r}")
        return key

    async def _verify(self, token: str) -> dict:
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError as exc:
            raise _TokenInvalid(str(exc)) from exc
        kid = header.get("kid")
        if not kid:
            raise _TokenInvalid("Token header has no kid")
        try:
            unverified = jwt.decode(token, options={"verify_signature": False})
        except jwt.PyJWTError as exc:
            raise _TokenInvalid(str(exc)) from exc
        # iss may carry an externally-reachable host (e.g. the browser's own
        # address) unresolvable from inside Docker -- same issue module 13's
        # app/core/security.py handles the same way: JWKS always comes from
        # our own configured keycloak_url, never from iss itself. What iss
        # must still prove is which *realm* issued the token, so a token
        # from a different realm on the same Keycloak instance (e.g.
        # master) is rejected even though its signature would verify fine.
        iss = unverified.get("iss", "")
        realm = iss.split("/realms/", 1)[-1].split("/")[0] if "/realms/" in iss else ""
        if realm != self._expected_realm:
            raise _TokenInvalid(f"Unexpected realm in iss: {iss!r}")
        key = await self._signing_key(kid)
        decode_kwargs: dict[str, Any] = {"issuer": iss}
        if self._audience:
            decode_kwargs["audience"] = self._audience
        else:
            decode_kwargs["options"] = {"verify_aud": False}
        try:
            return jwt.decode(token, key, algorithms=["RS256"], **decode_kwargs)
        except jwt.ExpiredSignatureError as exc:
            # A fresh key can't un-expire a token -- refetching and retrying
            # would just fail the same way again, for no reason.
            raise _TokenInvalid("Token expired") from exc
        except jwt.PyJWTError:
            # Any other failure (bad signature, wrong issuer/audience) gets
            # one retry against a freshly-fetched key first: this kid was in
            # our cache, but we've never actually verified a token against it
            # before, so a stale cache entry is worth ruling out before
            # calling the token itself invalid.
            await self._refresh_jwks()
            key = self._jwks.get(kid)
            if key is None:
                raise _TokenInvalid(f"No signing key for kid={kid!r}") from None
            try:
                return jwt.decode(token, key, algorithms=["RS256"], **decode_kwargs)
            except jwt.PyJWTError as exc:
                raise _TokenInvalid(str(exc)) from exc

    async def _identify(self, handler_call_details) -> tuple[dict, tuple[grpc.StatusCode, str] | None]:
        """(user, error). error is None on success; user is _DEV_USER whenever there's no
        Bearer token to verify an identity from (an x-api-key call, already validated by
        AuthenticationInterceptor) -- but never when a Bearer token IS present and invalid."""
        token = None
        for key, value in handler_call_details.invocation_metadata or []:
            if key.lower() == "authorization" and value.lower().startswith("bearer "):
                token = value[len("Bearer "):]
                break
        if token is None:
            return _DEV_USER, None
        try:
            claims = await self._verify(token)
        except _TokenInvalid as exc:
            _logger.warning("Bearer token rejected: %s", exc)
            return self._fallback_or(grpc.StatusCode.UNAUTHENTICATED, _INVALID_TOKEN_MESSAGE)
        except _KeycloakUnavailable as exc:
            _logger.warning("Cannot verify Bearer token, Keycloak unreachable: %s", exc)
            return self._fallback_or(grpc.StatusCode.UNAVAILABLE, _KEYCLOAK_UNAVAILABLE_MESSAGE)
        sub = claims.get("sub")
        if not sub:
            return self._fallback_or(grpc.StatusCode.UNAUTHENTICATED, _INVALID_TOKEN_MESSAGE)
        return {"user_id": sub, "branch_id": sub}, None

    def _fallback_or(self, code: grpc.StatusCode, message: str) -> tuple[dict, tuple[grpc.StatusCode, str] | None]:
        """allow_insecure_dev_fallback=true (never in production) restores the
        pre-VIE01-383 behaviour: an unverifiable Bearer token becomes dev-user
        instead of being rejected. Default false -- (user, error)."""
        if self._dev_fallback:
            return _DEV_USER, None
        return _DEV_USER, (code, message)

    async def intercept_service(self, continuation, handler_call_details):
        handler = await continuation(handler_call_details)
        if handler is None:
            return None

        user, error = await self._identify(handler_call_details)

        def make_wrapper(behavior, shape):
            is_streaming_response = shape in ("unary_stream", "stream_stream")

            if is_streaming_response:
                async def wrapped(request_or_iterator, context):
                    user_var.set(user)
                    if error is not None:
                        auth_error_var.set(error)
                    async for item in iter_streaming(behavior, request_or_iterator, context):
                        yield item
            else:
                async def wrapped(request_or_iterator, context):
                    user_var.set(user)
                    if error is not None:
                        auth_error_var.set(error)
                    return await call_unary(behavior, request_or_iterator, context)

            return wrapped

        return wrap_handler(handler, make_wrapper)
