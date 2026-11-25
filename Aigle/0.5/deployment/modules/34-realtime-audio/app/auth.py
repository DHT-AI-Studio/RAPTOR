"""Bearer JWT verification for POST /audio/session/token.

Same approach as Module 13 (app/core/security.py): RS256 signature checked against
the Keycloak realm's JWKS, where the realm comes from the token's ``iss`` and the
JWKS is fetched from the internal ``KEYCLOAK_URL`` (the public iss host may not be
reachable from inside Docker).  ``RT_JWT_PUBLIC_KEY`` (PEM) replaces the JWKS
lookup for deployments/tests without Keycloak.
"""
from __future__ import annotations

import json
import logging
from typing import Annotated, Any

import httpx
import jwt
from fastapi import Depends, Request

from app.config import Settings
from app.errors import ServiceError

logger = logging.getLogger(__name__)


class KeyProvider:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._static = settings.jwt_public_key or None
        self._jwks: dict[str, Any] = {}   # realm url -> key

    async def key_for(self, token: str, refresh: bool = False) -> Any:
        if self._static:
            return self._static
        try:
            iss = jwt.decode(token, options={"verify_signature": False}).get("iss", "")
        except jwt.InvalidTokenError as exc:
            raise ServiceError(401, "invalid_token", "malformed token") from exc
        parts = iss.split("/realms/", 1)
        if len(parts) != 2 or not parts[1]:
            raise ServiceError(401, "invalid_token", "cannot parse realm from iss")
        realm = parts[1].split("/")[0]
        url = f"{self._settings.keycloak_url}/realms/{realm}/protocol/openid-connect/certs"
        if not refresh and url in self._jwks:
            return self._jwks[url]
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url)
                resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ServiceError(503, "auth_unavailable", f"cannot reach Keycloak: {exc}") from exc
        for key in resp.json().get("keys", []):
            if key.get("use") == "sig" and key.get("kty") == "RSA":
                self._jwks[url] = jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(key))
                return self._jwks[url]
        raise ServiceError(503, "auth_unavailable", "no RS256 signing key in JWKS")

    async def verify(self, token: str) -> dict[str, Any]:
        options = {"verify_aud": bool(self._settings.jwt_audience)}
        audience: str | None = self._settings.jwt_audience or None
        for refresh in (False, True):
            key = await self.key_for(token, refresh=refresh)
            try:
                return jwt.decode(token, key, algorithms=["RS256"], audience=audience, options=options)
            except jwt.ExpiredSignatureError as exc:
                raise ServiceError(401, "token_expired", "token expired") from exc
            except jwt.InvalidTokenError as exc:
                if refresh or self._static:   # retried with a fresh JWKS already
                    raise ServiceError(401, "invalid_token", str(exc)) from exc
        raise ServiceError(401, "invalid_token")  # pragma: no cover


def get_key_provider(request: Request) -> KeyProvider:
    return request.app.state.key_provider


async def get_current_user(
    request: Request, keys: Annotated[KeyProvider, Depends(get_key_provider)]
) -> dict[str, Any]:
    header = request.headers.get("Authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise ServiceError(401, "missing_token", "Authorization: Bearer <JWT> required",
                           headers={"WWW-Authenticate": "Bearer"})
    claims = await keys.verify(token)
    if not claims.get("sub"):
        raise ServiceError(401, "invalid_token", "token has no sub claim")
    return claims
