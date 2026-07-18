"""Security utilities for JWT verification and permission checks via module 06 auth service."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

import httpx
import jwt
from fastapi import HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import ExpiredSignatureError, InvalidTokenError

from app.core.config import Settings, get_settings

_logger = logging.getLogger(__name__)

_jwks_cache: Any = None

bearer_scheme = HTTPBearer(auto_error=True)


class AuthenticationError(HTTPException):
    """HTTPException subclass for authentication failures."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(status_code=status_code, detail=detail)


def _extract_bearer_token(request: Request) -> str:
    header = request.headers.get("Authorization")
    if not header:
        raise AuthenticationError(status.HTTP_401_UNAUTHORIZED, "Missing Authorization header")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise AuthenticationError(status.HTTP_401_UNAUTHORIZED, "Invalid authorization scheme")
    return token


def _get_public_key(token: str, use_cache: bool = True) -> Any:
    global _jwks_cache
    if use_cache and _jwks_cache is not None:
        return _jwks_cache

    try:
        unverified = jwt.decode(token, options={"verify_signature": False})
        iss = unverified.get("iss", "")
        if not iss:
            raise AuthenticationError(
                status.HTTP_401_UNAUTHORIZED, "Token missing iss claim"
            )
        # iss may contain an external IP (e.g. http://192.168.x.x:8080/realms/xxx)
        # which is unreachable from inside Docker. Extract the realm and use the
        # internal Keycloak hostname instead.
        parts = iss.split("/realms/", 1)
        if len(parts) != 2:
            raise AuthenticationError(
                status.HTTP_401_UNAUTHORIZED, f"Cannot parse realm from iss: {iss}"
            )
        realm = parts[1].split("/")[0]
        keycloak_url = get_settings().keycloak_url.rstrip("/")
        jwks_url = f"{keycloak_url}/realms/{realm}/protocol/openid-connect/certs"
    except AuthenticationError:
        raise
    except Exception as exc:
        raise AuthenticationError(
            status.HTTP_401_UNAUTHORIZED, f"Cannot parse token issuer: {exc}"
        ) from exc

    try:
        resp = httpx.get(jwks_url, timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        raise AuthenticationError(
            status.HTTP_503_SERVICE_UNAVAILABLE, f"Cannot reach Keycloak: {exc}"
        ) from exc

    for key in resp.json().get("keys", []):
        if key.get("use") == "sig" and key.get("kty") == "RSA":
            _jwks_cache = jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(key))
            return _jwks_cache

    raise AuthenticationError(
        status.HTTP_503_SERVICE_UNAVAILABLE, "No RS256 signing key found in JWKS"
    )


async def check_uma_permission(token: str, request_path: str) -> None:
    """Call module 06 /auth/permission to verify the token has access to request_path."""
    settings = get_settings()
    url = f"{settings.auth_service_url}/auth/permission"

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                url,
                params={"request_path": request_path},
                headers={"Authorization": f"Bearer {token}"},
            )
    except Exception as exc:
        raise AuthenticationError(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"Cannot reach auth service: {exc}",
        ) from exc

    if resp.status_code == 401:
        raise AuthenticationError(status.HTTP_401_UNAUTHORIZED, f"Unauthorized: {resp.text}")
    if resp.status_code == 403:
        try:
            detail = resp.json().get("detail") or resp.text
        except Exception:
            detail = resp.text or f"Permission denied for: {request_path}"
        _permission_denied_keywords = ("access_denied", "request_denied", "not_authorized", "forbidden", "not allowed", "no permission")
        if any(k in detail.lower() for k in _permission_denied_keywords):
            raise AuthenticationError(status.HTTP_403_FORBIDDEN, detail)
        raise AuthenticationError(
            status.HTTP_401_UNAUTHORIZED,
            f"Session expired due to inactivity, please log in again ({detail})",
        )
    if resp.status_code != 200:
        raise AuthenticationError(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"Auth service error {resp.status_code}: {resp.text}",
        )


def verify_jwt(request: Request, settings: Optional[Settings] = None) -> Dict[str, Any]:
    token = _extract_bearer_token(request)

    try:
        public_key = _get_public_key(token=token)
        return jwt.decode(token, public_key, algorithms=["RS256"], options={"verify_aud": False})
    except ExpiredSignatureError as exc:
        _logger.warning("JWT expired")
        raise AuthenticationError(status.HTTP_401_UNAUTHORIZED, "Token expired") from exc
    except InvalidTokenError:
        # Retry once with a fresh key in case Keycloak rotated keys
        try:
            public_key = _get_public_key(token=token, use_cache=False)
            return jwt.decode(token, public_key, algorithms=["RS256"], options={"verify_aud": False})
        except ExpiredSignatureError as exc:
            raise AuthenticationError(status.HTTP_401_UNAUTHORIZED, "Token expired") from exc
        except InvalidTokenError as exc:
            _logger.warning("JWT invalid", exc_info=exc)
            raise AuthenticationError(status.HTTP_403_FORBIDDEN, "Invalid token") from exc
