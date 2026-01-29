"""Security utilities for JWT verification."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import jwt
from fastapi import HTTPException, Request, status
from jwt import ExpiredSignatureError, InvalidAudienceError, InvalidTokenError

from app.core.config import Settings, get_settings

_logger = logging.getLogger(__name__)


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


# def verify_jwt(request: Request, settings: Optional[Settings] = None) -> Dict[str, Any]:
#     """Validate the JWT bearer token from the incoming request."""

#     settings = settings or get_settings()
#     token = _extract_bearer_token(request)

#     try:
#         payload = jwt.decode(
#             token,
#             settings.jwt_secret_key,
#             algorithms=[settings.jwt_algorithm],
#             audience=settings.jwt_audience,
#             options={"verify_aud": settings.jwt_audience is not None},
#         )
#     except ExpiredSignatureError as exc:
#         _logger.warning("JWT expired", exc_info=exc)
#         raise AuthenticationError(status.HTTP_401_UNAUTHORIZED, "Token expired") from exc
#     except InvalidAudienceError as exc:
#         _logger.warning("JWT audience mismatch", exc_info=exc)
#         raise AuthenticationError(status.HTTP_403_FORBIDDEN, "Invalid token audience") from exc
#     except InvalidTokenError as exc:
#         _logger.warning("JWT invalid", exc_info=exc)
#         raise AuthenticationError(status.HTTP_403_FORBIDDEN, "Invalid token") from exc

#     return payload

from app.my_keycloak.scripts import keycloak_master
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends
import json
bearer_scheme = HTTPBearer(auto_error=True)  # 這裡 auto_error=True，沒有 token 就 401

# def verify_jwt(request: Request, 
#                settings: Optional[Settings] = None,
#                token: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=True)),
#                ) -> Dict[str, Any]:
#     """Validate the JWT bearer token from the incoming request."""

#     settings = settings or get_settings()
#     #token = _extract_bearer_token(request)
#     user_token = token.credentials

#     try:
#         public_key = keycloak_master.KeycloakMaster.get_realm_key("dhtsolution").json()["keys"][1]["publicKey"]  # 取 RS256 key
#         public_key_pem = f"-----BEGIN PUBLIC KEY-----\n{public_key}\n-----END PUBLIC KEY-----"

#         if type(user_token) != str:
#             if type(user_token) == dict:
#                 user_token = user_token["access_token"]
#             else:
#                 user_token = user_token.json()["access_token"]

#         payload = jwt.decode(
#             user_token,
#             public_key_pem,
#             algorithms=["RS256"],
#             options={"verify_aud": False}  # 不驗證 audience
#             #audience=decode_payload(user_token)["aud"]   # 或你的 client_id
#         )

#     except ExpiredSignatureError as exc:
#         _logger.warning("JWT expired", exc_info=exc)
#         raise AuthenticationError(status.HTTP_401_UNAUTHORIZED, "Token expired") from exc
#     except InvalidAudienceError as exc:
#         _logger.warning("JWT audience mismatch", exc_info=exc)
#         raise AuthenticationError(status.HTTP_403_FORBIDDEN, "Invalid token audience") from exc
#     except InvalidTokenError as exc:
#         _logger.warning("JWT invalid", exc_info=exc)
#         raise AuthenticationError(status.HTTP_403_FORBIDDEN, "Invalid token") from exc

#     return payload

def verify_jwt(request: Request, settings: Optional[Settings] = None) -> dict:
    settings = settings or get_settings()
    
    token_header = request.headers.get("Authorization")
    if not token_header:
        raise AuthenticationError(401, "Missing Authorization header")
    scheme, _, token = token_header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise AuthenticationError(401, "Invalid authorization scheme")

    public_key = keycloak_master.KeycloakMaster.get_realm_key("dhtsolution").json()["keys"][1]["publicKey"]
    public_key_pem = f"-----BEGIN PUBLIC KEY-----\n{public_key}\n-----END PUBLIC KEY-----"

    payload = jwt.decode(
        token,
        public_key_pem,
        algorithms=["RS256"],
        options={"verify_aud": False},
    )
    return payload

