"""Shared FastAPI dependencies for the gateway."""
from __future__ import annotations

from typing import Any, Dict

import httpx
from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import OAuth2PasswordBearer

from app.core.config import Settings, get_settings
from app.core.rate_limiter import check_rate_limit
from app.core.security import check_uma_permission, verify_jwt
from app.services.storage_service import StorageService

# OAuth2 Password Flow for Swagger UI - points to SSO login endpoint
_bearer = OAuth2PasswordBearer(tokenUrl=f"/api/{get_settings().api_version}/sso/login", auto_error=True)


def get_http_client(request: Request) -> httpx.AsyncClient:
    """Return the shared async HTTP client from app state."""
    client = getattr(request.app.state, "http_client", None)
    if not client:
        raise RuntimeError("HTTP client is not initialised")
    return client


def get_storage_service(
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
) -> StorageService:
    """Return a StorageService instance backed by the shared HTTP client."""
    return StorageService(client, settings)


async def get_current_user(
    request: Request,
    token: str = Depends(_bearer),
    settings: Settings = Depends(get_settings),
) -> Dict[str, Any]:
    """Validate the bearer token via JWT signature check and module 06 /auth/permission check."""
    payload = verify_jwt(request, settings=settings)
    if "sub" not in payload:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid token payload",
        )
    await check_uma_permission(token, request.url.path)

    redis = getattr(request.app.state, "redis_client", None)
    if redis is not None:
        await check_rate_limit(redis, payload["sub"], settings.rate_limit_per_user_per_minute)

    return payload
