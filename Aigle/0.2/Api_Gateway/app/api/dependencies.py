"""Shared FastAPI dependencies for the gateway."""
from __future__ import annotations

from typing import Any, Dict

from fastapi import Depends, HTTPException, Request, Security, status
#from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, OAuth2PasswordBearer

from app.core.config import Settings, get_settings
from app.core.security import verify_jwt
from app.services.auth_service import AuthService
from app.services.user_service import UserService

from keycloak.app.dependencies.security import APIsecurity

# OAuth2 Password flow for Swagger Authorize dialog
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)

# Keep HTTP Bearer in case you want to support simple bearer scheme too (optional)
#http_bearer = HTTPBearer(auto_error=False)


async def get_user_service(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> UserService:
    """Return the user service instance, initialising it if needed."""
    service: UserService | None = getattr(request.app.state, "user_service", None)
    if service is None:
        service = UserService(
            settings.gateway_db_dsn,
            min_size=settings.gateway_db_pool_min_size,
            max_size=settings.gateway_db_pool_max_size,
        )
        await service.initialise()
        request.app.state.user_service = service
    elif not service.is_ready:
        await service.initialise()
    return service


def get_auth_service(
    settings: Settings = Depends(get_settings),
    user_service: UserService = Depends(get_user_service),
) -> AuthService:
    """Construct an authentication service bound to the shared user store."""
    return AuthService(settings, user_service)


def get_current_user(
    request: Request,
    token: str | None = Depends(APIsecurity.authenticate),
    settings: Settings = Depends(get_settings),
) -> Dict[str, Any]:
    """Dependency that validates the bearer token and returns its payload.

    Using OAuth2PasswordBearer makes Swagger UI display username/password fields
    in the Authorize modal and automatically call /login to obtain a token.
    """
    payload = verify_jwt(request, settings=settings)
    if "sub" not in payload:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid token payload",
        )
    return payload
