from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.services.token_manager import TokenManager

router = APIRouter()

# Declared as an HTTP Bearer scheme so Swagger renders an "Authorize" button for
# these routes. The gateway (Module 13) does the real JWT validation; here only
# the presence of a human's Bearer token is required.
_human_bearer = HTTPBearer(
    auto_error=False,
    description="The human's JWT authorizing agent registration (any Bearer token).",
)


def _get_token_manager(request: Request) -> TokenManager:
    tm = getattr(request.app.state, "token_manager", None)
    if tm is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Token manager not initialised")
    return tm


def require_human_jwt(
    cred: HTTPAuthorizationCredentials = Depends(_human_bearer),
) -> str:
    """Registering/revoking an agent must be initiated by an authenticated human.

    Requires an ``Authorization: Bearer <jwt>`` to be present so the endpoint
    cannot be called anonymously even when hit directly. The gateway (Module 13)
    performs the real JWT validation, so any Bearer token is accepted here.
    """
    if cred is None or not cred.credentials.strip():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Human JWT required to register an agent")
    return cred.credentials.strip()


class RegisterRequest(BaseModel):
    client_id: str = Field(..., description="Keycloak confidential client id (service account).")
    client_secret: str = Field(..., description="Client secret for the service account.")
    scope: Optional[str] = Field(None, description="Optional space-separated OAuth2 scopes.")


class RegisterResponse(BaseModel):
    agent_token: str
    expires_in: int


@router.post("/register", response_model=RegisterResponse, tags=["Agent Auth"])
async def register_agent(
    body: RegisterRequest,
    _human_jwt: str = Depends(require_human_jwt),
    tm: TokenManager = Depends(_get_token_manager),
) -> RegisterResponse:
    """Register an agent's client credentials for autonomous token refresh.

    Returns an opaque ``agent_token`` (Module 27-issued). Include it as
    ``Authorization: Bearer <agent_token>`` on MCP tool calls; the server
    resolves it to a Keycloak token and auto-refreshes via client_credentials.
    """
    try:
        agent_token = await tm.register_agent(body.client_id, body.client_secret, body.scope)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail=f"Client credential validation failed: {exc}")

    return RegisterResponse(agent_token=agent_token, expires_in=get_settings().agent_token_ttl_seconds)


@router.delete("/register/{agent_token}", status_code=status.HTTP_204_NO_CONTENT,
               response_model=None, tags=["Agent Auth"])
async def revoke_agent(
    agent_token: str,
    _human_jwt: str = Depends(require_human_jwt),
    tm: TokenManager = Depends(_get_token_manager),
) -> None:
    """Revoke an agent_token (removes its Redis credential blob).

    response_model=None is required here: with `from __future__ import
    annotations` (top of file), FastAPI reads `-> None` as the string
    "None" rather than the actual None object and infers a truthy
    response_model, which then fails FastAPI's "204 must have no body"
    assertion at import time — crashing the whole app on startup.
    """
    removed = await tm.revoke_agent(agent_token)
    if not removed:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="Unknown or already-revoked agent_token")
