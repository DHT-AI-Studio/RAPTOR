from __future__ import annotations

import logging

from app.core.config import get_settings
from app.services.token_manager import OwnershipConflictError, TokenManager
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import jwt as jose_jwt
from jose.exceptions import JOSEError
from pydantic import BaseModel, Field

_logger = logging.getLogger(__name__)

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


def _best_effort_sub(human_jwt: str) -> str:
    """Extract the registering human's `sub` claim, unverified.

    Purely for the ownership-conflict check below (see OwnershipConflictError)
    -- never used as an authorization decision. Module 27 doesn't validate
    signatures (Module 13 already did, or this JWT wouldn't have reached
    here as a legitimate call); a garbage/unparseable token just falls back
    to "unknown" rather than failing the request, since ownership-tracking
    is a safety net, not the actual auth check (require_human_jwt already
    ran, and downstream Raptor calls re-validate independently).

    Security note (PYSEC-2026-4298, CVSS 9.3, no fix released -- python-jose
    is stuck at 3.5.0): that bug is in jwt.decode()'s signature-verification
    path (HMAC/asymmetric algorithm confusion via a DER-encoded public key).
    get_unverified_claims() never verifies a signature or takes a key at
    all -- it's a plain base64 decode of the claims segment -- so this call
    site can't reach the vulnerable code path regardless of what's in the
    token. Accepted as a non-issue here, not something to re-flag.
    """
    try:
        claims = jose_jwt.get_unverified_claims(human_jwt)
        sub = claims.get("sub")
        return sub if isinstance(sub, str) and sub else "unknown"
    except JOSEError as exc:
        _logger.warning("Could not decode registering human's JWT for ownership tracking: %s", exc)
        return "unknown"


class RegisterRequest(BaseModel):
    client_id: str = Field(..., description="Keycloak confidential client id (service account).")
    client_secret: str = Field(..., description="Client secret for the service account.")
    scope: str | None = Field(None, description="Optional space-separated OAuth2 scopes.")


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

    Only for genuinely autonomous, unattended agents — client_credentials is
    a client-only grant (no per-user ``sub``), so every agent registered
    against the same Keycloak client resolves to that client's one shared
    service-account identity, not the registering human. Do not use this as
    a way for a person to skip re-logging in; use their own Bearer JWT
    instead. client_id must be in MCP_AGENT_ALLOWED_CLIENT_IDS.

    Each client_id may only ever be registered by the human who first
    claimed it (409 for anyone else) — closes the practical failure mode of
    two different people accidentally sharing one client_id's identity,
    though it doesn't change what that identity actually is; see
    TokenManager.register_agent's OwnershipConflictError docs.
    """
    settings = get_settings()
    if body.client_id not in settings.agent_allowed_client_ids_set:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"client_id {body.client_id!r} is not in this deployment's agent allowlist "
                "(MCP_AGENT_ALLOWED_CLIENT_IDS). Agent registration is for autonomous, "
                "unattended integrations only — a human should use their own Bearer JWT instead."
            ),
        )
    registered_by = _best_effort_sub(_human_jwt)
    try:
        agent_token = await tm.register_agent(
            body.client_id, body.client_secret, body.scope, registered_by=registered_by,
        )
    except OwnershipConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 -- catch-all for any other Keycloak/validation failure, surfaced as 401 with detail
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
