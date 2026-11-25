"""GD-5 / VIE01-250: internal consent endpoints.

GET/PUT /consent: identity comes from the X-User-ID Module 13 injects after
its own JWT check -- same trust pattern as erasure.py/export.py/access.py.
X-Forwarded-For is the end user's, passed through by Module 13; only its
SHA-256 is ever stored.

POST /internal/gdpr/new-user: the registration webhook, called by Module 06
(keycloak-api) right after it creates a user.

Never reachable from outside the raptor network (X-Internal-Token).
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Header, HTTPException, Request
from keycloak.exceptions import KeycloakError

from app.models.consent import (
    ConsentPurposeId,
    ConsentResponse,
    ConsentUpdateRequest,
    ConsentUpdateResponse,
    NewUserRequest,
)
from app.services.consent_store import (
    ConsentStore,
    ConsentStoreError,
    UserNotFound,
    hash_ip,
)

logger = logging.getLogger("privacy_compliance.consent_router")
router = APIRouter(tags=["consent"])


def _require_user(x_user_id: str | None) -> str:
    if not x_user_id:
        raise HTTPException(status_code=400, detail="Missing X-User-ID header")
    return x_user_id


def _store(request: Request) -> ConsentStore:
    return request.app.state.consent_store


async def _call(coro):
    try:
        return await coro
    except UserNotFound:
        raise HTTPException(status_code=404, detail="User not found in Keycloak")
    except ConsentStoreError as exc:
        logger.error("consent write not persisted: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    except KeycloakError as exc:
        logger.error("Keycloak call failed: %s", exc)
        raise HTTPException(status_code=502, detail="Keycloak request failed")


@router.get("/consent", response_model=ConsentResponse)
async def get_consent(request: Request, x_user_id: str = Header(None, alias="X-User-ID")):
    return await _call(_store(request).get(_require_user(x_user_id)))


@router.put("/consent/{purpose}", response_model=ConsentUpdateResponse)
async def put_consent(
    purpose: ConsentPurposeId,
    body: ConsentUpdateRequest,
    request: Request,
    x_user_id: str = Header(None, alias="X-User-ID"),
    x_forwarded_for: str = Header(None, alias="X-Forwarded-For"),
):
    user_id = _require_user(x_user_id)
    return await _call(_store(request).set(user_id, purpose, body.granted, hash_ip(x_forwarded_for)))


@router.post("/internal/gdpr/new-user", status_code=204, response_model=None)
async def new_user(body: NewUserRequest, request: Request) -> None:
    await _call(_store(request).set_registration_defaults(body.user_id))
