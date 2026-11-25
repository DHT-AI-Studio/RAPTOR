"""PRIV-X3 / VIE01-357: internal endpoints for rectification (GDPR Art. 16),
restriction of processing (Art. 18) and objection to processing (Art. 21).

Identity comes from the X-User-ID Module 13 injects after its own JWT check
(X-Requested-By: the admin, for the admin-only lift) -- same trust pattern
as consent.py. X-Forwarded-For is the end user's, passed through by Module
13; only its SHA-256 is stored. Never reachable from outside the raptor
network (X-Internal-Token, on the router).
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Header, HTTPException, Request
from keycloak.exceptions import KeycloakError

from app.models.rights import (
    LiftRestrictionRequest,
    ObjectRequest,
    ObjectResponse,
    RectifyRequest,
    RectifyResponse,
    RestrictedResponse,
    RestrictionStatus,
    RestrictRequest,
)
from app.services.consent_store import ConsentStoreError, UserNotFound, hash_ip
from app.services.rights_service import RightsError, RightsService

logger = logging.getLogger("privacy_compliance.rights_router")
router = APIRouter(tags=["rights"])


def _require_user(x_user_id: str | None) -> str:
    if not x_user_id:
        raise HTTPException(status_code=400, detail="Missing X-User-ID header")
    return x_user_id


def _service(request: Request) -> RightsService:
    return request.app.state.rights_service


async def _call(coro):
    try:
        return await coro
    except RightsError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)
    except UserNotFound:
        raise HTTPException(status_code=404, detail="User not found in Keycloak")
    except ConsentStoreError as exc:
        logger.error("rights write not persisted: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    except KeycloakError as exc:
        if exc.response_code == 409:  # e.g. the new email belongs to another account
            raise HTTPException(status_code=409, detail="Keycloak rejected the change: value already in use")
        logger.error("Keycloak call failed: %s", exc)
        raise HTTPException(status_code=502, detail="Keycloak request failed")


@router.post("/rectify", response_model=RectifyResponse)
async def rectify(body: RectifyRequest, request: Request, x_user_id: str = Header(None, alias="X-User-ID")):
    """One Keycloak profile field per request: email, firstName or lastName.

    A changed email is saved unverified, so the user must verify it at their
    next login. When the realm requires that but has no mail server (Realm
    settings -> Email -> Host empty) the link could never arrive and the user
    would be locked out, so an email change is refused with 409 until one is
    set (503 if the realm settings can't be read). Names are never affected."""
    return await _call(_service(request).rectify(_require_user(x_user_id), body.field, body.value))


@router.post("/restrict", response_model=RestrictionStatus)
async def restrict(
    request: Request,
    body: RestrictRequest | None = None,
    x_user_id: str = Header(None, alias="X-User-ID"),
):
    user_id = _require_user(x_user_id)
    return await _call(_service(request).restrict(user_id, body.reason if body else None))


@router.get("/restrict/status", response_model=RestrictedResponse)
async def restriction_status(request: Request, x_user_id: str = Header(None, alias="X-User-ID")):
    """Module 13's middleware asks this before every write request."""
    restricted = await _call(_service(request).is_restricted(_require_user(x_user_id)))
    return RestrictedResponse(restricted=restricted)


@router.post("/restrict/lift", response_model=RestrictionStatus)
async def lift_restriction(
    body: LiftRestrictionRequest,
    request: Request,
    x_requested_by: str = Header(None, alias="X-Requested-By"),
):
    if not x_requested_by:
        raise HTTPException(status_code=400, detail="Missing X-Requested-By header")
    return await _call(_service(request).lift(body.user_id, x_requested_by))


@router.post("/object", response_model=ObjectResponse)
async def object_to_processing(
    body: ObjectRequest,
    request: Request,
    x_user_id: str = Header(None, alias="X-User-ID"),
    x_forwarded_for: str = Header(None, alias="X-Forwarded-For"),
):
    user_id = _require_user(x_user_id)
    return await _call(_service(request).object_to(user_id, body.processing_purpose, hash_ip(x_forwarded_for)))
