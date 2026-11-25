"""PRIV-X4 / VIE01-358: internal admin endpoints -- legal holds (GDPR Art. 17(3)).

Module 13 enforces gdpr-admin and passes the admin as X-Requested-By, the
same trust pattern as rights.py's restriction lift. Never reachable from
outside the raptor network (X-Internal-Token, on the router).
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Header, HTTPException, Request
from keycloak.exceptions import KeycloakError

from app.models.legal_hold import LegalHold, LegalHoldList, LegalHoldRequest
from app.services.consent_store import ConsentStoreError, UserNotFound
from app.services.legal_hold import LegalHoldError, LegalHoldService

logger = logging.getLogger("privacy_compliance.admin_router")
router = APIRouter(prefix="/admin", tags=["legal-hold"])


def _service(request: Request) -> LegalHoldService:
    return request.app.state.legal_hold_service


@router.post("/legal-hold", response_model=LegalHold)
async def place_legal_hold(
    body: LegalHoldRequest,
    request: Request,
    x_requested_by: str = Header(None, alias="X-Requested-By"),
):
    if not x_requested_by:
        raise HTTPException(status_code=400, detail="Missing X-Requested-By header")
    try:
        hold = await _service(request).place(body, x_requested_by)
    except LegalHoldError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)
    except UserNotFound:
        raise HTTPException(status_code=404, detail="User not found in Keycloak")
    except ConsentStoreError as exc:
        logger.error("legal hold not persisted: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    except KeycloakError as exc:
        logger.error("Keycloak call failed: %s", exc)
        raise HTTPException(status_code=502, detail="Keycloak request failed")
    logger.info("legal hold id=%s placed by %s", hold.id, x_requested_by)
    return hold


@router.get("/legal-holds", response_model=LegalHoldList)
async def list_legal_holds(request: Request):
    """Every hold in force now, the soonest to end first."""
    return LegalHoldList(holds=await _service(request).list_active())
