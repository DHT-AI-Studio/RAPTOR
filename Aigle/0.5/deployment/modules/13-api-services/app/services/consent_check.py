"""MTR-24: deny a request unless the caller granted a consent purpose (GD-5).

Consent lives in Keycloak user attributes that only Module 33 reads and
writes, so it is read through Module 33's GET /consent, once per request.
A lookup that fails denies (503), the same as the processing-restriction
check -- a biometric feature never runs on an unknown consent state.
"""
from __future__ import annotations

import logging

import httpx
from fastapi import Depends, HTTPException

from app.api.dependencies import get_http_client
from app.core.config import Settings, get_settings

_logger = logging.getLogger(__name__)
_TIMEOUT = 5.0


class ConsentCheck:
    def __init__(self, client: httpx.AsyncClient, settings: Settings):
        self.client = client
        self.settings = settings

    async def require(self, user_id: str, purpose: str) -> None:
        """Raises 403 PERMISSION_DENIED unless `purpose` is granted, 503 if it can't be read."""
        try:
            resp = await self.client.get(
                f"{self.settings.privacy_compliance_url}/consent",
                headers={"X-Internal-Token": self.settings.gdpr_internal_token, "X-User-ID": user_id},
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            granted = any(p["id"] == purpose and p["granted"] is True for p in resp.json()["purposes"])
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            _logger.error("Consent check (%s) failed for user=%s -- denying: %s", purpose, user_id, exc)
            raise HTTPException(status_code=503, detail={"error": "CONSENT_CHECK_UNAVAILABLE",
                                                         "message": "consent could not be checked"}) from exc
        if not granted:
            raise HTTPException(status_code=403, detail={"error": "PERMISSION_DENIED",
                                                         "message": f"{purpose} consent not granted"})


def get_consent_check(client: httpx.AsyncClient = Depends(get_http_client),
                      settings: Settings = Depends(get_settings)) -> ConsentCheck:
    return ConsentCheck(client, settings)
