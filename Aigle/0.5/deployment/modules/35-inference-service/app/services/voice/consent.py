"""GD-11: /voice/clone only for a caller who granted voice_clone consent.

A cloned voice is built from a speaker embedding -- GDPR Art. 9 biometric
data -- so the purpose defaults to withdrawn and needs an explicit opt-in.
Consent lives in Keycloak user attributes that only Module 33 reads, so it is
read through Module 33's GET /consent on every clone request. A lookup that
fails denies (503), as Module 13's People Search and Module 34's recording
do: a biometric feature never runs on an unknown consent state.

Withdrawing the purpose also deletes the caller's speaker embeddings: Module 33
calls this module's DELETE /internal/gdpr/user/{user_id} (routers/gdpr.py).
"""
from __future__ import annotations

import logging

import httpx

from app.config import Settings
from app.errors import ServiceError

logger = logging.getLogger(__name__)

PURPOSE = "voice_clone"
_TIMEOUT_S = 5.0


async def require_voice_clone_consent(http: httpx.AsyncClient, settings: Settings, user_id: str) -> None:
    """Raises 403 consent_required unless the caller granted voice_clone, 503
    consent_check_unavailable when Module 33 can't tell."""
    try:
        resp = await http.get(
            f"{settings.module33_url}/consent",
            headers={"X-User-ID": user_id, "X-Internal-Token": settings.gdpr_internal_token},
            timeout=_TIMEOUT_S,
        )
        if resp.status_code == 404:          # not a Keycloak user: never consented
            granted = False
        else:
            resp.raise_for_status()
            granted = any(p["id"] == PURPOSE and p["granted"] is True for p in resp.json()["purposes"])
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        logger.error("voice_clone consent check failed for user=%s -- denying: %s", user_id, exc)
        raise ServiceError(503, "consent_check_unavailable", purpose=PURPOSE) from exc
    if not granted:
        raise ServiceError(403, "consent_required", purpose=PURPOSE)
