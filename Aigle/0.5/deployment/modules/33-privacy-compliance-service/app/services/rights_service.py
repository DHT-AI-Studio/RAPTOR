"""PRIV-X3 / VIE01-357: rectification (GDPR Art. 16), restriction of
processing (Art. 18) and objection to processing (Art. 21).

Rectification: one Keycloak profile field (email, firstName, lastName) per
request, read back to confirm Keycloak kept it. Keycloak is the only store
of a user's profile -- see app/models/rights.py on the ticket's "ArcadeDB
user vertex", which doesn't exist.

Restriction: Keycloak attribute gdpr_processing_restricted=true is the
state Module 13's middleware enforces (via is_restricted(), since Module 13
has no Keycloak admin access), gdpr_restriction_log the legal record (one
open row per user). Only a gdpr-admin lifts a restriction (Module 13
enforces the role).

Objection: GD-5's withdrawal without its deletion cascade
(ConsentStore.stop_processing), plus a gdpr_objection_log row.

V05-10 / VIE01-364: each carried-out or failed request also leaves a CF-4
audit event (gdpr.rectification / gdpr.restriction / gdpr.objection, see
app/services/audit_events.py). A request rejected by the rules (RightsError)
changed nothing and leaves none.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import asyncpg

from app.models.consent import ConsentPurposeId
from app.models.rights import (
    RECTIFIABLE_FIELDS,
    UNSUPPORTED_OBJECTION_PURPOSES,
    ObjectResponse,
    RectifiableField,
    RectifyResponse,
    RestrictionStatus,
)
from app.services.audit_events import PrivacyEventType, audit_writer, privacy_event
from app.services.consent_store import ConsentStore, ConsentStoreError

logger = logging.getLogger("privacy_compliance.rights")

RESTRICTED_ATTRIBUTE = "gdpr_processing_restricted"


class RightsError(Exception):
    """A request the rules reject -- carries the HTTP status to answer with."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class RightsService:
    def __init__(self, pool: asyncpg.Pool, consent_store: ConsentStore) -> None:
        self.pool = pool
        self.consent_store = consent_store
        self.audit = audit_writer(pool)

    async def _audit(self, event_type: PrivacyEventType, user_id: str, action: str, ok: bool, **kw: Any) -> None:
        await self.audit.emit(privacy_event(event_type, user_id=user_id, action=action, ok=ok, **kw))

    # ── Rectification (Art. 16) ─────────────────────────────────────────────

    async def rectify(self, user_id: str, field: RectifiableField, value: str) -> RectifyResponse:
        key = RECTIFIABLE_FIELDS[field.value]
        value = value.strip()
        if field == RectifiableField.EMAIL and not _looks_like_email(value):
            raise RightsError(422, "value is not a valid email address")
        if field == RectifiableField.EMAIL:
            await self._check_email_can_be_verified()

        def mutate(user: dict[str, Any]) -> None:
            user[key] = value
            if field == RectifiableField.EMAIL:
                user["emailVerified"] = False  # a changed address hasn't been verified

        audit = {"requested_by": user_id, "metadata": {"field": field.value}}  # the field, never the value
        try:
            stored = await self.consent_store.update_user(user_id, mutate)
            # Keycloak stores email lower-cased; names exactly as given.
            same = str.casefold if field == RectifiableField.EMAIL else str
            if same(stored.get(key) or "") != same(value):
                raise ConsentStoreError(f"Keycloak did not persist {key}")
        except Exception:
            await self._audit("gdpr.rectification", user_id, "rectify", False, **audit)
            raise
        await self._audit("gdpr.rectification", user_id, "rectify", True, **audit)
        return RectifyResponse(fields_updated=[f"keycloak.{key}"], propagation_status="complete")

    async def _check_email_can_be_verified(self) -> None:
        """A changed email is saved unverified, and Keycloak then makes the user
        verify it at their next login. With no mail server the link never
        arrives and the user is locked out, so refuse the change instead; it
        works again as soon as a mail server is set in the realm. If the realm
        settings can't be read, refuse too rather than risk the lockout."""
        try:
            unavailable = await self.consent_store.email_verification_unavailable()
        except Exception as exc:  # Keycloak down / no permission: don't guess
            logger.error("could not read the realm's email settings: %s", exc)
            raise RightsError(503, "Could not check whether email verification is available; "
                                   "try again later") from exc
        if unavailable:
            raise RightsError(409, "Email changes need email verification, which is not set up on this "
                                   "server (no mail server in Keycloak); ask an administrator to change "
                                   "the email")

    # ── Restriction (Art. 18) ───────────────────────────────────────────────

    async def is_restricted(self, user_id: str) -> bool:
        """The Keycloak attribute itself -- what Module 13's middleware asks
        about on every write request."""
        user = await self.consent_store.get_user(user_id)
        return (user.get("attributes") or {}).get(RESTRICTED_ATTRIBUTE) == ["true"]

    async def status(self, user_id: str) -> RestrictionStatus:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM gdpr_restriction_log WHERE user_id = $1 ORDER BY restricted_at DESC LIMIT 1",
                user_id,
            )
        return _status(user_id, row)

    async def restrict(self, user_id: str, reason: str | None) -> RestrictionStatus:
        """Idempotent: restricting an already-restricted user returns the
        restriction in force. The log row commits only once Keycloak has
        accepted and kept the attribute."""
        try:
            current = await self.status(user_id)
            if current.restricted:
                await self._set_attribute(user_id, True)  # repairs an attribute removed by hand
                await self._audit("gdpr.restriction", user_id, "restrict", True, requested_by=user_id,
                                  metadata={"already_restricted": True})
                return current
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    try:
                        row = await conn.fetchrow(
                            """
                            INSERT INTO gdpr_restriction_log (user_id, reason)
                            VALUES ($1, $2) RETURNING *
                            """,
                            user_id, reason,
                        )
                    except asyncpg.UniqueViolationError:
                        # A concurrent request restricted the user first.
                        return await self.status(user_id)
                    await self._set_attribute(user_id, True)
        except Exception:
            await self._audit("gdpr.restriction", user_id, "restrict", False, requested_by=user_id)
            raise
        logger.info("restricted processing for user=%s", user_id)
        await self._audit("gdpr.restriction", user_id, "restrict", True, requested_by=user_id)
        return _status(user_id, row)

    async def lift(self, user_id: str, lifted_by: str) -> RestrictionStatus:
        """Who lifted it is in the GD-7 access log (Module 13 records the
        admin's call); the ticket's log row only gets its lifted_at."""
        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    row = await conn.fetchrow(
                        """
                        UPDATE gdpr_restriction_log SET lifted_at = now()
                        WHERE user_id = $1 AND lifted_at IS NULL RETURNING *
                        """,
                        user_id,
                    )
                    if row is None:
                        raise RightsError(404, "Processing is not restricted for this user")
                    await self._set_attribute(user_id, False)
        except RightsError:
            raise
        except Exception:
            await self._audit("gdpr.restriction", user_id, "lift", False, requested_by=lifted_by)
            raise
        logger.info("lifted processing restriction for user=%s (by %s)", user_id, lifted_by)
        await self._audit("gdpr.restriction", user_id, "lift", True, requested_by=lifted_by)
        return _status(user_id, row)

    async def _set_attribute(self, user_id: str, restricted: bool) -> None:
        def mutate(user: dict[str, Any]) -> None:
            attributes = dict(user.get("attributes") or {})
            if restricted:
                attributes[RESTRICTED_ATTRIBUTE] = ["true"]
            else:
                attributes.pop(RESTRICTED_ATTRIBUTE, None)
            user["attributes"] = attributes

        stored = await self.consent_store.update_user(user_id, mutate)
        kept = (stored.get("attributes") or {}).get(RESTRICTED_ATTRIBUTE) == ["true"]
        if kept != restricted:
            raise ConsentStoreError(
                f"Keycloak did not persist {RESTRICTED_ATTRIBUTE} -- the realm's 'Unmanaged attributes' "
                "policy must allow admin writes (see 06-authentication/README.md)"
            )

    # ── Objection (Art. 21) ─────────────────────────────────────────────────

    async def object_to(self, user_id: str, processing_purpose: str, ip_hash: str | None) -> ObjectResponse:
        if processing_purpose in UNSUPPORTED_OBJECTION_PURPOSES:
            raise RightsError(
                422, f"Objection to '{processing_purpose}' isn't supported yet: "
                     f"{UNSUPPORTED_OBJECTION_PURPOSES[processing_purpose]}",
            )
        try:
            purpose = ConsentPurposeId(processing_purpose)
        except ValueError:
            allowed = ", ".join(p.value for p in ConsentPurposeId)
            raise RightsError(422, f"Unknown processing_purpose '{processing_purpose}' (one of: {allowed})") from None
        audit = {"requested_by": user_id, "metadata": {"purpose": purpose.value}}
        try:
            await self.consent_store.stop_processing(user_id, purpose, ip_hash)
            async with self.pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO gdpr_objection_log (user_id, processing_purpose) VALUES ($1, $2)",
                    user_id, purpose.value,
                )
        except Exception:
            await self._audit("gdpr.objection", user_id, "object", False, **audit)
            raise
        await self._audit("gdpr.objection", user_id, "object", True, **audit)
        return ObjectResponse(purposes_stopped=[purpose], data_deleted=False)


def _looks_like_email(value: str) -> bool:
    local, _, domain = value.partition("@")
    return bool(local) and "." in domain and " " not in value and not domain.startswith(".")


def _status(user_id: str, row: Any) -> RestrictionStatus:
    if row is None:
        return RestrictionStatus(user_id=user_id, restricted=False)
    lifted_at: datetime | None = row["lifted_at"]
    return RestrictionStatus(
        user_id=user_id, restricted=lifted_at is None,
        restricted_at=row["restricted_at"], lifted_at=lifted_at, reason=row["reason"],
    )
