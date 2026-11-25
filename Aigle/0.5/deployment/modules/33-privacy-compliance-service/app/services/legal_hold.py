"""PRIV-X4 / VIE01-358: legal holds on erasure (GDPR Art. 17(3)).

gdpr_legal_holds is the legal record and what the erasure flow checks. The
Keycloak attributes gdpr_legal_hold / gdpr_hold_until mirror the hold on the
account, as the ticket specifies (like PRIV-X3's gdpr_processing_restricted).
A hold is active while released_at is NULL and hold_until, a UTC date, has
not been reached.

POST /erasure for a user under an active hold records a DEFERRED job instead
of erasing; ErasureOrchestrator.run() checks again before deleting anything,
so a job created just before a hold was placed is deferred too. The daily
expiry job releases holds whose hold_until has come, then sets every deferred
job whose user has no active hold back to pending and runs it.

V05-10 / VIE01-364: placing a hold (by the gdpr-admin in X-Requested-By) and
each release by the daily job leave a CF-4 `gdpr.legal_hold` event with the
hold id and hold_until -- never the reason or authority, which are free text.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date
from typing import Any

import asyncpg

from app.models.job import JobStatus
from app.models.legal_hold import LegalHold, LegalHoldRequest
from app.services.audit_events import audit_writer, privacy_event
from app.services.consent_store import ConsentStore, ConsentStoreError, UserNotFound
from app.services.erasure_orchestrator import ErasureOrchestrator
from app.services.legal_hold_state import ACTIVE_CONDITION, lock_user, today_utc

logger = logging.getLogger("privacy_compliance.legal_hold")

HOLD_ATTRIBUTE = "gdpr_legal_hold"
HOLD_UNTIL_ATTRIBUTE = "gdpr_hold_until"


class LegalHoldError(Exception):
    """A request the rules reject -- carries the HTTP status to answer with."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class LegalHoldService:
    def __init__(
        self, pool: asyncpg.Pool, consent_store: ConsentStore, orchestrator: ErasureOrchestrator,
        start_erasure: Callable[[str], None],
    ) -> None:
        self.pool = pool
        self.consent_store = consent_store
        self.orchestrator = orchestrator
        self.start_erasure = start_erasure  # runs one erasure job in the background
        self.audit = audit_writer(pool)

    async def place(self, request: LegalHoldRequest, placed_by: str) -> LegalHold:
        """The row commits only once Keycloak has accepted and kept the
        attributes. The erasure check runs under the lock an erasure takes to
        start (lock_user), so a running erasure is seen, never missed."""
        user_id = request.user_id
        if request.hold_until <= today_utc():
            raise LegalHoldError(422, "hold_until must be a future date")
        try:
            hold = await self._place(request)
        except (LegalHoldError, UserNotFound):
            raise
        except Exception:
            await self.audit.emit(privacy_event(
                "gdpr.legal_hold", user_id=user_id, action="place", ok=False, requested_by=placed_by,
                metadata={"hold_until": request.hold_until.isoformat()}))
            raise
        await self.audit.emit(privacy_event(
            "gdpr.legal_hold", user_id=user_id, action="place", ok=True, requested_by=placed_by,
            resource_id=str(hold.id), metadata={"hold_until": hold.hold_until.isoformat()}))
        return hold

    async def _place(self, request: LegalHoldRequest) -> LegalHold:
        user_id = request.user_id
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await lock_user(conn, user_id)
                latest = await self.orchestrator.find_active_or_completed_job(user_id)
                if latest is not None and latest.status == JobStatus.COMPLETED:
                    raise LegalHoldError(409, "This user has already been erased")
                if latest is not None and latest.status == JobStatus.IN_PROGRESS:
                    raise LegalHoldError(409, "An erasure is already running for this user; a hold cannot stop it")
                try:
                    row = await conn.fetchrow(
                        """
                        INSERT INTO gdpr_legal_holds (user_id, reason, authority, hold_until)
                        VALUES ($1, $2, $3, $4) RETURNING *
                        """,
                        user_id, request.reason, request.authority, request.hold_until,
                    )
                except asyncpg.UniqueViolationError:
                    existing = await self.pool.fetchval(
                        "SELECT hold_until FROM gdpr_legal_holds WHERE user_id = $1 AND released_at IS NULL", user_id,
                    )
                    raise LegalHoldError(409, f"This user already has an open legal hold (until {existing})") from None
                await self._set_attributes(user_id, request.hold_until)
        logger.info("legal hold placed for user=%s until %s", user_id, request.hold_until)
        return LegalHold(**dict(row))

    async def list_active(self) -> list[LegalHold]:
        rows = await self.pool.fetch(
            f"SELECT * FROM gdpr_legal_holds WHERE {ACTIVE_CONDITION} ORDER BY hold_until, id",  # nosec B608 -- ACTIVE_CONDITION is a constant
            today_utc(),
        )
        return [LegalHold(**dict(r)) for r in rows]

    async def release_expired(self) -> None:
        """The daily job. Each step stands on its own, so a run interrupted
        between them is completed by the next one."""
        today = today_utc()
        released = await self.pool.fetch(
            """
            UPDATE gdpr_legal_holds SET released_at = now()
            WHERE released_at IS NULL AND hold_until <= $1 RETURNING id, user_id, hold_until
            """,
            today,
        )
        for row in released:
            await self.audit.emit(privacy_event(
                "gdpr.legal_hold", user_id=row["user_id"], action="release", ok=True, requested_by=None,
                resource_id=str(row["id"]), metadata={"hold_until": row["hold_until"].isoformat(), "reason": "expired"}))
            try:
                await self._set_attributes(row["user_id"], None)
            except UserNotFound:
                pass  # the account is gone already
            except Exception as exc:  # noqa: BLE001 -- Keycloak/HTTP errors of any kind; the table, not the attribute, is what erasure checks, already logged
                logger.error("legal hold released for user=%s, but its Keycloak attributes remain: %s", row["user_id"], exc)
        queued = await self.pool.fetch(
            """
            UPDATE gdpr_erasure_jobs j SET status = $1
            WHERE j.type = 'erasure' AND j.status = $2 AND NOT EXISTS (
                SELECT 1 FROM gdpr_legal_holds h
                WHERE h.user_id = j.user_id AND h.released_at IS NULL AND h.hold_until > $3
            )
            RETURNING j.job_id
            """,
            JobStatus.PENDING.value, JobStatus.DEFERRED.value, today,
        )
        for row in queued:
            self.start_erasure(str(row["job_id"]))
        if released or queued:
            logger.info("legal hold expiry: %d hold(s) released, %d deferred erasure(s) started", len(released), len(queued))

    async def _set_attributes(self, user_id: str, hold_until: date | None) -> None:
        def mutate(user: dict[str, Any]) -> None:
            attributes = dict(user.get("attributes") or {})
            if hold_until is not None:
                attributes[HOLD_ATTRIBUTE] = ["true"]
                attributes[HOLD_UNTIL_ATTRIBUTE] = [hold_until.isoformat()]
            else:
                attributes.pop(HOLD_ATTRIBUTE, None)
                attributes.pop(HOLD_UNTIL_ATTRIBUTE, None)
            user["attributes"] = attributes

        stored = (await self.consent_store.update_user(user_id, mutate)).get("attributes") or {}
        expected = (["true"], [hold_until.isoformat()]) if hold_until is not None else (None, None)
        if (stored.get(HOLD_ATTRIBUTE), stored.get(HOLD_UNTIL_ATTRIBUTE)) != expected:
            raise ConsentStoreError(
                f"Keycloak did not persist {HOLD_ATTRIBUTE} / {HOLD_UNTIL_ATTRIBUTE} -- the realm's 'Unmanaged "
                "attributes' policy must allow admin writes (see 06-authentication/README.md)"
            )
