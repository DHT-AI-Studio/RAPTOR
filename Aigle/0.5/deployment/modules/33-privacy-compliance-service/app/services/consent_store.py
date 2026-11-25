"""GD-5 / VIE01-250: consent state in Keycloak user attributes (via
python-keycloak), the append-only gdpr_consent_history record of every
change, and what a withdrawal sets off.

Keycloak is the source of truth for *current* consent; gdpr_consent_history
is the legal record (GDPR Art. 7(1)) and the source of granted_at.

On withdrawal:
- memory / benchmark / voice_recording / face_detection / voice_clone
  enqueue a `consent_withdrawal` job in gdpr_erasure_jobs -- a partial
  erasure of that one purpose's store (Module 26 / 22 / 34 / 25 / 35),
  never the account itself -- run in the background with the erasure
  orchestrator's retry.
  Each job's final state also writes a CF-4 `gdpr.erasure` audit event
  (V05-10 / VIE01-364).
- voice_recording also sets gdpr:voice_recording_disabled:{user_id} (90d)
  for Module 34 to check at session start; granting deletes it.
- face_detection (MTR-24, GD-11): Module 25 deletes the face layer of the
  user's database (DELETE /internal/gdpr/faces/user/{user_id}) -- faces,
  traces, groups, identities and their undo history; the rest of the
  database stays. People Search (Module 13), face detection (Module 11)
  and seed photos (Module 13) check the purpose per request.
- voice_clone (GD-11): Module 35 deletes the user's speaker embeddings (the
  same DELETE /internal/gdpr/user/{user_id} GD-1 uses -- they are all
  Module 35 holds); /voice/clone checks the purpose per request.
Module 34 deletes the recordings at DELETE /internal/gdpr/user/{user_id}.
With GD_MODULE_34_URL / GD_MODULE_35_URL empty, that store is recorded as
`skipped`.

Keycloak 24+ behaviours this module depends on:
- Unmanaged user attributes are dropped silently unless the realm's
  "Unmanaged attributes" policy allows admin writes (06-authentication/
  README.md) -- every write is read back, and a drop is raised instead of
  being reported as a change that never happened.
- A PUT of a partial user representation clears the profile fields it omits
  (firstName/lastName/email), so updates always PUT the whole user back.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import asyncpg
import httpx
from keycloak import KeycloakAdmin, KeycloakOpenIDConnection
from keycloak.exceptions import KeycloakError
from redis.asyncio import Redis

from app.core.config import settings
from app.models.consent import (
    CONSENT_VERSION_ATTRIBUTE,
    PURPOSES,
    ConsentAction,
    ConsentPurposeId,
    ConsentRecord,
    ConsentResponse,
    ConsentSource,
    ConsentUpdateResponse,
)
from app.models.job import JobStatus, JobType, StoreStatus
from app.services.audit_events import audit_writer, erasure_event, privacy_event
from app.services.erasure_orchestrator import ErasureOrchestrator

logger = logging.getLogger("privacy_compliance.consent")

_INTERNAL_HEADERS = {"X-Internal-Token": settings.internal_token}
VOICE_DISABLED_KEY = "gdpr:voice_recording_disabled:{user_id}"
VOICE_DISABLED_TTL_SECONDS = 90 * 86400  # ticket: TTL 90d
DEFAULTS: dict[ConsentPurposeId, bool] = {p.id: p.default for p in PURPOSES.values()}

# Attribute updates are read-modify-write on the whole Keycloak user, so two
# concurrent changes for one user must not interleave. Striped rather than
# one lock per user, so memory stays bounded.
_LOCK_STRIPES = 64


class UserNotFound(Exception):
    pass


class ConsentStoreError(Exception):
    pass


def hash_ip(x_forwarded_for: str | None) -> str | None:
    """SHA-256 of the raw X-Forwarded-For value -- the IP itself is never stored."""
    if not x_forwarded_for or not x_forwarded_for.strip():
        return None
    return hashlib.sha256(x_forwarded_for.strip().encode("utf-8")).hexdigest()


def parse_consent(user: dict[str, Any]) -> dict[ConsentPurposeId, bool | None]:
    """{purpose: granted} from a Keycloak user; None = attribute never set."""
    attributes = user.get("attributes") or {}
    result: dict[ConsentPurposeId, bool | None] = {}
    for purpose in PURPOSES.values():
        values = attributes.get(purpose.attribute) or []
        result[purpose.id] = values[0].lower() == "true" if values else None
    return result


class ConsentStore:
    def __init__(self, pool: asyncpg.Pool, redis: Redis, orchestrator: ErasureOrchestrator):
        self.pool = pool
        self.redis = redis
        self.orchestrator = orchestrator  # for its per-store retry wrapper
        self.audit = audit_writer(pool)
        self.admin = KeycloakAdmin(connection=KeycloakOpenIDConnection(
            server_url=settings.keycloak_url,
            username=settings.keycloak_admin_username,
            password=settings.keycloak_admin_password,
            realm_name=settings.keycloak_realm,
            user_realm_name="master",
            client_id="admin-cli",
        ))
        self.http = httpx.AsyncClient(timeout=settings.erasure_timeout_seconds)
        self._locks = [asyncio.Lock() for _ in range(_LOCK_STRIPES)]
        # Strong references to running cascade jobs -- asyncio only keeps
        # weak ones, and a GC'd task would silently drop a job.
        self._tasks: set[asyncio.Task] = set()

    async def close(self) -> None:
        await self.http.aclose()

    def _lock(self, user_id: str) -> asyncio.Lock:
        return self._locks[int(hashlib.sha256(user_id.encode()).hexdigest(), 16) % _LOCK_STRIPES]

    # ── Public operations ───────────────────────────────────────────────────

    async def get(self, user_id: str) -> ConsentResponse:
        consent = parse_consent(await self._get_user(user_id))
        if any(v is None for v in consent.values()):
            # Created before GD-5, or its registration webhook call failed.
            # Defaults only -- no gdpr_consent_version, so REVIEW_CONSENT
            # still treats this user as never having seen the notice.
            _, consent = await self._write(user_id, DEFAULTS, ConsentSource.FIRST_READ_DEFAULT, only_if_unset=True)

        granted_at = await self._last_grants(user_id)
        return ConsentResponse(purposes=[
            ConsentRecord(
                id=p.id, granted=bool(consent[p.id]), description=p.description,
                granted_at=granted_at.get(p.id.value) if consent[p.id] else None,
            )
            for p in PURPOSES.values()
        ])

    async def set(
        self, user_id: str, purpose: ConsentPurposeId, granted: bool, ip_hash: str | None,
    ) -> ConsentUpdateResponse:
        """V05-10 / VIE01-364: leaves a CF-4 `gdpr.consent` event (action
        grant / withdraw, whether it changed anything, and the withdrawal's
        erasure job if one was started). The erasure job writes its own
        `gdpr.erasure` event when it finishes."""
        action = (ConsentAction.GRANT if granted else ConsentAction.WITHDRAW).value
        meta: dict[str, Any] = {"purpose": purpose.value, "policy_version": settings.consent_policy_version}
        try:
            changed, _ = await self._write(user_id, {purpose: granted}, ConsentSource.USER, ip_hash=ip_hash)

            await self._sync_voice_flag(user_id, purpose, granted)

            job_id = None
            store = PURPOSES[purpose].withdrawal_store
            if purpose in changed and not granted and store:
                job_id = await self._enqueue_partial_erasure(user_id, purpose, store)
        except UserNotFound:
            raise
        except Exception:
            await self.audit.emit(privacy_event(
                "gdpr.consent", user_id=user_id, action=action, ok=False, requested_by=user_id, metadata=meta))
            raise
        await self.audit.emit(privacy_event(
            "gdpr.consent", user_id=user_id, action=action, ok=True, requested_by=user_id, resource_id=job_id,
            metadata={**meta, "changed": purpose in changed}))
        return ConsentUpdateResponse(id=purpose, granted=granted, changed=purpose in changed, erasure_job_id=job_id)

    async def stop_processing(self, user_id: str, purpose: ConsentPurposeId, ip_hash: str | None) -> None:
        """PRIV-X3 / VIE01-357 objection (GDPR Art. 21): withdraw consent for
        ``purpose`` -- same Keycloak attribute, history row (source
        'objection', with the IP hash) and voice flag as a GD-5 withdrawal --
        but never the partial-erasure cascade: an objection stops future
        processing and leaves past data alone."""
        await self._write(user_id, {purpose: False}, ConsentSource.OBJECTION, ip_hash=ip_hash)
        await self._sync_voice_flag(user_id, purpose, False)

    async def get_user(self, user_id: str) -> dict[str, Any]:
        """PRIV-X3 / VIE01-357: the full Keycloak user representation."""
        return await self._get_user(user_id)

    async def update_user(self, user_id: str, mutate: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        """PRIV-X3 / VIE01-357: apply ``mutate`` to the full Keycloak user
        representation and PUT it back, under the same per-user lock as
        consent writes, so a rectification, a restriction and a consent
        change can't overwrite each other's read-modify-write. Returns the
        user as Keycloak stored it (read back)."""
        async with self._lock(user_id):
            user = await self._get_user(user_id)
            mutate(user)
            await self.admin.a_update_user(user_id, user)
            return await self._get_user(user_id)

    async def email_verification_unavailable(self) -> bool:
        """PRIV-X3 / VIE01-357: True when the realm requires verified emails
        but has no mail server (Realm settings -> Email -> Host is empty).
        A user whose email is changed there must verify it at their next
        login, and the link can never be sent, so they are locked out."""
        realm = await self.admin.a_get_realm(settings.keycloak_realm)
        smtp_host = ((realm.get("smtpServer") or {}).get("host") or "").strip()
        return bool(realm.get("verifyEmail")) and not smtp_host

    async def _sync_voice_flag(self, user_id: str, purpose: ConsentPurposeId, granted: bool) -> None:
        if purpose != ConsentPurposeId.VOICE_RECORDING:
            return
        # Every time, not only on change: re-arms the 90-day TTL and
        # repairs a key lost from Redis.
        key = VOICE_DISABLED_KEY.format(user_id=user_id)
        if granted:
            await self.redis.delete(key)
        else:
            await self.redis.set(key, "1", ex=VOICE_DISABLED_TTL_SECONDS)

    async def set_registration_defaults(self, user_id: str) -> None:
        """Registration webhook. Idempotent: a repeated call (or one for a
        user who already chose) never overwrites an existing value."""
        await self._write(
            user_id, DEFAULTS, ConsentSource.REGISTRATION_DEFAULT,
            only_if_unset=True, stamp_version=True,
        )

    # ── Keycloak + history ──────────────────────────────────────────────────

    async def _get_user(self, user_id: str) -> dict[str, Any]:
        try:
            return await self.admin.a_get_user(user_id)
        except KeycloakError as exc:
            if exc.response_code == 404:
                raise UserNotFound(user_id) from exc
            raise

    async def _write(
        self,
        user_id: str,
        changes: dict[ConsentPurposeId, bool],
        source: ConsentSource,
        *,
        ip_hash: str | None = None,
        only_if_unset: bool = False,
        stamp_version: bool = False,
    ) -> tuple[dict[ConsentPurposeId, bool], dict[ConsentPurposeId, bool | None]]:
        """Apply ``changes``; return ({purpose: granted} actually changed,
        resulting consent). The history rows are inserted in a transaction
        that only commits once Keycloak has accepted *and kept* the new
        attributes, so the legal record never claims a change Keycloak
        didn't make."""
        async with self._lock(user_id):
            user = await self._get_user(user_id)
            before = parse_consent(user)
            attributes = dict(user.get("attributes") or {})

            changed = {
                pid: granted for pid, granted in changes.items()
                if before[pid] != granted and not (only_if_unset and before[pid] is not None)
            }
            new_version = stamp_version and not attributes.get(CONSENT_VERSION_ATTRIBUTE)
            if not changed and not new_version:
                return {}, before

            for pid, granted in changed.items():
                attributes[PURPOSES[pid].attribute] = ["true" if granted else "false"]
            if new_version:
                attributes[CONSENT_VERSION_ATTRIBUTE] = [settings.consent_policy_version]
            user["attributes"] = attributes

            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    await conn.executemany(
                        """
                        INSERT INTO gdpr_consent_history
                            (user_id, purpose, action, ip_hash, policy_version, source)
                        VALUES ($1, $2, $3, $4, $5, $6)
                        """,
                        [
                            (user_id, pid.value,
                             (ConsentAction.GRANT if granted else ConsentAction.WITHDRAW).value,
                             ip_hash, settings.consent_policy_version, source.value)
                            for pid, granted in changed.items()
                        ],
                    )
                    await self.admin.a_update_user(user_id, user)
                    after_user = await self._get_user(user_id)
                    after = parse_consent(after_user)
                    dropped = [PURPOSES[pid].attribute for pid, granted in changed.items() if after[pid] != granted]
                    if new_version and not (after_user.get("attributes") or {}).get(CONSENT_VERSION_ATTRIBUTE):
                        dropped.append(CONSENT_VERSION_ATTRIBUTE)
                    if dropped:
                        raise ConsentStoreError(
                            f"Keycloak did not persist {', '.join(dropped)} -- the realm's 'Unmanaged "
                            "attributes' policy must allow admin writes (see 06-authentication/README.md)"
                        )
            return changed, after

    async def _last_grants(self, user_id: str) -> dict[str, datetime]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT purpose, MAX(timestamp) AS granted_at FROM gdpr_consent_history
                WHERE user_id = $1 AND action = $2
                GROUP BY purpose
                """,
                user_id, ConsentAction.GRANT.value,
            )
        return {r["purpose"]: r["granted_at"] for r in rows}

    # ── Partial erasure on withdrawal ───────────────────────────────────────

    async def _enqueue_partial_erasure(self, user_id: str, purpose: ConsentPurposeId, store: str) -> str:
        job_id = str(uuid.uuid4())
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO gdpr_erasure_jobs
                    (job_id, user_id, requested_at, requested_by, status, store_statuses, type, notes)
                VALUES ($1, $2, $3, $2, $4, $5, $6, $7)
                """,
                job_id, user_id, datetime.now(timezone.utc), JobStatus.PENDING.value,
                json.dumps({store: StoreStatus.PENDING.value}), JobType.CONSENT_WITHDRAWAL.value,
                f"consent withdrawn: {purpose.value}",
            )
        task = asyncio.create_task(self._run_partial_erasure(job_id, user_id, store, purpose))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job_id

    async def _run_partial_erasure(self, job_id: str, user_id: str, store: str, purpose: ConsentPurposeId) -> None:
        await self._update_job(job_id, JobStatus.IN_PROGRESS, {store: StoreStatus.IN_PROGRESS})
        if (store == "module_34" and not settings.module_34_url) or (store == "module_35" and not settings.module_35_url):
            # GD_MODULE_34_URL / GD_MODULE_35_URL empty: that module isn't run here.
            await self._finish_partial_erasure(job_id, user_id, purpose, JobStatus.COMPLETED, {store: StoreStatus.SKIPPED})
            return
        url = {
            "module_26": f"{settings.module_26_url}/internal/gdpr/memory/user/{user_id}",
            "module_22": f"{settings.module_22_url}/internal/gdpr/benchmark/user/{user_id}",
            "module_34": f"{settings.module_34_url}/internal/gdpr/user/{user_id}",
            "module_25": f"{settings.module_25_url}/internal/gdpr/faces/user/{user_id}",
            "module_35": f"{settings.module_35_url}/internal/gdpr/user/{user_id}",
        }[store]

        async def delete(_: str) -> None:
            resp = await self.http.delete(url, headers=_INTERNAL_HEADERS)
            resp.raise_for_status()

        try:
            await self.orchestrator._with_retry(delete, user_id)
        except Exception as exc:  # noqa: BLE001 -- partial-erasure delete can fail for any reason; marked FAILED and logged, consent withdrawal itself still recorded
            logger.error("consent job=%s store=%s: all attempts failed: %s", job_id, store, exc)
            await self._finish_partial_erasure(job_id, user_id, purpose, JobStatus.PARTIAL_FAILURE, {store: StoreStatus.FAILED})
            return
        await self._finish_partial_erasure(job_id, user_id, purpose, JobStatus.COMPLETED, {store: StoreStatus.DELETED})

    async def _finish_partial_erasure(
        self, job_id: str, user_id: str, purpose: ConsentPurposeId, status: JobStatus,
        store_statuses: dict[str, StoreStatus],
    ) -> None:
        """V05-10 / VIE01-364: the job's final state plus its CF-4
        `gdpr.erasure` audit event (fail-open: the withdrawal itself is
        already on record in gdpr_consent_history)."""
        await self._update_job(job_id, status, store_statuses)
        await self.orchestrator.audit.emit(erasure_event(
            job_id=job_id, user_id=user_id, requested_by=user_id,
            job_type=JobType.CONSENT_WITHDRAWAL.value,
            outcome="erase" if status == JobStatus.COMPLETED else "error",
            store_statuses=store_statuses, purpose=purpose.value,
        ))

    async def _update_job(self, job_id: str, status: JobStatus, store_statuses: dict[str, StoreStatus]) -> None:
        completed_at = datetime.now(timezone.utc) if status == JobStatus.COMPLETED else None
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE gdpr_erasure_jobs SET status = $1, store_statuses = $2, completed_at = $3 WHERE job_id = $4",
                status.value, json.dumps({k: v.value for k, v in store_statuses.items()}), completed_at, job_id,
            )
