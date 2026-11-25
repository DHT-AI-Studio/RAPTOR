"""GD-1 / VIE01-246: the erasure state machine.

Sequence (per the ticket): create job -> disable Keycloak account -> mark
the user in Redis so Module 25's Kafka consumer stops indexing new content
for them -> fan out deletion to all stores in parallel -> publish a
tombstone -> once every store reports deleted, hard-delete the Keycloak
account and mark COMPLETED. Any store still failed after its retries
leaves the job PARTIAL_FAILURE and emails the DPO.

V05-10 / VIE01-364: every finish (COMPLETED or PARTIAL_FAILURE, first run
or retry) writes a CF-4 `gdpr.erasure` audit event (app/services/
audit_events.py). The COMPLETED one is fail-closed, as CF-4 specifies for
erasure confirmation: if it can't be written the job stays PARTIAL_FAILURE
and a retry writes it, so no job reads COMPLETED without its audit event.
"""
from __future__ import annotations

import asyncio
import json
import logging
import smtplib
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from typing import Any

import asyncpg
import httpx
from aiokafka import AIOKafkaProducer
from raptor_audit import AuditWriteError
from redis.asyncio import Redis

from app.core.config import settings
from app.models.job import (
    ErasureJob,
    JobStatus,
    JobType,
    StoreStatus,
    initial_store_statuses,
)
from app.models.legal_hold import LegalHold
from app.services.audit_events import audit_writer, erasure_event
from app.services.legal_hold_state import active_hold, lock_user

logger = logging.getLogger("privacy_compliance.erasure")

_INTERNAL_HEADERS = {"X-Internal-Token": settings.internal_token}
_ERASED_USERS_SET = "gdpr:erased_users"
# A skipped store had nothing to delete, so it doesn't hold the job back.
_DONE = (StoreStatus.DELETED, StoreStatus.SKIPPED)
# PRIV-X4: what an erasure leaves about the user in this service's database
# (none of these is in the fan-out), and why each is kept.
_RETAINED: tuple[tuple[str, str], ...] = (
    ("gdpr_erasure_jobs", "Legal record: proof the erasure request was handled (GDPR Art. 5(2))"),
    ("gdpr_consent_history", "Legal record: proof of consent (GDPR Art. 7(1))"),
    ("gdpr_access_log", "Accountability: record of access to personal data (GDPR Art. 5(2), Art. 30)"),
    ("gdpr_restriction_log", "Legal record of restrictions of processing (GDPR Art. 18)"),
    ("gdpr_objection_log", "Legal record of objections to processing (GDPR Art. 21)"),
    ("gdpr_legal_holds", "Legal record of legal holds (GDPR Art. 17(3))"),
    ("audit_events", "Proof of erasure (CF-4 audit event); user ids stored only as SHA-256 hashes"),
)


class ErasureOrchestrator:
    def __init__(self, pool: asyncpg.Pool, redis: Redis, kafka_producer: AIOKafkaProducer | None):
        self.pool = pool
        self.redis = redis
        self.kafka_producer = kafka_producer
        self.http = httpx.AsyncClient(timeout=settings.erasure_timeout_seconds)
        self.audit = audit_writer(pool)

    async def close(self) -> None:
        await self.http.aclose()

    # ── Job lookup / creation ──────────────────────────────────────────────

    async def find_active_or_completed_job(self, user_id: str) -> ErasureJob | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM gdpr_erasure_jobs
                WHERE user_id = $1 AND type = 'erasure'
                ORDER BY requested_at DESC LIMIT 1
                """,
                user_id,
            )
        return _row_to_job(row) if row else None

    async def create_job(self, user_id: str, requested_by: str, status: JobStatus = JobStatus.PENDING) -> ErasureJob:
        """PRIV-X4: status DEFERRED records a request made under a legal
        hold -- no completion estimate until the hold ends."""
        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        eta = now + timedelta(days=30) if status == JobStatus.PENDING else None
        store_statuses = initial_store_statuses()
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO gdpr_erasure_jobs
                    (job_id, user_id, requested_at, requested_by, status, store_statuses, completed_at, notes)
                VALUES ($1, $2, $3, $4, $5, $6, NULL, NULL)
                """,
                job_id, user_id, now, requested_by, status.value,
                json.dumps({k: v.value for k, v in store_statuses.items()}),
            )
        return ErasureJob(
            job_id=job_id, user_id=user_id, requested_at=now, requested_by=requested_by,
            status=status, store_statuses=store_statuses,
            estimated_completion_at=eta,
        )

    async def retained_records(self) -> list[str]:
        """PRIV-X4: everything an erasure leaves about the user, each kept for
        a legal reason (_RETAINED), as "<data>_<retention>: <basis>" -- the
        ticket's "audit_logs_7yr" with its basis cited. Retention comes from
        the GD-3 policy row where there is one; a row with auto_delete=false
        is never purged, so its period is the policy's intent only, and the
        entry says so."""
        try:
            rows = await self.pool.fetch(
                "SELECT data_category, retention_days, auto_delete FROM gdpr_retention_policies "
                "WHERE data_category = ANY($1)",
                [data for data, _ in _RETAINED],
            )
        except asyncpg.UndefinedTableError:
            logger.error("gdpr_retention_policies is missing -- exception_fields show no policy periods")
            rows = []
        policies = {r["data_category"]: r for r in rows}

        def entry(data: str, basis: str) -> str:
            if data == "audit_events":
                return f"{data}_{_period(settings.audit_retention_days)}: {basis}"
            policy = policies.get(data)
            if policy is None:
                return f"{data}_indefinite: {basis}"
            text = f"{data}_{_period(policy['retention_days'])}: {basis}"
            if not policy["auto_delete"]:
                text += " -- policy period, not deleted automatically"
            return text

        return [entry(data, basis) for data, basis in _RETAINED]

    async def get_job(self, job_id: str) -> ErasureJob | None:
        async with self.pool.acquire() as conn:
            # The table also holds export and consent-withdrawal jobs --
            # neither may be mistaken for (or retried as) a full erasure.
            row = await conn.fetchrow(
                "SELECT * FROM gdpr_erasure_jobs WHERE job_id = $1 AND type = 'erasure'", job_id,
            )
        return _row_to_job(row) if row else None

    async def _update_job(
        self, job_id: str, *, status: JobStatus | None = None,
        store_statuses: dict[str, StoreStatus] | None = None,
        completed_at: datetime | None = None, notes: str | None = None,
    ) -> None:
        sets, args = [], []
        if status is not None:
            args.append(status.value)
            sets.append(f"status = ${len(args)}")
        if store_statuses is not None:
            args.append(json.dumps({k: v.value for k, v in store_statuses.items()}))
            sets.append(f"store_statuses = ${len(args)}")
        if completed_at is not None:
            args.append(completed_at)
            sets.append(f"completed_at = ${len(args)}")
        if notes is not None:
            args.append(notes)
            sets.append(f"notes = ${len(args)}")
        if not sets:
            return
        args.append(job_id)
        async with self.pool.acquire() as conn:
            await conn.execute(f"UPDATE gdpr_erasure_jobs SET {', '.join(sets)} WHERE job_id = ${len(args)}", *args)  # nosec B608 -- sets built above from hardcoded "col = $N"-shaped literals only, column names never request-derived

    # ── Orchestration entry point (run as a background task) ───────────────

    async def run(self, job_id: str) -> None:
        job = await self.get_job(job_id)
        if job is None:
            logger.error("run() called for unknown job_id=%s", job_id)
            return

        # PRIV-X4: a hold placed after the job was created still stops it,
        # before anything is disabled or deleted.
        if not await self._start_unless_held(job_id, job.user_id):
            logger.info("job=%s deferred: the user is under a legal hold", job_id)
            return

        try:
            await asyncio.wait_for(self._disable_keycloak_account(job.user_id), timeout=60.0)
        except Exception as exc:  # noqa: BLE001 -- disabling can fail for any reason; erasure still proceeds against the data stores, already logged
            logger.error("job=%s: disabling Keycloak account failed: %s", job_id, exc)
            # Erasure can still proceed against the data stores even if the
            # account couldn't be disabled -- the account itself is deleted
            # at the very end anyway, once every store confirms.

        # Stop Module 25's Kafka consumer from indexing anything further for
        # this user before the parallel fan-out even starts.
        await self.redis.sadd(_ERASED_USERS_SET, job.user_id)
        await self.redis.expire(_ERASED_USERS_SET, settings.erased_users_ttl_seconds)

        store_statuses = dict(job.store_statuses)
        deleters = self._deleters()

        async def _run_one(key: str) -> None:
            if deleters[key] is None:
                store_statuses[key] = StoreStatus.SKIPPED
                return
            store_statuses[key] = StoreStatus.IN_PROGRESS
            try:
                await self._with_retry(deleters[key], job.user_id)
                store_statuses[key] = StoreStatus.DELETED
            except Exception as exc:  # noqa: BLE001 -- deleters[key] spans httpx/asyncpg/redis/timeout errors depending on store; marked FAILED, other stores continue, already logged
                logger.error("job=%s store=%s: all attempts failed: %s", job_id, key, exc)
                store_statuses[key] = StoreStatus.FAILED

        await asyncio.gather(*(_run_one(key) for key in deleters))
        await self._update_job(job_id, store_statuses=store_statuses)

        await self._publish_tombstone(job.user_id, job_id)

        await self._finish(job, store_statuses, notify_dpo=True)

    async def _start_unless_held(self, job_id: str, user_id: str) -> bool:
        """IN_PROGRESS and True, or DEFERRED and False under a legal hold --
        decided under the lock LegalHoldService.place() takes, so a hold
        either stops this run or is refused because the run has started."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await lock_user(conn, user_id)
                held = await active_hold(conn, user_id) is not None
                status = JobStatus.DEFERRED if held else JobStatus.IN_PROGRESS
                await conn.execute("UPDATE gdpr_erasure_jobs SET status = $1 WHERE job_id = $2", status.value, job_id)
        return not held

    async def claim_retry(self, job_id: str, user_id: str) -> tuple[LegalHold | None, bool]:
        """PRIV-X4: (the user's active hold, whether the retry may run). Under
        the same lock as _start_unless_held and LegalHoldService.place(): with
        no hold, the PARTIAL_FAILURE job is set IN_PROGRESS, so a hold placed
        meanwhile is refused as for a running erasure. False without a hold
        means another retry claimed the job first."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await lock_user(conn, user_id)
                hold = await active_hold(conn, user_id)
                if hold is not None:
                    return hold, False
                claimed = await conn.fetchval(
                    "UPDATE gdpr_erasure_jobs SET status = $1 WHERE job_id = $2 AND status = $3 RETURNING job_id",
                    JobStatus.IN_PROGRESS.value, job_id, JobStatus.PARTIAL_FAILURE.value,
                )
        return None, claimed is not None

    async def retry(self, job_id: str) -> ErasureJob:
        """Re-run only the stores still marked failed."""
        job = await self.get_job(job_id)
        if job is None:
            raise ValueError(f"Unknown job_id {job_id}")

        deleters = self._deleters()
        store_statuses = dict(job.store_statuses)
        to_retry = [k for k, v in store_statuses.items() if v == StoreStatus.FAILED]

        async def _run_one(key: str) -> None:
            if deleters.get(key) is None:
                store_statuses[key] = StoreStatus.SKIPPED
                return
            store_statuses[key] = StoreStatus.IN_PROGRESS
            try:
                await self._with_retry(deleters[key], job.user_id)
                store_statuses[key] = StoreStatus.DELETED
            except Exception as exc:  # noqa: BLE001 -- deleters[key] spans httpx/asyncpg/redis/timeout errors depending on store; marked FAILED, other stores continue, already logged
                logger.error("job=%s store=%s: retry failed: %s", job_id, key, exc)
                store_statuses[key] = StoreStatus.FAILED

        await asyncio.gather(*(_run_one(key) for key in to_retry))
        await self._update_job(job_id, store_statuses=store_statuses)

        await self._finish(job, store_statuses, notify_dpo=False)

        return await self.get_job(job_id)

    async def _finish(self, job: ErasureJob, store_statuses: dict[str, StoreStatus], *, notify_dpo: bool) -> None:
        """Shared tail of run() and retry(): COMPLETED once every store is
        done, PARTIAL_FAILURE otherwise -- each with its audit event."""
        job_id = job.job_id

        def event(outcome):
            return erasure_event(
                job_id=job_id, user_id=job.user_id, requested_by=job.requested_by,
                job_type=JobType.ERASURE.value, outcome=outcome, store_statuses=store_statuses,
            )

        if all(s in _DONE for s in store_statuses.values()):
            try:
                await self._hard_delete_keycloak_account(job.user_id)
            except Exception as exc:  # noqa: BLE001 -- hard-delete can fail for any reason; job is already marked completed, already logged
                logger.error("job=%s: hard-delete of Keycloak account failed: %s", job_id, exc)
            try:
                await self.audit.emit(event("erase"), fail_closed=True)
            except AuditWriteError as exc:
                logger.error("job=%s: erasure audit event not written, job left PARTIAL_FAILURE: %s", job_id, exc)
                await self._update_job(
                    job_id, status=JobStatus.PARTIAL_FAILURE,
                    notes="Every store erased, but the erasure audit event could not be written -- "
                          "retry the job to write it",
                )
                if notify_dpo:
                    await self._notify_dpo(job.user_id, job_id, ["audit_event"])
                return
            await self._update_job(
                job_id, status=JobStatus.COMPLETED,
                completed_at=datetime.now(timezone.utc), notes=_completion_notes(store_statuses),
            )
        else:
            failed = [k for k, v in store_statuses.items() if v not in _DONE]
            await self._update_job(
                job_id, status=JobStatus.PARTIAL_FAILURE,
                notes=f"Stores still failing after {settings.erasure_max_attempts} attempts each: {', '.join(failed)}",
            )
            await self.audit.emit(event("error"))
            if notify_dpo:
                await self._notify_dpo(job.user_id, job_id, failed)

    # ── Retry wrapper ───────────────────────────────────────────────────────

    async def _with_retry(self, fn: Callable[[str], Awaitable[None]], user_id: str) -> None:
        last_exc: Exception | None = None
        for attempt in range(1, settings.erasure_max_attempts + 1):
            try:
                await asyncio.wait_for(fn(user_id), timeout=settings.erasure_timeout_seconds)
                return
            except Exception as exc:  # noqa: BLE001 -- fn is any per-store deleter (httpx/asyncpg/redis) plus asyncio timeout; retried then re-raised to the caller, already logged
                last_exc = exc
                logger.warning("attempt %d/%d for %s failed: %s", attempt, settings.erasure_max_attempts, fn.__name__, exc)
                if attempt < settings.erasure_max_attempts:
                    await asyncio.sleep(settings.erasure_retry_delay_seconds)
        raise last_exc  # type: ignore[misc]

    # ── Per-store deletion ──────────────────────────────────────────────────

    def _deleters(self) -> dict[str, Callable[[str], Awaitable[None]] | None]:
        """Every store in STORE_KEYS; None for a module with no URL
        configured (recorded as `skipped`)."""
        return {
            "module_04": self._delete_module_04,
            "module_25": self._delete_module_25,
            "module_26": self._delete_module_26,
            "module_03": self._delete_module_03,
            "module_23": self._delete_module_23,
            "module_02": self._delete_module_02,
            "module_34": self._delete_module_34 if settings.module_34_url else None,
            "module_35": self._delete_module_35 if settings.module_35_url else None,
        }

    async def _delete_module_04(self, user_id: str) -> None:
        resp = await self.http.delete(f"{settings.module_04_url}/internal/gdpr/user/{user_id}", headers=_INTERNAL_HEADERS)
        resp.raise_for_status()

    async def _delete_module_25(self, user_id: str) -> None:
        resp = await self.http.delete(f"{settings.module_25_url}/internal/gdpr/user/{user_id}", headers=_INTERNAL_HEADERS)
        resp.raise_for_status()

    async def _delete_module_26(self, user_id: str) -> None:
        # Always 200 with {"deleted": bool} -- no memory ever existing for
        # this user isn't an erasure failure, so the endpoint doesn't 404.
        resp = await self.http.delete(f"{settings.module_26_url}/internal/gdpr/user/{user_id}", headers=_INTERNAL_HEADERS)
        resp.raise_for_status()

    async def _delete_module_03(self, user_id: str) -> None:
        """Direct SQL deletion against Module 03's `benchmark` database --
        the only table in the ticket's "Module 03" data-inventory bullet
        that both actually exists and has a per-user column. The other
        three named there (violations, assets, consent_history) don't exist
        in this codebase: `violations` is really `guardrail_violations` in
        the `guardrails` database and has no user_id/branch_id column at
        all (nothing to scope a delete to); `assets` is module 04's own
        commit_history, already handled by module 04's own endpoint;
        `consent_history` doesn't exist anywhere. Connects directly (not
        via self.pool, which points at this service's own
        privacy_compliance database) since each service here owns its own
        database rather than sharing one."""
        conn = await asyncpg.connect(
            host=settings.postgres_host, port=settings.postgres_port,
            user=settings.postgres_user, password=settings.postgres_password,
            database=settings.benchmark_postgres_db,
        )
        try:
            await conn.execute("DELETE FROM benchmark_runs WHERE branch_id = $1", user_id)
        finally:
            await conn.close()

    async def _delete_module_23(self, user_id: str) -> None:
        """Direct SQL deletion against Module 23's `guardrails` database --
        moderation_flags (CM-5 / VIE01-408) has a branch_id column, unlike
        guardrail_violations in the same database (no user column, never
        erasable -- see _delete_module_03's docstring for that reasoning).
        Same connect-directly pattern as _delete_module_03."""
        conn = await asyncpg.connect(
            host=settings.postgres_host, port=settings.postgres_port,
            user=settings.postgres_user, password=settings.postgres_password,
            database=settings.guardrails_postgres_db,
        )
        try:
            await conn.execute("DELETE FROM moderation_flags WHERE branch_id = $1", user_id)
        finally:
            await conn.close()

    async def _delete_module_02(self, user_id: str) -> None:
        patterns = [f"*:{user_id}:*", f"chat_memory:{user_id}:*", f"chat_memory:{user_id}"]
        keys: list[str] = []
        for pattern in patterns:
            async for key in self.redis.scan_iter(match=pattern):
                keys.append(key)
        if keys:
            await self.redis.delete(*keys)

    async def _delete_module_34(self, user_id: str) -> None:
        # Voice recordings (WAV files + rt_audio_recordings) -- only called
        # while GD_MODULE_34_URL is set; empty records module_34 as skipped.
        resp = await self.http.delete(f"{settings.module_34_url}/internal/gdpr/user/{user_id}", headers=_INTERNAL_HEADERS)
        resp.raise_for_status()

    async def _delete_module_35(self, user_id: str) -> None:
        # Voice-clone speaker embeddings. Module 35 keys them under one
        # per-user namespace, so this removes all of them, not best-effort.
        resp = await self.http.delete(f"{settings.module_35_url}/internal/gdpr/user/{user_id}", headers=_INTERNAL_HEADERS)
        resp.raise_for_status()

    # ── Keycloak (Module 06) ─────────────────────────────────────────────────

    async def _keycloak_admin_token(self) -> str:
        resp = await self.http.post(
            f"{settings.keycloak_url}/realms/master/protocol/openid-connect/token",
            data={
                "grant_type": "password",
                "client_id": "admin-cli",
                "username": settings.keycloak_admin_username,
                "password": settings.keycloak_admin_password,
            },
        )
        resp.raise_for_status()
        return resp.json()["access_token"]

    async def _disable_keycloak_account(self, user_id: str) -> None:
        # user_id IS the Keycloak user UUID already -- it comes from the
        # caller's JWT `sub` claim (module 13 forwards it as X-User-ID),
        # and Keycloak's `sub` *is* that user's Admin API id in this realm.
        # No username lookup needed -- searching ?username=<uuid> (the
        # previous approach here) never matches anything, since a UUID is
        # never anyone's username, and silently no-ops instead of erroring.
        token = await self._keycloak_admin_token()
        resp = await self.http.put(
            f"{settings.keycloak_url}/admin/realms/{settings.keycloak_realm}/users/{user_id}",
            headers={"Authorization": f"Bearer {token}"},
            json={"enabled": False},
        )
        if resp.status_code == 404:
            return  # already gone / never existed
        resp.raise_for_status()
        # Revoke all active sessions immediately -- disabling alone doesn't
        # invalidate tokens already issued.
        await self.http.post(
            f"{settings.keycloak_url}/admin/realms/{settings.keycloak_realm}/users/{user_id}/logout",
            headers={"Authorization": f"Bearer {token}"},
        )

    async def _hard_delete_keycloak_account(self, user_id: str) -> None:
        token = await self._keycloak_admin_token()
        resp = await self.http.delete(
            f"{settings.keycloak_url}/admin/realms/{settings.keycloak_realm}/users/{user_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        if resp.status_code == 404:
            return
        resp.raise_for_status()

    # ── Kafka tombstone ─────────────────────────────────────────────────────

    async def _publish_tombstone(self, user_id: str, job_id: str) -> None:
        if self.kafka_producer is None:
            logger.warning("Kafka producer not configured -- skipping tombstone publish for job=%s", job_id)
            return
        message = {
            "event": "user-erasure-requested",
            "user_id": user_id,
            "job_id": job_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await self.kafka_producer.send_and_wait(
            settings.kafka_erasure_topic, json.dumps(message).encode("utf-8"), key=user_id.encode("utf-8"),
        )

    # ── DPO notification ─────────────────────────────────────────────────────

    async def _notify_dpo(self, user_id: str, job_id: str, failed_stores: list[str]) -> None:
        if not settings.dpo_email or not settings.smtp_host:
            logger.warning("GD_DPO_EMAIL/GD_SMTP_HOST not configured -- cannot notify DPO for job=%s", job_id)
            return

        def _send() -> None:
            msg = EmailMessage()
            msg["From"] = settings.smtp_from or settings.smtp_user
            msg["To"] = settings.dpo_email
            msg["Subject"] = f"[RAPTOR GDPR] Erasure job {job_id} — PARTIAL_FAILURE"
            msg.set_content(
                f"Erasure job {job_id} for user {user_id} is in PARTIAL_FAILURE state.\n\n"
                f"Stores that failed after {settings.erasure_max_attempts} attempts: {', '.join(failed_stores)}\n\n"
                f"Retry with: POST /api/{settings.api_version}/gdpr/erasure/jobs/{job_id}/retry"
            )
            with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
                server.starttls()
                if settings.smtp_user:
                    server.login(settings.smtp_user, settings.smtp_password)
                server.send_message(msg)

        try:
            await asyncio.to_thread(_send)
        except OSError as exc:
            logger.error("Failed to send DPO notification for job=%s: %s", job_id, exc)


def _period(days: int) -> str:
    """The ticket's "7yr" style: whole years as yr, anything else in days."""
    return f"{days // 365}yr" if days % 365 == 0 else f"{days}d"


def _completion_notes(store_statuses: dict[str, StoreStatus]) -> str:
    notes = (
        f"log_expiry_at: {(datetime.now(timezone.utc) + timedelta(days=30)).isoformat()} "
        "(Loki logs are not deleted -- see GD-3 retention policy; documented in DSAR/RoPA)"
    )
    skipped = [k for k, v in store_statuses.items() if v == StoreStatus.SKIPPED]
    if skipped:
        notes += f"; skipped (no GDPR endpoint configured, nothing stored there): {', '.join(skipped)}"
    return notes


def _row_to_job(row: Any) -> ErasureJob:
    store_statuses_raw = row["store_statuses"]
    if isinstance(store_statuses_raw, str):
        store_statuses_raw = json.loads(store_statuses_raw)
    return ErasureJob(
        # asyncpg returns the UUID column as a uuid.UUID, not a str -- the
        # model (and every API response) expects a plain string.
        job_id=str(row["job_id"]),
        user_id=row["user_id"],
        requested_at=row["requested_at"],
        requested_by=row["requested_by"],
        status=JobStatus(row["status"]),
        store_statuses={k: StoreStatus(v) for k, v in store_statuses_raw.items()},
        completed_at=row["completed_at"],
        notes=row["notes"],
    )
