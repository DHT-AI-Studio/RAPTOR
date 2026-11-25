"""GD-4 / VIE01-249: Data Subject Access Request (GDPR Art. 15 / CCPA
§ 1798.100) assembler.

Counts only, never content: Modules 04/25/26 -- every module with GD-1's
/internal/gdpr/user/{user_id} endpoints -- are asked for their count-only
.../summary endpoint instead of GD-2's export path, Keycloak for the
identity fields and GD-5's consent attributes, and Module 03's
benchmark_runs / benchmark_schemas for a COUNT(*) each.
Every store is queried concurrently; one that fails doesn't sink the
report -- its fields stay null and it's named in stores_failed.

Sync vs async (the ticket's ">10k indexed items" rule): Module 25's summary
is the decision input, since indexed items are exactly what it counts.
If it reports more than GD_DSAR_ASYNC_THRESHOLD, the caller gets
{job_id, status: "pending"} immediately and the report finishes in the
background (gdpr_erasure_jobs, type='dsar'), polled like GD-1/GD-2. A
normal account is answered synchronously -- unless assembly overruns
GD_DSAR_SYNC_TIMEOUT_SECONDS for any reason (a slow store, Module 25 itself
not answering in time), in which case the very same in-flight assembly is
handed off to a job rather than restarted or abandoned. That keeps the
ticket's 10-second response bound unconditional, not just typical.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import asyncpg
import httpx

from app.core.config import settings
from app.models.dsar import (
    ConsentStatus,
    DataCategories,
    DSARJob,
    DSARPendingResponse,
    DSARReport,
    RetentionSummary,
    Rights,
)
from app.models.job import JobStatus
from app.services.audit_events import audit_writer, privacy_event
from app.services.consent_store import DEFAULTS, parse_consent

logger = logging.getLogger("privacy_compliance.dsar")

_INTERNAL_HEADERS = {"X-Internal-Token": settings.internal_token}

# Ticket-specified CCPA disclosures -- fixed facts about this platform, not
# per-user data.
CCPA_CATEGORY = "Identifiers, Commercial information, Internet activity"
CCPA_SOURCE = "Directly from consumer"
CCPA_SELLS_DATA = False

# Fields the ticket's structure asks for that this codebase has no real
# backing for. Reported as null with the reason, never as a fabricated
# value -- same policy GD-1/GD-2 follow for the same gaps.
UNAVAILABLE_FIELDS: dict[str, str] = {
    "data_categories.uploaded_files.total_bytes": (
        "Module 04's commit_history records no file size, and every user's "
        "objects share one LakeFS branch -- no per-user size exists to sum "
        "without stat-ing every object, which a count-only DSAR must not do"
    ),
}

# retention_summary key -> gdpr_retention_policies.data_category (GD-3).
# `files` has no policy row: uploaded files are kept until the user
# deletes them or sets their own archive/destroy date on an asset.
# `violations` is handled separately (_violations_label) -- moderation_flags
# reports guardrails.moderation_flags' policy, not guardrail_violations' --
# moderation_flags is the table actually scoped to this user (branch_id);
# guardrail_violations has no owner column and was never erasable, per
# safety_violations.count below.
_RETENTION_CATEGORIES = {"memory": "memvid_sessions"}
# CM-5 / VIE01-408: moderation_flags was split by status into two policies --
# 'cleared' purges sooner than 'flagged'/'removed' -- so there's no single
# number that's correct for every user; _violations_label reports both.
_MODERATION_FLAG_CATEGORIES = ("moderation_flags", "moderation_flags_actionable")
_INDEFINITE = "indefinite"


def indexed_items(module_25_summary: dict) -> int:
    """The ticket's ">10k indexed items" -- every indexed record Module 25
    holds for the user (all chunks + entities), from the summary's
    internal-only indexed_items field. Not a sum of the report's
    indexed_content fields: `documents` there counts documents, not the
    chunks a large account's cost actually scales with."""
    return int(module_25_summary.get("indexed_items") or 0)


def retention_label(policy: dict | None) -> str:
    """A category only has a finite retention if a policy row exists AND
    actually auto-deletes -- a legal-hold or operator-disabled row means
    the data is in practice kept indefinitely."""
    if not policy or not policy.get("auto_delete"):
        return _INDEFINITE
    return f"{int(policy['retention_days'])} days"


def _violations_label(policies: dict[str, dict]) -> str:
    """Policy-level description, like every other retention_summary field --
    not which one applies to this specific user's own flags (the report
    doesn't fetch their statuses, just a count)."""
    cleared = retention_label(policies.get("moderation_flags"))
    actionable = retention_label(policies.get("moderation_flags_actionable"))
    return f"{cleared} (cleared) / {actionable} (flagged or removed)"


def _rights() -> Rights:
    base = f"/api/{settings.api_version}/gdpr"
    return Rights(erasure_endpoint=f"{base}/erasure", export_endpoint=f"{base}/export", consent_endpoint=f"{base}/consent")


def consent_from_profile(profile: dict) -> dict[str, bool]:
    """GD-5's consent attributes from a Keycloak user, read-only. A purpose
    never set (an account from before GD-5) reports GD-5's default -- what
    GET /gdpr/consent would show -- without ConsentStore.get()'s write of
    those defaults: a DSAR changes nothing."""
    return {
        purpose.value: DEFAULTS[purpose] if granted is None else granted
        for purpose, granted in parse_consent(profile).items()
    }


class DSARAssembler:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool  # this service's own privacy_compliance database
        self.http = httpx.AsyncClient(timeout=30.0)
        self.audit = audit_writer(pool)
        # Strong references to handed-off assembly tasks -- asyncio only
        # keeps weak ones, and a GC'd task would silently drop a job.
        self._tasks: set[asyncio.Task] = set()

    async def close(self) -> None:
        for task in self._tasks:
            task.cancel()
        await self.http.aclose()

    # ── Entry point ─────────────────────────────────────────────────────────

    async def _audit(self, user_id: str, requested_by: str, ok: bool, *, job_id: str | None,
                     stores_failed: list[str] | None = None) -> None:
        """V05-10 / VIE01-364: one CF-4 `gdpr.access` event per finished
        request -- answered at once (no job id) or through a job."""
        await self.audit.emit(privacy_event(
            "gdpr.access", user_id=user_id, action="access", ok=ok, requested_by=requested_by,
            resource_id=job_id, metadata={"stores_failed": stores_failed or [], "async": job_id is not None}))

    async def request(self, user_id: str, requested_by: str) -> DSARReport | DSARPendingResponse:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + settings.dsar_sync_timeout_seconds

        m25_task = asyncio.create_task(self._fetch_module_25(user_id))
        report_task = asyncio.create_task(self._build_report(user_id, m25_task))

        await asyncio.wait({m25_task}, timeout=settings.dsar_sync_timeout_seconds)
        if (m25_task.done() and not m25_task.cancelled() and m25_task.exception() is None
                and indexed_items(m25_task.result()) > settings.dsar_async_threshold):
            return await self._hand_off(report_task, user_id, requested_by)

        await asyncio.wait({report_task}, timeout=max(deadline - loop.time(), 0))
        if report_task.done():
            try:
                report = report_task.result()
            except Exception:
                await self._audit(user_id, requested_by, False, job_id=None)
                raise
            await self._audit(user_id, requested_by, not report.stores_failed, job_id=None,
                              stores_failed=report.stores_failed)
            return report
        logger.info("dsar user=%s: sync assembly overran %.1fs, handing off to a job", user_id, settings.dsar_sync_timeout_seconds)
        return await self._hand_off(report_task, user_id, requested_by)

    async def _hand_off(self, report_task: asyncio.Task, user_id: str, requested_by: str) -> DSARPendingResponse:
        job_id = await self.create_job(user_id, requested_by)
        task = asyncio.create_task(self._finish_job(job_id, report_task, user_id, requested_by))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return DSARPendingResponse(job_id=job_id, status=JobStatus.PENDING.value)

    async def _finish_job(self, job_id: str, report_task: asyncio.Task, user_id: str, requested_by: str) -> None:
        await self._update_job(job_id, status=JobStatus.IN_PROGRESS)
        try:
            report: DSARReport = await report_task
        except Exception as exc:  # noqa: BLE001 -- report assembly spans httpx/DB/pydantic/dict-access errors; already logged and recorded as job failure
            logger.error("dsar job=%s: assembly failed: %s", job_id, exc)
            await self._update_job(job_id, status=JobStatus.PARTIAL_FAILURE, notes=f"DSAR assembly failed: {exc}")
            await self._audit(user_id, requested_by, False, job_id=job_id)
            return
        now = datetime.now(timezone.utc)
        await self._update_job(
            job_id,
            status=JobStatus.COMPLETED if not report.stores_failed else JobStatus.PARTIAL_FAILURE,
            notes=f"stores_failed: {', '.join(report.stores_failed)}" if report.stores_failed else None,
            result=report.model_dump(mode="json"),
            completed_at=now,
            expires_at=now + timedelta(hours=settings.dsar_result_ttl_hours),
        )
        await self._audit(user_id, requested_by, not report.stores_failed, job_id=job_id,
                          stores_failed=report.stores_failed)

    # ── Report assembly ─────────────────────────────────────────────────────

    async def _build_report(self, user_id: str, m25_task: asyncio.Task) -> DSARReport:
        names = ("module_06", "module_04", "module_23", "module_26", "module_03", "retention")
        results = await asyncio.gather(
            self._fetch_identity(user_id),
            self._fetch_uploaded_files(user_id),
            self._fetch_moderation_flags(user_id),
            self._fetch_module_26(user_id),
            self._fetch_benchmark(user_id),
            self._fetch_retention_policies(),
            return_exceptions=True,
        )
        fetched: dict[str, Any] = dict(zip(names, results))
        try:
            fetched["module_25"] = await m25_task
        except Exception as exc:  # noqa: BLE001 -- m25_task can fail for any reason (httpx/json/other); recorded as a per-store failure and logged by the caller, not lost
            fetched["module_25"] = exc

        stores_failed: list[str] = []
        for name in ("module_06", "module_04", "module_23", "module_25", "module_26", "module_03", "retention"):
            if isinstance(fetched[name], BaseException):
                logger.error("dsar user=%s: %s failed: %s", user_id, name, fetched[name])
                stores_failed.append(name)
                fetched[name] = None

        categories = DataCategories()
        consent = ConsentStatus()
        if (identity := fetched["module_06"]) is not None:
            categories.identity.email = identity.get("email")
            categories.identity.username = identity.get("username")
            categories.identity.created_at = identity.get("created_at")
        # GD-5 / VIE01-250: absent when there's no Keycloak account.
        if identity is not None and identity.get("consent") is not None:
            consent = ConsentStatus(**identity["consent"])
        if (files := fetched["module_04"]) is not None:
            categories.uploaded_files.count = files["count"]
            categories.uploaded_files.file_types = files["file_types"]
        if (flags := fetched["module_23"]) is not None:
            categories.safety_violations.count = flags["count"]
        if (m25 := fetched["module_25"]) is not None:
            categories.indexed_content.documents = m25.get("documents")
            categories.indexed_content.entities = m25.get("entities")
            categories.indexed_content.moments = m25.get("moments")
            # GD-6 / VIE01-251: Chunks flagged as containing PII at ingestion.
            categories.indexed_content.pii_flagged = m25.get("pii_flagged")
        if (m26 := fetched["module_26"]) is not None:
            categories.memory.sessions = m26.get("sessions")
            categories.memory.long_term_facts = m26.get("long_term_facts")
            categories.memory.oldest_session = m26.get("oldest_session")
        if (benchmark := fetched["module_03"]) is not None:
            categories.benchmark.runs = benchmark["runs"]
            categories.benchmark.schemas_owned = benchmark["schemas_owned"]

        policies = fetched["retention"]
        if policies is None:
            # Can't read the policy table -- no honest value to report for
            # memory/violations, and `files` would look authoritative next
            # to them; stores_failed already names "retention".
            retention = RetentionSummary(files="unknown", memory="unknown", violations="unknown")
        else:
            retention = RetentionSummary(
                files=_INDEFINITE,
                violations=_violations_label(policies),
                **{key: retention_label(policies.get(cat)) for key, cat in _RETENTION_CATEGORIES.items()},
            )

        return DSARReport(
            user_id=user_id,
            generated_at=datetime.now(timezone.utc),
            data_categories=categories,
            consent_status=consent,
            retention_summary=retention,
            ccpa_category=CCPA_CATEGORY,
            ccpa_source=CCPA_SOURCE,
            ccpa_sells_data=CCPA_SELLS_DATA,
            rights=_rights(),
            unavailable_fields=dict(UNAVAILABLE_FIELDS),
            stores_failed=stores_failed,
        )

    # ── Per-store fetchers (counts only) ────────────────────────────────────

    async def _fetch_module_25(self, user_id: str) -> dict:
        resp = await self.http.get(
            f"{settings.module_25_url}/internal/gdpr/user/{user_id}/summary", headers=_INTERNAL_HEADERS,
        )
        resp.raise_for_status()
        return resp.json()

    async def _fetch_module_26(self, user_id: str) -> dict:
        resp = await self.http.get(
            f"{settings.module_26_url}/internal/gdpr/user/{user_id}/summary", headers=_INTERNAL_HEADERS,
        )
        resp.raise_for_status()
        return resp.json()

    async def _fetch_uploaded_files(self, user_id: str) -> dict:
        resp = await self.http.get(
            f"{settings.module_04_url}/internal/gdpr/user/{user_id}/summary", headers=_INTERNAL_HEADERS,
        )
        resp.raise_for_status()
        return resp.json()

    async def _keycloak_admin_token(self) -> str:
        resp = await self.http.post(
            f"{settings.keycloak_url}/realms/master/protocol/openid-connect/token",
            data={
                "grant_type": "password", "client_id": "admin-cli",
                "username": settings.keycloak_admin_username, "password": settings.keycloak_admin_password,
            },
        )
        resp.raise_for_status()
        return resp.json()["access_token"]

    async def _fetch_identity(self, user_id: str) -> dict:
        token = await self._keycloak_admin_token()
        resp = await self.http.get(
            f"{settings.keycloak_url}/admin/realms/{settings.keycloak_realm}/users/{user_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        if resp.status_code == 404:
            # No Keycloak account (e.g. already erased) -- a real answer,
            # not a store failure: there is no identity data held.
            return {}
        resp.raise_for_status()
        profile = resp.json()
        created_ms = profile.get("createdTimestamp")
        return {
            "email": profile.get("email"),
            "username": profile.get("username"),
            "created_at": (
                datetime.fromtimestamp(created_ms / 1000, tz=timezone.utc).isoformat() if created_ms else None
            ),
            "consent": consent_from_profile(profile),
        }

    async def _fetch_benchmark(self, user_id: str) -> dict[str, int]:
        """benchmark_runs.branch_id == submitter's JWT sub -- same column
        GD-1's erasure and GD-2's export scope this table by.
        benchmark_schemas.owner_id is the creator's JWT sub (GD-5); schemas
        created before GD-5 have no owner and can't be counted for anyone."""
        conn = await asyncpg.connect(
            host=settings.postgres_host, port=settings.postgres_port,
            user=settings.postgres_user, password=settings.postgres_password,
            database=settings.benchmark_postgres_db,
        )
        try:
            return {
                "runs": await conn.fetchval("SELECT COUNT(*) FROM benchmark_runs WHERE branch_id = $1", user_id),
                "schemas_owned": await conn.fetchval(
                    "SELECT COUNT(*) FROM benchmark_schemas WHERE owner_id = $1", user_id,
                ),
            }
        finally:
            await conn.close()

    async def _fetch_moderation_flags(self, user_id: str) -> dict[str, int]:
        """data_categories.safety_violations.count: moderation_flags (Module
        23, CM-5) has a branch_id column, unlike guardrail_violations (no
        owner column at all -- see UNAVAILABLE_FIELDS' history for that
        gap). Direct SQL against the `guardrails` database, same
        connect-directly pattern as _fetch_benchmark."""
        conn = await asyncpg.connect(
            host=settings.postgres_host, port=settings.postgres_port,
            user=settings.postgres_user, password=settings.postgres_password,
            database=settings.guardrails_postgres_db,
        )
        try:
            count = await conn.fetchval("SELECT COUNT(*) FROM moderation_flags WHERE branch_id = $1", user_id)
        finally:
            await conn.close()
        return {"count": count}

    async def _fetch_retention_policies(self) -> dict[str, dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT data_category, retention_days, auto_delete FROM gdpr_retention_policies "
                "WHERE data_category = ANY($1)",
                [*_RETENTION_CATEGORIES.values(), *_MODERATION_FLAG_CATEGORIES],
            )
        return {row["data_category"]: dict(row) for row in rows}

    # ── Job persistence (gdpr_erasure_jobs, type='dsar') ────────────────────

    async def create_job(self, user_id: str, requested_by: str) -> str:
        job_id = str(uuid.uuid4())
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO gdpr_erasure_jobs
                    (job_id, user_id, requested_at, requested_by, status, store_statuses, type)
                VALUES ($1, $2, $3, $4, $5, $6, 'dsar')
                """,
                job_id, user_id, datetime.now(timezone.utc), requested_by, JobStatus.PENDING.value, json.dumps({}),
            )
        return job_id

    async def get_job(self, job_id: str) -> DSARJob | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM gdpr_erasure_jobs WHERE job_id = $1 AND type = 'dsar'", job_id)
        return _row_to_dsar_job(row) if row else None

    async def _update_job(
        self, job_id: str, *, status: JobStatus | None = None, notes: str | None = None,
        result: dict | None = None, completed_at: datetime | None = None,
        expires_at: datetime | None = None,
    ) -> None:
        sets, args = [], []
        for col, val in (
            ("status", status.value if status is not None else None), ("notes", notes),
            ("result", json.dumps(result) if result is not None else None),
            ("completed_at", completed_at), ("expires_at", expires_at),
        ):
            if val is not None:
                args.append(val)
                sets.append(f"{col} = ${len(args)}")
        if not sets:
            return
        args.append(job_id)
        async with self.pool.acquire() as conn:
            await conn.execute(f"UPDATE gdpr_erasure_jobs SET {', '.join(sets)} WHERE job_id = ${len(args)}", *args)  # nosec B608 -- sets built above from a hardcoded ("col", val) tuple, column names never request-derived

    async def cleanup_expired(self) -> int:
        """Clears the stored report of every DSAR job past its expires_at.
        The job row itself stays (GD-3's gdpr_erasure_jobs legal hold) --
        only the copy of the user's data in `result` goes."""
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE gdpr_erasure_jobs SET result = NULL
                WHERE type = 'dsar' AND result IS NOT NULL
                  AND expires_at IS NOT NULL AND expires_at < now()
                """,
            )
        cleared = int(result.split()[-1])
        if cleared:
            logger.info("cleanup: cleared %d expired DSAR result(s)", cleared)
        return cleared


def _row_to_dsar_job(row: Any) -> DSARJob:
    result = row["result"]
    if isinstance(result, str):
        result = json.loads(result)
    return DSARJob(
        job_id=str(row["job_id"]),
        user_id=row["user_id"],
        requested_at=row["requested_at"],
        requested_by=row["requested_by"],
        status=JobStatus(row["status"]),
        completed_at=row["completed_at"],
        notes=row["notes"],
        result=result,
        expires_at=row["expires_at"],
    )
