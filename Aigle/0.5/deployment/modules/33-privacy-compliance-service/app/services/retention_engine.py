"""GD-3 / VIE01-248: the retention state machine.

enforce() runs daily (APScheduler, 02:00 UTC by default -- see app/main.py)
and purges every auto_delete=True category past its retention_days.
dry_run() runs the exact same per-category counts without deleting
anything, for GET /retention/policy/dry-run.

Two of the seeded policy rows (gdpr_consent_history, gdpr_erasure_jobs)
are auto_delete=False legal-hold records by design -- enforce()/dry_run()
skip them via the same auto_delete check a real operator-toggled row would
hit, not a special case. ENFORCEABLE_CATEGORIES below is the map of every
category this engine actually knows how to purge; a category not in it
(including gdpr_consent_history, since no consent_history table exists in
this codebase to purge anyway) is skipped even if somehow marked
auto_delete=True, rather than raising -- see the module docstring's note in
004_gdpr_retention.sql on why that row exists as documentation only.
GD-10 / VIE01-416's voice_recordings and loki_logs rows are not in it
either: Module 34 deletes its own recordings after 90 days and Loki's
compactor its own logs after 30 -- both rows record the policy, this engine
purges neither.

Every enforcement run -- success or failure, per category -- writes one row
to gdpr_retention_log; a category whose deletion raises gets
rows_deleted=-1 rather than being silently skipped, per the ticket's
explicit requirement.

V05-10 / VIE01-364: purge_audit_events() is a second daily job (same
time) for the CF-4 audit_events tables -- this service's own and Module
23's in the guardrails database. Those rows carry their own retention_days
(set by whoever wrote the event), so they are purged by that value, not by
a gdpr_retention_policies row. Each table's run is logged to
gdpr_retention_log the same way, as data_category `audit_events` /
`audit_events:guardrails`.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import asyncpg
import httpx

from app.core.config import settings
from app.models.retention import RetentionPolicy

logger = logging.getLogger("privacy_compliance.retention")

_INTERNAL_HEADERS = {"X-Internal-Token": settings.internal_token}

# data_category -> (table, timestamp column, extra WHERE fragment | None) for
# the categories enforced by a direct SQL DELETE against another module's own
# Postgres database.
_SQL_CATEGORIES: dict[str, tuple] = {
    "benchmark_runs": ("benchmark_runs", "created_at", None),
    "guardrail_violations": ("guardrail_violations", "created_at", None),
    # CM-5 / VIE01-408: unlike guardrail_violations, moderation_flags has a
    # branch_id column -- also erasable on demand via GD-1 (see
    # ErasureOrchestrator._delete_module_23), which this age-based purge
    # doesn't replace, just as GD-1 and this engine already coexist for
    # every other store here. Split by status, not one blanket policy:
    # 'cleared' (false positive, reviewed and resolved) has no ongoing value
    # and purges at its own short policy below; 'flagged' (still pending
    # review -- silently losing it isn't "resolved") and 'removed'
    # (confirmed violation -- needed for dispute defense / repeat-offender
    # detection) share moderation_flags_actionable's longer, "Legal record"
    # policy, same precedent as gdpr_erasure_jobs' own 1095-day row.
    "moderation_flags": ("moderation_flags", "created_at", "status = 'cleared'"),
    "moderation_flags_actionable": ("moderation_flags", "created_at", "status <> 'cleared'"),
    # Lives in this service's own privacy_compliance database, not a
    # separate connection -- see _db_for_category.
    "gdpr_access_log": ("gdpr_access_log", "timestamp", None),
}
# Enforced via an HTTP call to another module instead of direct SQL.
_HTTP_CATEGORIES = {"memvid_sessions"}
ENFORCEABLE_CATEGORIES = set(_SQL_CATEGORIES) | _HTTP_CATEGORIES

# Each CF-4 event is kept for its own retention_days.
_PURGE_AUDIT_EVENTS_SQL = "DELETE FROM audit_events WHERE timestamp < NOW() - make_interval(days => retention_days)"


def _audit_event_tables() -> tuple[tuple[str, str | None], ...]:
    """(gdpr_retention_log category, database) -- None is this service's
    own database, as in _db_for_category."""
    return (("audit_events", None), ("audit_events:guardrails", settings.guardrails_postgres_db))


async def _purge_audit_events_on(conn) -> int:
    # Module 23 only creates its audit_events table once it runs CF-4's
    # code -- an older deployment has nothing to purge yet.
    if not await conn.fetchval("SELECT to_regclass('audit_events') IS NOT NULL"):
        return 0
    return int((await conn.execute(_PURGE_AUDIT_EVENTS_SQL)).split()[-1])


def _delete_sql(table: str, column: str, days: int, extra_where: str | None = None) -> str:
    """Pure and independently testable -- the ticket's own acceptance
    criterion asks for "per-category SQL generation" to be unit-tested."""
    # table/column/extra_where always come from _SQL_CATEGORIES[category], a
    # fixed whitelist dict (_enforce_one() below); days is int()-coerced right here.
    where = f"{column} < NOW() - INTERVAL '{int(days)} days'"
    if extra_where:
        where += f" AND {extra_where}"
    return f"DELETE FROM {table} WHERE {where}"  # nosec B608


def _count_sql(table: str, column: str, days: int, extra_where: str | None = None) -> str:
    where = f"{column} < NOW() - INTERVAL '{int(days)} days'"
    if extra_where:
        where += f" AND {extra_where}"
    return f"SELECT COUNT(*) FROM {table} WHERE {where}"  # nosec B608 -- same guarantee as _delete_sql above


def _db_for_category(category: str) -> str | None:
    """None means "this service's own privacy_compliance database" (use the
    shared pool); otherwise the database name to open a one-off connection
    against, same pattern ErasureOrchestrator._delete_module_03 already
    uses for benchmark_runs."""
    if category == "benchmark_runs":
        return settings.benchmark_postgres_db
    if category in ("guardrail_violations", "moderation_flags", "moderation_flags_actionable"):
        return settings.guardrails_postgres_db
    return None


class RetentionEngine:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool  # this service's own privacy_compliance database
        self.http = httpx.AsyncClient(timeout=60.0)
        self.last_run: datetime | None = None

    async def close(self) -> None:
        await self.http.aclose()

    # ── Policy CRUD ──────────────────────────────────────────────────────

    async def get_policies(self) -> list[RetentionPolicy]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM gdpr_retention_policies ORDER BY data_category")
        return [RetentionPolicy(**dict(row)) for row in rows]

    async def update_policies(self, updates: list[dict], updated_by: str) -> list[RetentionPolicy]:
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                for update in updates:
                    sets, args = [], []
                    for col in ("retention_days", "auto_delete", "legal_basis", "notes"):
                        if col in update and update[col] is not None:
                            args.append(update[col])
                            sets.append(f"{col} = ${len(args)}")
                    if not sets:
                        continue
                    args.append(datetime.now(timezone.utc))
                    sets.append(f"updated_at = ${len(args)}")
                    args.append(updated_by)
                    sets.append(f"updated_by = ${len(args)}")
                    args.append(update["data_category"])
                    await conn.execute(
                        f"UPDATE gdpr_retention_policies SET {', '.join(sets)} WHERE data_category = ${len(args)}",  # nosec B608 -- col iterates a hardcoded tuple above, column names never request-derived
                        *args,
                    )
        return await self.get_policies()

    # ── Enforcement (daily, via APScheduler) ────────────────────────────────

    async def enforce(self) -> dict[str, dict]:
        policies = await self.get_policies()
        results: dict[str, dict] = {}
        for policy in policies:
            if not policy.auto_delete or policy.data_category not in ENFORCEABLE_CATEGORIES:
                continue
            start = time.monotonic()
            try:
                rows_deleted = await self._enforce_one(policy.data_category, policy.retention_days)
                duration_ms = int((time.monotonic() - start) * 1000)
                await self._log_run(policy.data_category, rows_deleted, duration_ms)
                results[policy.data_category] = {"rows_deleted": rows_deleted, "duration_ms": duration_ms}
            except Exception as exc:  # noqa: BLE001 -- per-category enforcement spans httpx or asyncpg depending on data category; already logged and recorded with error detail preserved
                duration_ms = int((time.monotonic() - start) * 1000)
                logger.error("retention enforce: category=%s failed: %s", policy.data_category, exc)
                await self._log_run(policy.data_category, -1, duration_ms)
                results[policy.data_category] = {"rows_deleted": -1, "duration_ms": duration_ms, "error": str(exc)}
        self.last_run = datetime.now(timezone.utc)
        return results

    async def _enforce_one(self, category: str, days: int) -> int:
        if category in _HTTP_CATEGORIES:
            resp = await self.http.delete(
                f"{settings.module_26_url}/internal/gdpr/memory/sessions/expired",
                headers=_INTERNAL_HEADERS, params={"older_than_days": days},
            )
            resp.raise_for_status()
            return resp.json()["sessions_deleted"]

        table, column, extra_where = _SQL_CATEGORIES[category]
        sql = _delete_sql(table, column, days, extra_where)
        db_name = _db_for_category(category)
        if db_name is None:
            async with self.pool.acquire() as conn:
                result = await conn.execute(sql)
        else:
            conn = await asyncpg.connect(
                host=settings.postgres_host, port=settings.postgres_port,
                user=settings.postgres_user, password=settings.postgres_password,
                database=db_name,
            )
            try:
                result = await conn.execute(sql)
            finally:
                await conn.close()
        # asyncpg's execute() returns a command tag like "DELETE 42".
        return int(result.split()[-1])

    async def purge_audit_events(self) -> dict[str, dict]:
        """Delete every CF-4 audit event past its own retention_days, in
        each table of _audit_event_tables(); same per-table logging and
        rows_deleted=-1-on-failure as enforce()."""
        results: dict[str, dict] = {}
        for category, db_name in _audit_event_tables():
            start = time.monotonic()
            try:
                if db_name is None:
                    async with self.pool.acquire() as conn:
                        rows_deleted = await _purge_audit_events_on(conn)
                else:
                    conn = await asyncpg.connect(
                        host=settings.postgres_host, port=settings.postgres_port,
                        user=settings.postgres_user, password=settings.postgres_password,
                        database=db_name,
                    )
                    try:
                        rows_deleted = await _purge_audit_events_on(conn)
                    finally:
                        await conn.close()
                duration_ms = int((time.monotonic() - start) * 1000)
                await self._log_run(category, rows_deleted, duration_ms)
                results[category] = {"rows_deleted": rows_deleted, "duration_ms": duration_ms}
            except (asyncpg.PostgresError, asyncpg.InterfaceError, OSError) as exc:
                duration_ms = int((time.monotonic() - start) * 1000)
                logger.error("audit_events purge: %s failed: %s", category, exc)
                await self._log_run(category, -1, duration_ms)
                results[category] = {"rows_deleted": -1, "duration_ms": duration_ms, "error": str(exc)}
        return results

    async def _log_run(self, category: str, rows_deleted: int, duration_ms: int) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO gdpr_retention_log (run_at, data_category, rows_deleted, duration_ms) VALUES ($1, $2, $3, $4)",
                datetime.now(timezone.utc), category, rows_deleted, duration_ms,
            )

    # ── Dry run (read-only) ──────────────────────────────────────────────

    async def dry_run(self) -> dict[str, int]:
        policies = await self.get_policies()
        counts: dict[str, int] = {}
        for policy in policies:
            if not policy.auto_delete or policy.data_category not in ENFORCEABLE_CATEGORIES:
                continue
            counts[policy.data_category] = await self._count_one(policy.data_category, policy.retention_days)
        return counts

    async def _count_one(self, category: str, days: int) -> int:
        if category in _HTTP_CATEGORIES:
            resp = await self.http.delete(
                f"{settings.module_26_url}/internal/gdpr/memory/sessions/expired",
                headers=_INTERNAL_HEADERS, params={"older_than_days": days, "dry_run": "true"},
            )
            resp.raise_for_status()
            return resp.json()["sessions_deleted"]

        table, column, extra_where = _SQL_CATEGORIES[category]
        sql = _count_sql(table, column, days, extra_where)
        db_name = _db_for_category(category)
        if db_name is None:
            async with self.pool.acquire() as conn:
                return await conn.fetchval(sql)
        conn = await asyncpg.connect(
            host=settings.postgres_host, port=settings.postgres_port,
            user=settings.postgres_user, password=settings.postgres_password,
            database=db_name,
        )
        try:
            return await conn.fetchval(sql)
        finally:
            await conn.close()
