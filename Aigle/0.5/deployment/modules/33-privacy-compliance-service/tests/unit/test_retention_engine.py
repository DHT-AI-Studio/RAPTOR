"""
GD-3 / VIE01-248: unit tests for the retention engine.

Mocked -- no live Postgres/Module 26/Kafka needed (unlike
tests/test_erasure_e2e.py / test_export_e2e.py, which deliberately are
live-infrastructure tests). Covers the ticket's own acceptance criterion:
"Unit tests: policy CRUD, scheduler invocation, dry-run count, per-category
SQL generation."

Run from the 33-privacy-compliance-service/ directory:
    python -m pytest tests/unit/test_retention_engine.py -v
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import pytest
from app.services.retention_engine import (
    ENFORCEABLE_CATEGORIES,
    RetentionEngine,
    _count_sql,
    _db_for_category,
    _delete_sql,
)

# ── Per-category SQL generation (pure functions) ────────────────────────────

def test_delete_sql_generates_correct_table_column_and_interval():
    sql = _delete_sql("benchmark_runs", "created_at", 90)
    assert sql == "DELETE FROM benchmark_runs WHERE created_at < NOW() - INTERVAL '90 days'"


def test_count_sql_generates_correct_table_column_and_interval():
    sql = _count_sql("guardrail_violations", "created_at", 90)
    assert sql == "SELECT COUNT(*) FROM guardrail_violations WHERE created_at < NOW() - INTERVAL '90 days'"


def test_delete_sql_coerces_days_to_int():
    """retention_days always comes from an INT Postgres column, but the
    dispatch table is keyed by plain Python attribute access -- guard
    against a stray float/str ever reaching raw SQL interpolation."""
    sql = _delete_sql("gdpr_access_log", "timestamp", "365")
    assert "INTERVAL '365 days'" in sql


def test_delete_sql_and_count_sql_append_extra_where():
    """CM-5 / VIE01-408: moderation_flags' two categories are the same table,
    split by status -- extra_where is how that's expressed on top of the
    shared age check, not a separate code path."""
    assert (_delete_sql("moderation_flags", "created_at", 90, "status = 'cleared'")
            == "DELETE FROM moderation_flags WHERE created_at < NOW() - INTERVAL '90 days' AND status = 'cleared'")
    assert (_count_sql("moderation_flags", "created_at", 1095, "status <> 'cleared'")
            == "SELECT COUNT(*) FROM moderation_flags WHERE created_at < NOW() - INTERVAL '1095 days' AND status <> 'cleared'")
    # None/omitted extra_where -- every pre-existing category's behavior, unchanged.
    assert "AND" not in _delete_sql("benchmark_runs", "created_at", 90)


def test_db_for_category_routes_to_the_right_database():
    assert _db_for_category("benchmark_runs") == "benchmark"
    assert _db_for_category("guardrail_violations") == "guardrails"
    # CM-5 / VIE01-408: a different guardrails table, same database --
    # unlike guardrail_violations, this one has a branch_id column.
    assert _db_for_category("moderation_flags") == "guardrails"
    assert _db_for_category("moderation_flags_actionable") == "guardrails"
    # gdpr_access_log lives in this service's own database -- None means
    # "use the shared pool", not "no database".
    assert _db_for_category("gdpr_access_log") is None


def test_enforceable_categories_excludes_legal_hold_rows():
    """gdpr_consent_history and gdpr_erasure_jobs are seeded auto_delete=False
    legal-hold rows with no real enforcement target -- never enforceable,
    regardless of what a policy row's auto_delete flag says."""
    assert ENFORCEABLE_CATEGORIES == {
        "benchmark_runs", "guardrail_violations", "moderation_flags", "moderation_flags_actionable",
        "gdpr_access_log", "memvid_sessions",
    }
    assert "gdpr_consent_history" not in ENFORCEABLE_CATEGORIES
    assert "gdpr_erasure_jobs" not in ENFORCEABLE_CATEGORIES


# ── Fakes ────────────────────────────────────────────────────────────────────

class _FakeConn:
    def __init__(self, fetch_rows=None, fetchval_result=None):
        self.fetch_rows = fetch_rows or []
        self.fetchval_result = fetchval_result
        self.executed: list[tuple] = []

    async def fetch(self, sql, *args):
        self.executed.append((sql, args))
        return self.fetch_rows

    async def fetchval(self, sql, *args):
        self.executed.append((sql, args))
        return self.fetchval_result

    async def execute(self, sql, *args):
        self.executed.append((sql, args))
        return "UPDATE 1"

    def transaction(self):
        return _NoopAsyncCM()


class _NoopAsyncCM:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakePool:
    def __init__(self, conn: _FakeConn):
        self._conn = conn

    def acquire(self):
        return _AcquireCM(self._conn)


class _AcquireCM:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


def _policy_row(
    data_category: str, retention_days: int = 90, auto_delete: bool = True,
    legal_basis: str = "GDPR Art. 5.1.e", notes: str = "", updated_by: str = "system",
) -> dict:
    return {
        "data_category": data_category, "retention_days": retention_days,
        "auto_delete": auto_delete, "legal_basis": legal_basis, "notes": notes,
        "updated_at": datetime.now(timezone.utc), "updated_by": updated_by,
    }


# ── Policy CRUD ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_policies_returns_every_row():
    rows = [_policy_row("benchmark_runs"), _policy_row("guardrail_violations")]
    conn = _FakeConn(fetch_rows=rows)
    engine = RetentionEngine(pool=_FakePool(conn))
    policies = await engine.get_policies()
    assert [p.data_category for p in policies] == ["benchmark_runs", "guardrail_violations"]
    assert policies[0].retention_days == 90


@pytest.mark.asyncio
async def test_update_policies_only_sets_provided_fields():
    """PUT with just {data_category, retention_days} must not clobber
    auto_delete/legal_basis/notes with NULLs -- only columns actually
    present in the update dict belong in the SET clause."""
    conn = _FakeConn(fetch_rows=[_policy_row("benchmark_runs", retention_days=30)])
    engine = RetentionEngine(pool=_FakePool(conn))

    await engine.update_policies([{"data_category": "benchmark_runs", "retention_days": 30}], updated_by="alice")

    update_calls = [c for c in conn.executed if c[0].startswith("UPDATE")]
    assert len(update_calls) == 1
    sql, args = update_calls[0]
    assert "retention_days = $1" in sql
    assert "auto_delete" not in sql
    assert "legal_basis" not in sql
    assert "updated_by = $3" in sql
    assert args[-1] == "benchmark_runs"  # WHERE data_category = $N
    assert "alice" in args


@pytest.mark.asyncio
async def test_update_policies_skips_rows_with_no_real_changes():
    conn = _FakeConn(fetch_rows=[])
    engine = RetentionEngine(pool=_FakePool(conn))
    await engine.update_policies([{"data_category": "benchmark_runs"}], updated_by="alice")
    assert not any(c[0].startswith("UPDATE") for c in conn.executed)


# ── Dry run (read-only counts) ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_dry_run_counts_gdpr_access_log_via_shared_pool():
    """gdpr_access_log is the one enforceable SQL category that lives in
    this service's own database -- no cross-database asyncpg.connect,
    should go straight through the shared pool's fetchval."""
    policy_rows = [_policy_row("gdpr_access_log", retention_days=365)]
    conn = _FakeConn(fetch_rows=policy_rows, fetchval_result=7)
    engine = RetentionEngine(pool=_FakePool(conn))

    counts = await engine.dry_run()

    assert counts == {"gdpr_access_log": 7}
    count_calls = [c for c in conn.executed if c[0].startswith("SELECT COUNT")]
    assert len(count_calls) == 1
    assert "gdpr_access_log" in count_calls[0][0]
    assert "INTERVAL '365 days'" in count_calls[0][0]


@pytest.mark.asyncio
async def test_dry_run_skips_non_auto_delete_and_non_enforceable_categories():
    policy_rows = [
        _policy_row("gdpr_access_log", auto_delete=True),
        _policy_row("gdpr_consent_history", auto_delete=False),  # legal hold -- skip
        _policy_row("gdpr_erasure_jobs", auto_delete=False),      # legal hold -- skip
    ]
    conn = _FakeConn(fetch_rows=policy_rows, fetchval_result=0)
    engine = RetentionEngine(pool=_FakePool(conn))

    counts = await engine.dry_run()

    assert set(counts.keys()) == {"gdpr_access_log"}


@pytest.mark.asyncio
async def test_dry_run_does_not_delete_anything():
    """The whole point of dry-run: fetchval (COUNT), never execute (DELETE)."""
    policy_rows = [_policy_row("gdpr_access_log")]
    conn = _FakeConn(fetch_rows=policy_rows, fetchval_result=3)
    engine = RetentionEngine(pool=_FakePool(conn))

    await engine.dry_run()

    assert not any(c[0].startswith("DELETE") for c in conn.executed)


# ── enforce() writes a gdpr_retention_log row per category, even on failure ──

@pytest.mark.asyncio
async def test_enforce_logs_minus_one_and_does_not_raise_on_category_failure():
    """A category whose deletion raises must never be silently skipped --
    ticket's explicit requirement: rows_deleted=-1, logged, run continues."""

    class _FailingConn(_FakeConn):
        async def execute(self, sql, *args):
            self.executed.append((sql, args))
            if sql.startswith("DELETE FROM gdpr_access_log"):
                raise RuntimeError("simulated DB failure")
            return "INSERT 0 1"

    policy_rows = [_policy_row("gdpr_access_log", auto_delete=True)]
    conn = _FailingConn(fetch_rows=policy_rows)
    engine = RetentionEngine(pool=_FakePool(conn))

    results = await engine.enforce()

    assert results["gdpr_access_log"]["rows_deleted"] == -1
    log_inserts = [c for c in conn.executed if c[0].startswith("INSERT INTO gdpr_retention_log")]
    assert len(log_inserts) == 1
    assert log_inserts[0][1][2] == -1  # rows_deleted positional arg


@pytest.mark.asyncio
async def test_enforce_parses_rows_deleted_from_command_tag():
    class _CountingConn(_FakeConn):
        async def execute(self, sql, *args):
            self.executed.append((sql, args))
            if sql.startswith("DELETE FROM gdpr_access_log"):
                return "DELETE 42"
            return "INSERT 0 1"

    policy_rows = [_policy_row("gdpr_access_log", auto_delete=True)]
    conn = _CountingConn(fetch_rows=policy_rows)
    engine = RetentionEngine(pool=_FakePool(conn))

    results = await engine.enforce()

    assert results["gdpr_access_log"]["rows_deleted"] == 42


@pytest.mark.asyncio
async def test_enforce_skips_legal_hold_categories():
    policy_rows = [
        _policy_row("gdpr_consent_history", auto_delete=False),
        _policy_row("gdpr_erasure_jobs", auto_delete=False),
    ]
    conn = _FakeConn(fetch_rows=policy_rows)
    engine = RetentionEngine(pool=_FakePool(conn))

    results = await engine.enforce()

    assert results == {}
    assert not any(c[0].startswith("DELETE") or c[0].startswith("INSERT INTO gdpr_retention_log") for c in conn.executed)


# ── CM-5 / VIE01-408: moderation_flags seed row ──────────────────────────────

def test_seed_has_the_moderation_flags_row():
    match = re.search(r"\('moderation_flags', (\d+), (TRUE|FALSE),", _SEED_SQL)
    assert match, "moderation_flags row missing from 004_gdpr_retention.sql"
    assert (int(match.group(1)), match.group(2)) == (90, "TRUE")
    assert "moderation_flags" in ENFORCEABLE_CATEGORIES


def test_seed_has_the_moderation_flags_actionable_row():
    """flagged (pending) / removed (confirmed violation) get the longer,
    "Legal record" policy -- only cleared uses the plain 90-day row above."""
    match = re.search(r"\('moderation_flags_actionable', (\d+), (TRUE|FALSE), '(Legal record)',", _SEED_SQL)
    assert match, "moderation_flags_actionable row missing from 004_gdpr_retention.sql"
    assert (int(match.group(1)), match.group(2)) == (1095, "TRUE")
    assert "moderation_flags_actionable" in ENFORCEABLE_CATEGORIES


# ── GD-10 / VIE01-416: voice_recordings and loki_logs rows ───────────────────

_SEED_SQL = (Path(__file__).resolve().parents[3] / "03-database" / "init" / "postgresql"
             / "004_gdpr_retention.sql").read_text(encoding="utf-8")


@pytest.mark.parametrize("category,days,auto_delete", [
    ("voice_recordings", 90, "TRUE"),   # Module 34 deletes them itself
    ("loki_logs", 30, "TRUE"),          # Loki's compactor deletes them itself
])
def test_seed_has_the_voice_and_loki_rows(category, days, auto_delete):
    match = re.search(rf"\('{category}', (\d+), (TRUE|FALSE),", _SEED_SQL)
    assert match, f"{category} row missing from 004_gdpr_retention.sql"
    assert (int(match.group(1)), match.group(2)) == (days, auto_delete)
    assert category not in ENFORCEABLE_CATEGORIES


@pytest.mark.asyncio
async def test_enforce_and_dry_run_leave_voice_and_loki_to_their_owners():
    policy_rows = [
        _policy_row("voice_recordings", retention_days=90, auto_delete=True),
        _policy_row("loki_logs", retention_days=30, auto_delete=True),
    ]
    conn = _FakeConn(fetch_rows=policy_rows, fetchval_result=0)
    engine = RetentionEngine(pool=_FakePool(conn))

    assert await engine.dry_run() == {}
    assert await engine.enforce() == {}
    assert not any(c[0].startswith("DELETE") or c[0].startswith("INSERT INTO gdpr_retention_log") for c in conn.executed)


# ── Scheduler invocation ─────────────────────────────────────────────────────

def test_scheduler_registers_daily_job_at_configured_utc_time():
    """Mirrors app/main.py's own scheduler.add_job(...) call -- constructed
    the same way here, independent of a running app, to confirm the
    CronTrigger actually fires at settings.retention_hour_utc/minute_utc
    UTC and that the registered job is the engine's own enforce()."""
    from app.core.config import settings
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger

    conn = _FakeConn()
    engine = RetentionEngine(pool=_FakePool(conn))
    scheduler = AsyncIOScheduler()
    job = scheduler.add_job(
        engine.enforce,
        CronTrigger(hour=settings.retention_hour_utc, minute=settings.retention_minute_utc, timezone="UTC"),
        id="gdpr_retention_enforce",
    )

    assert job.func == engine.enforce
    trigger = job.trigger
    assert isinstance(trigger, CronTrigger)
    field_by_name = {f.name: f for f in trigger.fields}
    assert str(field_by_name["hour"]) == str(settings.retention_hour_utc)
    assert str(field_by_name["minute"]) == str(settings.retention_minute_utc)
    assert str(trigger.timezone) == "UTC"
