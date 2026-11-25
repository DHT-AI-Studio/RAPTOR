"""CM-7 / VIE01-410: compliance audit report.

One report for a period, built from the four logs the ticket names:

  section               source table (database)                       time column
  guardrail_violations  guardrail_violations + its audit_events row    created_at
                        (guardrails, Module 23)
  moderation_flags      moderation_flags (guardrails, Module 23, CM-5) created_at
  gdpr_jobs             gdpr_erasure_jobs (privacy_compliance):         requested_at
                        erasure, export, DSAR, consent withdrawal
  gdpr_rights_requests  audit_events gdpr.* (privacy_compliance):       timestamp
                        rectification, restriction, objection, consent,
                        legal hold -- the GDPR requests that never make a job
  access_log            gdpr_access_log (privacy_compliance, GD-7)      timestamp

The guardrails database is read with direct SQL on the same Postgres server,
like the DSAR assembler and the retention engine already do -- Module 23 has
no endpoint for this and needs none.

Privacy: every person id (user, requester, accessor, a flag's branch_id) is
reported only as its SHA-256 hex digest -- raptor_audit.hash_value, the same
hash CF-4's audit_events.actor_id_hash holds, so report rows can be matched to
audit events. A flag's asset_path (a file name the user chose) is hashed the
same way. No content is read: no text, no request ids (caller-chosen), no job
notes, no restriction reasons, and an access-log endpoint has any id-like path
segment replaced by a placeholder.

Each section has exact counts for the whole period (SQL aggregates) and up to
`max_rows` event rows, oldest first; `truncated` says when there were more. A
section whose table doesn't exist yet (e.g. moderation_flags before Module 23
has CM-5) is reported as unavailable, with the reason, rather than as empty.
"""
from __future__ import annotations

import csv
import io
import json
import re
from collections import Counter
from datetime import date, datetime, timezone
from typing import Any

import asyncpg
from raptor_audit import hash_value

SECTIONS = ("guardrail_violations", "moderation_flags", "gdpr_jobs", "gdpr_rights_requests", "access_log")

# The GDPR requests recorded only as Module 33 audit events. gdpr.erasure,
# gdpr.export and gdpr.access are left out: each of those is a job already
# counted in gdpr_jobs.
RIGHTS_EVENT_TYPES = ("gdpr.rectification", "gdpr.restriction", "gdpr.objection", "gdpr.consent", "gdpr.legal_hold")
# The only metadata keys a rights event row carries over (all non-sensitive).
_RIGHTS_METADATA = ("action", "purpose")

CSV_COLUMNS = ("section", "occurred_at", "record_id", "actor_hash", "subject_hash", "kind", "category",
               "outcome", "status", "module", "details")

_UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_LONG_HEX = re.compile(r"^[0-9a-fA-F]{32,}$")
_CSV_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


# ── small helpers ────────────────────────────────────────────────────────────


def utc(value: datetime) -> datetime:
    """A naive datetime is taken as UTC (as asyncpg does for timestamptz)."""
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _iso(value: datetime | date | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return utc(value).isoformat()
    return value.isoformat()


def _str(value: Any) -> str | None:
    return None if value is None else str(value)


def scrub_endpoint(endpoint: str, *raw_ids: str | None) -> str:
    """The access log's endpoint with every path segment that names someone
    replaced: the row's own user/accessor id -> {user_id}, any other UUID or
    long hex id -> {id}. Query strings are dropped (the log stores paths, but
    never trust that)."""
    path = endpoint.split("?", 1)[0]
    known = {i for i in raw_ids if i}
    parts = []
    for segment in path.split("/"):
        if segment in known:
            parts.append("{user_id}")
        elif _UUID.match(segment) or _LONG_HEX.match(segment):
            parts.append("{id}")
        else:
            parts.append(segment)
    return "/".join(parts)


def _json_list(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


# ── row mapping (raw DB row -> report row, ids hashed, content dropped) ──────


def violation_row(r: dict) -> dict:
    return {
        "id": _str(r["id"]), "occurred_at": _iso(r["created_at"]), "module": r["module"],
        "direction": r["direction"], "category": r["category"], "action_taken": r["action_taken"],
        "policy_id": _str(r["policy_id"]), "policy_version": r.get("policy_version"),
        "actor_hash": r.get("actor_id_hash"), "event_id": _str(r["event_id"]),
    }


def flag_row(r: dict) -> dict:
    scope = "moment" if r["moment_id"] else ("chunk" if r["chunk_id"] else "asset")
    return {
        "id": _str(r["id"]), "occurred_at": _iso(r["created_at"]), "updated_at": _iso(r["updated_at"]),
        "status": r["status"], "category": r["category"], "score": r["score"], "model": r["model"],
        "policy_version": r["policy_version"], "scope": scope,
        "owner_hash": hash_value(r["branch_id"]), "asset_hash": hash_value(r["asset_path"]),
        "version_id": r["version_id"], "moment_id": r["moment_id"], "chunk_id": r["chunk_id"],
        "event_id": _str(r["event_id"]),
    }


def job_row(r: dict) -> dict:
    statuses = _json_list(r["store_statuses"]) or {}
    return {
        "job_id": _str(r["job_id"]), "occurred_at": _iso(r["requested_at"]), "completed_at": _iso(r["completed_at"]),
        "type": r["type"], "status": r["status"],
        "subject_hash": hash_value(r["user_id"]), "requester_hash": hash_value(r["requested_by"]),
        "requested_by_subject": r["requested_by"] == r["user_id"],
        "failed_stores": sorted(k for k, v in statuses.items() if v == "failed"),
    }


def rights_row(r: dict) -> dict:
    meta = _json_list(r["metadata"]) or {}
    if meta.get("requested_by_system"):
        requested_by, requester = "system", None
    elif meta.get("requested_by_subject"):
        requested_by, requester = "subject", r["actor_id_hash"]
    else:
        requested_by, requester = "other", meta.get("requested_by_hash")
    return {
        "event_id": _str(r["event_id"]), "occurred_at": _iso(r["timestamp"]), "type": r["event_type"],
        "outcome": r["outcome"], "subject_hash": r["actor_id_hash"], "requester_hash": requester,
        "requested_by": requested_by, **{k: meta.get(k) for k in _RIGHTS_METADATA},
    }


def access_row(r: dict) -> dict:
    return {
        "id": _str(r["id"]), "occurred_at": _iso(r["timestamp"]),
        "accessor_hash": hash_value(r["accessor_id"]), "subject_hash": hash_value(r["user_id"]),
        "endpoint": scrub_endpoint(r["endpoint"], r["user_id"], r["accessor_id"]), "method": r["method"],
        "data_categories": list(r["data_categories"] or []), "purpose": r["purpose"], "ip_hash": r["ip_hash"],
    }


# ── SQL (table/column names are fixed literals; values always parameters) ────

_WINDOW = "{ts} >= $1 AND {ts} <= $2"

_VIOLATIONS_ROWS = (
    "SELECT v.id, v.created_at, v.module, v.direction, v.category, v.action_taken, v.policy_id, v.event_id, "
    "e.actor_id_hash, e.policy_version FROM guardrail_violations v "
    "LEFT JOIN audit_events e ON e.event_id = v.event_id "
    "WHERE v.created_at >= $1 AND v.created_at <= $2 ORDER BY v.created_at, v.id LIMIT $3"
)
_FLAGS_ROWS = (
    "SELECT id, created_at, updated_at, status, category, score, model, policy_version, branch_id, asset_path, "
    "version_id, moment_id, chunk_id, event_id FROM moderation_flags "
    "WHERE created_at >= $1 AND created_at <= $2 ORDER BY created_at, id LIMIT $3"
)
_JOBS_ROWS = (
    "SELECT job_id, requested_at, completed_at, type, status, user_id, requested_by, store_statuses "
    "FROM gdpr_erasure_jobs WHERE requested_at >= $1 AND requested_at <= $2 ORDER BY requested_at, job_id LIMIT $3"
)
_RIGHTS_ROWS = (
    "SELECT event_id, timestamp, event_type, outcome, actor_id_hash, metadata FROM audit_events "
    "WHERE timestamp >= $1 AND timestamp <= $2 AND event_type = ANY($4) ORDER BY timestamp, event_id LIMIT $3"
)
_ACCESS_ROWS = (
    "SELECT id, timestamp, user_id, accessor_id, endpoint, method, data_categories, purpose, ip_hash "
    "FROM gdpr_access_log WHERE timestamp >= $1 AND timestamp <= $2 ORDER BY timestamp, id LIMIT $3"
)


async def _grouped(conn, table: str, ts: str, columns: tuple[str, ...], start, end, extra: str = "",
                   extra_args: tuple = ()) -> list[dict]:
    """[{col: value, ..., "count": n}] for the period, most frequent first."""
    cols = ", ".join(columns)
    where = _WINDOW.format(ts=ts) + extra
    rows = await conn.fetch(
        f"SELECT {cols}, count(*) AS count FROM {table} WHERE {where} "  # nosec B608 -- table/columns/ts/extra are this module's fixed literals; values are parameters
        f"GROUP BY {cols} ORDER BY count DESC, {cols}",
        start, end, *extra_args,
    )
    return [dict(r) for r in rows]


async def _by_day(conn, table: str, ts: str, start, end, extra: str = "", extra_args: tuple = ()) -> list[dict]:
    where = _WINDOW.format(ts=ts) + extra
    rows = await conn.fetch(
        f"SELECT ({ts} AT TIME ZONE 'UTC')::date AS day, count(*) AS count FROM {table} "  # nosec B608 -- fixed literals as above
        f"WHERE {where} GROUP BY day ORDER BY day",
        start, end, *extra_args,
    )
    return [{"day": r["day"].isoformat(), "count": r["count"]} for r in rows]


async def _total(conn, table: str, ts: str, start, end, extra: str = "", extra_args: tuple = ()) -> int:
    where = _WINDOW.format(ts=ts) + extra
    return await conn.fetchval(f"SELECT count(*) FROM {table} WHERE {where}", start, end, *extra_args)  # nosec B608 -- fixed literals as above


def _section(total: int, rows: list[dict], summary: dict) -> dict:
    return {"total": total, "rows_included": len(rows), "truncated": total > len(rows), "summary": summary,
            "rows": rows}


async def violations_section(conn, start, end, max_rows: int) -> dict:
    t, ts = "guardrail_violations", "created_at"
    rows = await conn.fetch(_VIOLATIONS_ROWS, start, end, max_rows)
    return _section(await _total(conn, t, ts, start, end), [violation_row(dict(r)) for r in rows], {
        "by_category_and_action": await _grouped(conn, t, ts, ("category", "action_taken"), start, end),
        "by_module": await _grouped(conn, t, ts, ("module",), start, end),
        "by_direction": await _grouped(conn, t, ts, ("direction",), start, end),
        "by_day": await _by_day(conn, t, ts, start, end),
    })


async def flags_section(conn, start, end, max_rows: int) -> dict:
    t, ts = "moderation_flags", "created_at"
    rows = await conn.fetch(_FLAGS_ROWS, start, end, max_rows)
    return _section(await _total(conn, t, ts, start, end), [flag_row(dict(r)) for r in rows], {
        "by_category": await _grouped(conn, t, ts, ("category",), start, end),
        "by_status": await _grouped(conn, t, ts, ("status",), start, end),
        "by_model": await _grouped(conn, t, ts, ("model",), start, end),
        "by_day": await _by_day(conn, t, ts, start, end),
    })


async def jobs_section(conn, start, end, max_rows: int) -> dict:
    t, ts = "gdpr_erasure_jobs", "requested_at"
    rows = await conn.fetch(_JOBS_ROWS, start, end, max_rows)
    return _section(await _total(conn, t, ts, start, end), [job_row(dict(r)) for r in rows], {
        "by_type_and_status": await _grouped(conn, t, ts, ("type", "status"), start, end),
        "by_day": await _by_day(conn, t, ts, start, end),
    })


async def rights_section(conn, start, end, max_rows: int) -> dict:
    t, ts, only = "audit_events", "timestamp", (" AND event_type = ANY($3)", (list(RIGHTS_EVENT_TYPES),))
    rows = await conn.fetch(_RIGHTS_ROWS, start, end, max_rows, list(RIGHTS_EVENT_TYPES))
    mapped = [rights_row(dict(r)) for r in rows]
    by_action = await conn.fetch(
        "SELECT event_type, metadata->>'action' AS action, outcome, count(*) AS count FROM audit_events "
        "WHERE timestamp >= $1 AND timestamp <= $2 AND event_type = ANY($3) "
        "GROUP BY event_type, action, outcome ORDER BY count DESC, event_type, action, outcome",
        start, end, list(RIGHTS_EVENT_TYPES),
    )
    return _section(await _total(conn, t, ts, start, end, *only), mapped, {
        "by_type_action_and_outcome": [dict(r) for r in by_action],
        "by_day": await _by_day(conn, t, ts, start, end, *only),
    })


async def access_section(conn, start, end, max_rows: int) -> dict:
    t, ts = "gdpr_access_log", "timestamp"
    rows = await conn.fetch(_ACCESS_ROWS, start, end, max_rows)
    # Endpoints are grouped after scrubbing, so ids in paths never reach the summary.
    raw = await conn.fetch(
        "SELECT endpoint, method, user_id, accessor_id, count(*) AS count FROM gdpr_access_log "
        "WHERE timestamp >= $1 AND timestamp <= $2 GROUP BY endpoint, method, user_id, accessor_id",
        start, end,
    )
    endpoints: Counter = Counter()
    for r in raw:
        endpoints[(r["method"], scrub_endpoint(r["endpoint"], r["user_id"], r["accessor_id"]))] += r["count"]
    distinct = await conn.fetchrow(
        "SELECT count(DISTINCT accessor_id) AS accessors, count(DISTINCT user_id) AS subjects "
        "FROM gdpr_access_log WHERE timestamp >= $1 AND timestamp <= $2",
        start, end,
    )
    return _section(await _total(conn, t, ts, start, end), [access_row(dict(r)) for r in rows], {
        "distinct_accessors": distinct["accessors"], "distinct_subjects": distinct["subjects"],
        "by_purpose": await _grouped(conn, t, ts, ("purpose",), start, end),
        "by_endpoint": [{"method": m, "endpoint": e, "count": n}
                        for (m, e), n in sorted(endpoints.items(), key=lambda kv: (-kv[1], kv[0]))],
        "by_day": await _by_day(conn, t, ts, start, end),
    })


async def _available(build, conn, start, end, max_rows) -> dict:
    """The section, or -- when its table doesn't exist in that database -- an
    explicit "unavailable" entry. Any other database error propagates: a
    report must never quietly leave data out."""
    try:
        return {"available": True, **await build(conn, start, end, max_rows)}
    except asyncpg.UndefinedTableError as exc:
        return {"available": False, "error": str(exc), "total": 0, "rows_included": 0, "truncated": False,
                "summary": {}, "rows": []}


async def build_report(privacy_conn, guardrails_conn, start: datetime, end: datetime, max_rows: int) -> dict:
    """The whole report; `privacy_conn` is on privacy_compliance, `guardrails_conn` on guardrails."""
    start, end = utc(start), utc(end)
    sections = {
        "guardrail_violations": await _available(violations_section, guardrails_conn, start, end, max_rows),
        "moderation_flags": await _available(flags_section, guardrails_conn, start, end, max_rows),
        "gdpr_jobs": await _available(jobs_section, privacy_conn, start, end, max_rows),
        "gdpr_rights_requests": await _available(rights_section, privacy_conn, start, end, max_rows),
        "access_log": await _available(access_section, privacy_conn, start, end, max_rows),
    }
    return {
        "report": "compliance_audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "period": {"from": start.isoformat(), "to": end.isoformat()},
        "max_rows_per_section": max_rows,
        "id_hashing": "SHA-256 hex of the raw id (same as CF-4 audit_events.actor_id_hash)",
        "totals": {name: s["total"] for name, s in sections.items()},
        "unavailable_sections": [name for name, s in sections.items() if not s["available"]],
        "sections": sections,
    }


# ── CSV ──────────────────────────────────────────────────────────────────────


def _csv_records(report: dict) -> list[dict]:
    """One flat record per event row, in CSV_COLUMNS; section-specific fields
    go to `details` (compact JSON)."""
    s = report["sections"]
    out: list[dict] = []
    for r in s["guardrail_violations"]["rows"]:
        out.append({"section": "guardrail_violations", "occurred_at": r["occurred_at"], "record_id": r["id"],
                    "actor_hash": r["actor_hash"], "kind": r["direction"], "category": r["category"],
                    "outcome": r["action_taken"], "module": r["module"],
                    "details": {k: r[k] for k in ("policy_id", "policy_version", "event_id")}})
    for r in s["moderation_flags"]["rows"]:
        out.append({"section": "moderation_flags", "occurred_at": r["occurred_at"], "record_id": r["id"],
                    "subject_hash": r["owner_hash"], "kind": r["scope"], "category": r["category"],
                    "status": r["status"],
                    "details": {k: r[k] for k in ("score", "model", "policy_version", "asset_hash", "version_id",
                                                  "moment_id", "chunk_id", "updated_at", "event_id")}})
    for r in s["gdpr_jobs"]["rows"]:
        out.append({"section": "gdpr_jobs", "occurred_at": r["occurred_at"], "record_id": r["job_id"],
                    "actor_hash": r["requester_hash"], "subject_hash": r["subject_hash"], "kind": r["type"],
                    "status": r["status"], "module": "33",
                    "details": {k: r[k] for k in ("completed_at", "requested_by_subject", "failed_stores")}})
    for r in s["gdpr_rights_requests"]["rows"]:
        out.append({"section": "gdpr_rights_requests", "occurred_at": r["occurred_at"], "record_id": r["event_id"],
                    "actor_hash": r["requester_hash"], "subject_hash": r["subject_hash"], "kind": r["type"],
                    "category": r["action"], "outcome": r["outcome"], "module": "33",
                    "details": {k: r[k] for k in ("requested_by", "purpose")}})
    for r in s["access_log"]["rows"]:
        out.append({"section": "access_log", "occurred_at": r["occurred_at"], "record_id": r["id"],
                    "actor_hash": r["accessor_hash"], "subject_hash": r["subject_hash"], "kind": r["method"],
                    "category": r["purpose"], "module": "13",
                    "details": {k: r[k] for k in ("endpoint", "data_categories", "ip_hash")}})
    return out


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, separators=(",", ":"), sort_keys=True, default=str)
    text = str(value)
    # A cell a spreadsheet would read as a formula is prefixed with ' (CSV injection).
    return "'" + text if text.startswith(_CSV_FORMULA_START) else text


def to_csv(report: dict) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\r\n")
    writer.writerow(CSV_COLUMNS)
    for record in _csv_records(report):
        writer.writerow([_cell(record.get(col)) for col in CSV_COLUMNS])
    return buf.getvalue()
