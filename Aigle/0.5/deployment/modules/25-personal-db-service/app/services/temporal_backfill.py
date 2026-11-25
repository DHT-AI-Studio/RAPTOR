"""Backfill canonical dates onto existing TemporalFacts (TMP-3; temporal plan §5.8, P03).

Facts written before TMP-3 hold time_start / time_end as whatever string the extractor
produced ("2011", "2011-01-01", "2011-01-01T00:00:00", "null") and no ranges, so the TKG
window can only compare them as strings (searcher's fallback for rows with no s_state).
This rewrites each one through the same normaliser the write path uses:

- the original strings are kept in time_start_raw / time_end_raw, never altered;
- year-only "2011" becomes the range of all of 2011, never the day 2011-01-01;
- an unparseable value is quarantined -- the endpoint becomes unknown,
  quarantine_reason says why, precision is "unknown" and requires_reextraction is
  set (P03) -- and the row is kept, never dropped.

Checkpointed and re-runnable: each row is updated on its own and stamped with
migration_version = NORMALIZER_VERSION, which is the checkpoint -- the next page, or the
next run after an interruption, selects only rows not yet at the target version. A run
over a fully migrated database changes nothing. It does not read PD_TEMPORAL_V2: it runs
the same with the flag off, because v1 reads need the canonical dates too (D3).

Run inside the Module 25 container:
    python -m app.services.temporal_backfill --all            # every user_* database
    python -m app.services.temporal_backfill --branch <sub>   # one user
    python -m app.services.temporal_backfill --all --dry-run  # count only
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from app.core.config import settings
from app.services.arcadedb_client import ArcadeDBClient, db_name_for
from app.services.schema_init import initialize_schema
from app.services.temporal_normalize import NORMALIZER_VERSION, normalize_fact_dates

logger = logging.getLogger("personal_db.temporal_backfill")

# Fixed SQL, no string building: every value is a bound parameter. The row is addressed by
# fact_id (unique index); the SET list is every property normalize_fact_dates returns.
_COUNT_PENDING = ("SELECT count(*) AS n FROM TemporalFact "
                  "WHERE (migration_version IS NULL OR migration_version <> :target) "
                  "AND (derived IS NULL OR derived = false)")   # P15: a derived fact's bounds come from reasoning
_PAGE = ("SELECT fact_id, time_start, time_end, time_start_raw, time_end_raw, migration_version "
         "FROM TemporalFact WHERE (migration_version IS NULL OR migration_version <> :target) "
         "AND (derived IS NULL OR derived = false) ORDER BY fact_id LIMIT :n")
_DATE_PROPS = ("time_start", "time_end", "time_start_raw", "time_end_raw", "s_state", "e_state",
               "s_lo", "s_hi", "e_lo", "e_hi", "precision", "timezone", "timeline_id", "axis", "time_unit",
               "normalizer_version",
               "migration_version", "quarantine_reason", "requires_reextraction", "reextraction_reason")
_UPDATE = ("UPDATE TemporalFact SET time_start = :time_start, time_end = :time_end, "
           "time_start_raw = :time_start_raw, time_end_raw = :time_end_raw, "
           "s_state = :s_state, e_state = :e_state, s_lo = :s_lo, s_hi = :s_hi, e_lo = :e_lo, "
           "e_hi = :e_hi, precision = :precision, timezone = :timezone, timeline_id = :timeline_id, "
           "axis = :axis, time_unit = :time_unit, "
           "normalizer_version = :normalizer_version, migration_version = :migration_version, "
           "quarantine_reason = :quarantine_reason, requires_reextraction = :requires_reextraction, "
           "reextraction_reason = :reextraction_reason "
           "WHERE fact_id = :fact_id")
if set(normalize_fact_dates(None, None, "UTC")) != set(_DATE_PROPS):      # checked once, at import
    raise RuntimeError("temporal_backfill._UPDATE is out of step with normalize_fact_dates()")


@dataclass
class BackfillStats:
    database: str
    migrated: int = 0
    quarantined: int = 0
    pending_before: int = 0
    pending_after: int = 0
    errors: list[str] = field(default_factory=list)


def normalize_legacy_row(row: dict[str, Any], tz_name: str) -> dict[str, Any]:
    """One stored TemporalFact -> the date properties to write back.

    The values to normalise are the raw ones when a previous run kept them (time_start /
    time_end are by then projections), else the legacy strings themselves."""
    raw_start = row.get("time_start_raw") if row.get("migration_version") else row.get("time_start")
    raw_end = row.get("time_end_raw") if row.get("migration_version") else row.get("time_end")
    return normalize_fact_dates(raw_start, raw_end, tz_name)


async def _count_pending(client: ArcadeDBClient, db: str) -> int:
    rows = await client.query(db, _COUNT_PENDING, params={"target": NORMALIZER_VERSION})
    return int(rows[0]["n"]) if rows else 0


async def backfill_database(client: ArcadeDBClient, db: str, *, page_size: int = 200,
                            dry_run: bool = False, tz_name: str | None = None) -> BackfillStats:
    tz_name = tz_name or settings.temporal_default_timezone
    stats = BackfillStats(database=db)
    await initialize_schema(client, db)          # adds the v2 properties and indexes (idempotent)
    stats.pending_before = await _count_pending(client, db)
    if dry_run or not stats.pending_before:
        stats.pending_after = stats.pending_before
        return stats

    failed: set[str] = set()                     # rows whose update failed this run: not retried in it
    while True:
        rows = await client.query(db, _PAGE, params={"target": NORMALIZER_VERSION, "n": page_size + len(failed)})
        page = [r for r in rows if r.get("fact_id") not in failed][:page_size]
        if not page:
            break
        for row in page:
            fact_id = row.get("fact_id")
            props = normalize_legacy_row(row, tz_name)
            try:
                result = await client.command(db, _UPDATE, params={**props, "fact_id": fact_id})
                if fact_id is None or not result or not result[0].get("count"):
                    raise RuntimeError("row not updated (no fact_id?)")
            except Exception as exc:  # noqa: BLE001 -- ArcadeDB command can raise httpx errors too; one row's failure must not block the page, recorded in stats.errors
                failed.add(fact_id)
                stats.errors.append(f"{fact_id}: {exc}")
                continue
            stats.migrated += 1
            stats.quarantined += bool(props["quarantine_reason"])
        logger.info("[backfill] %s: %d migrated, %d quarantined so far", db, stats.migrated, stats.quarantined)

    stats.pending_after = await _count_pending(client, db)
    return stats


async def _databases(client: ArcadeDBClient, branches: list[str], all_users: bool) -> list[str]:
    if branches:
        return [db_name_for(b) for b in branches]
    if all_users:
        return sorted(d for d in await client.list_databases() if d.startswith("user_"))
    return []


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--branch", action="append", default=[], help="a user's branch id (JWT sub); repeatable")
    target.add_argument("--all", action="store_true", help="every user_* database")
    parser.add_argument("--page-size", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true", help="count rows still to migrate, change nothing")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    client = ArcadeDBClient()
    failed = False
    for db in await _databases(client, args.branch, args.all):
        if not await client.database_exists(db):
            logger.warning("[backfill] %s does not exist, skipped", db)
            continue
        stats = await backfill_database(client, db, page_size=args.page_size, dry_run=args.dry_run)
        logger.info("[backfill] %s", asdict(stats))
        failed |= bool(stats.errors) or (not args.dry_run and stats.pending_after > 0)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
