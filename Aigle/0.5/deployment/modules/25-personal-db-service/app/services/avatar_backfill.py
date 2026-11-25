"""Re-choose existing people's avatars by the current rule (identity_store.avatar_choice).

A person's avatar used to be the first face they got, kept for as long as it stayed theirs -- often a
poor frame from the trace a pending person was started from. New changes to a person's groups now
re-choose it (the best-quality frame of their largest group), but an avatar nothing has touched since
keeps the old pick. This re-chooses it once for every person:

- a seed photo always stays the avatar;
- merged and deleted people are skipped: their avatar is part of the merge / delete history that
  undo restores, and they hold no groups to choose from;
- only the HAS_PROFILE_FACE edge changes -- groups, traces and matching are untouched.

Re-runnable: a person whose avatar is already the chosen one is left as is, so a second run changes
nothing.

Run inside the Module 25 container:
    python -m app.services.avatar_backfill --all --dry-run   # list what would change, change nothing
    python -m app.services.avatar_backfill --all             # every user_* database
    python -m app.services.avatar_backfill --branch <sub>    # one user
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import asdict, dataclass

from app.services.arcadedb_client import ArcadeDBClient, db_name_for
from app.services.identity_store import _fix_avatar, avatar_choice, face_schema_present

logger = logging.getLogger("personal_db.avatar_backfill")


@dataclass
class AvatarStats:
    database: str
    people: int = 0
    changed: int = 0          # would change, with --dry-run
    unchanged: int = 0
    skipped: int = 0          # seed photo, or merged / deleted


async def backfill_avatars(client: ArcadeDBClient, db: str, *, dry_run: bool = False) -> AvatarStats:
    stats = AvatarStats(database=db)
    if not await face_schema_present(client, db):
        return stats
    for row in await client.query(db, "SELECT identity_id, identity_status FROM Identity"):
        stats.people += 1
        identity_id = row["identity_id"]
        if row.get("identity_status") in ("merged", "deleted"):
            stats.skipped += 1
            continue
        choice = await avatar_choice(client, db, identity_id)
        if choice is None:
            stats.skipped += 1
            continue
        current, best = choice
        if current == best:
            stats.unchanged += 1
            continue
        stats.changed += 1
        logger.info("[avatar] %s %s: %s -> %s%s", db, identity_id, current, best, " (dry run)" if dry_run else "")
        if not dry_run:
            await _fix_avatar(client, db, identity_id)
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
    parser.add_argument("--dry-run", action="store_true", help="list the avatars that would change, change nothing")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    client = ArcadeDBClient()
    for db in await _databases(client, args.branch, args.all):
        if not await client.database_exists(db):
            logger.warning("[avatar] %s does not exist, skipped", db)
            continue
        logger.info("[avatar] %s", asdict(await backfill_avatars(client, db, dry_run=args.dry_run)))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
