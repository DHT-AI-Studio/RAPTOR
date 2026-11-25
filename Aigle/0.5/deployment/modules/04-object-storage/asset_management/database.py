import asyncio
import json
import logging
import math
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import asyncpg

from .config import settings
from .models import (
    AssetMetadata,
    AssetMetadataResponse,
    ExistenceInfo,
    model_to_response,
)

logger = logging.getLogger(__name__)


def summarize_file_rows(rows) -> dict:
    """GD-4 / VIE01-249: {count, file_types} from commit_history
    (primary_filename, associated_filenames) rows. Counts each row's primary
    file plus its associated files -- the same set GD-2's export puts in the
    ZIP's files/ section -- skipping associated entries _row_to_metadata
    also drops (malformed, or with no version_id). file_types: distinct
    lowercased extensions without the dot; extension-less names add none."""
    filenames: list[str] = []
    for row in rows:
        if row["primary_filename"]:
            filenames.append(row["primary_filename"])
        assoc = row["associated_filenames"]
        if isinstance(assoc, str):
            assoc = json.loads(assoc)
        for item in assoc or []:
            if isinstance(item, list) and len(item) == 2 and item[0] and item[1] is not None:
                filenames.append(item[0])

    types = set()
    for name in filenames:
        base = name.rsplit("/", 1)[-1]
        if "." in base.strip("."):
            types.add(base.rsplit(".", 1)[-1].lower())
    return {"count": len(filenames), "file_types": sorted(types)}


def _jsonb(value):
    """A JSONB column as Python: asyncpg hands it over as a JSON string (no codec is
    registered on this pool)."""
    return json.loads(value) if isinstance(value, str) else value


class Database:
    def __init__(self):
        self.pool = None

    async def init_db(self):
        """
        initialize the database schema if not exists
        """
        self.pool = await asyncpg.create_pool(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            database=settings.postgres_db,
            min_size=1,
            max_size=10
        )
        
        async with self.pool.acquire() as conn:
            queries = [
                """
                CREATE TABLE IF NOT EXISTS commit_history (
                    asset_path VARCHAR(255),
                    version_id VARCHAR(255),
                    branch VARCHAR(255),
                    updated_by VARCHAR(255),
                    primary_filename VARCHAR(255),
                    asset_key VARCHAR(255),
                    associated_filenames JSONB,
                    upload_date TIMESTAMPTZ,
                    archive_date TIMESTAMPTZ,
                    destroy_date TIMESTAMPTZ,
                    status VARCHAR(50),
                    checksum VARCHAR(255),
                    correlation_id VARCHAR(255),
                    cloned_from VARCHAR(255),
                    PRIMARY KEY (asset_path, version_id, branch)
                )
                """,
                # VIE01-215: commit_history predates correlation_id -- ADD COLUMN IF NOT
                # EXISTS so an already-deployed table (created before this column existed)
                # still picks it up, since CREATE TABLE IF NOT EXISTS above is a no-op then.
                "ALTER TABLE commit_history ADD COLUMN IF NOT EXISTS correlation_id VARCHAR(255)",
                # cloned_from: source version_id when clone_point() cloned Module 25's
                # index data at upload time (NULL otherwise). Lets register_asset() skip
                # a Module 25 round-trip for the common case (most rows never cloned).
                "ALTER TABLE commit_history ADD COLUMN IF NOT EXISTS cloned_from VARCHAR(255)",
                # CM-2: Module 10's image moderation verdict (Module 23 image guard),
                # written by POST /update-moderation. NULL when the asset wasn't moderated.
                "ALTER TABLE commit_history ADD COLUMN IF NOT EXISTS moderation_flags JSONB",
                """
                CREATE TABLE IF NOT EXISTS audit_log (
                    id BIGSERIAL PRIMARY KEY,
                    username VARCHAR(255),
                    asset_path VARCHAR(255),
                    version_id VARCHAR(255),
                    branch VARCHAR(255),
                    operation VARCHAR(50),
                    timestamp TIMESTAMPTZ,
                    success BOOLEAN,
                    details TEXT
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS filemeta (
                    dirhash   BIGINT NOT NULL,
                    name      VARCHAR(1024) NOT NULL,
                    directory VARCHAR(1024),
                    meta      BYTEA,
                    PRIMARY KEY (dirhash, name)
                )
                """
            ]
            
            for query in queries:
                await conn.execute(query)

            # Create indexes for performance optimization
            index_queries = [
                "CREATE INDEX IF NOT EXISTS idx_archive_date ON commit_history (archive_date)",
                "CREATE INDEX IF NOT EXISTS idx_destroy_date ON commit_history (destroy_date)",
                "CREATE INDEX IF NOT EXISTS idx_updated_by ON commit_history (updated_by)",
                "CREATE INDEX IF NOT EXISTS idx_status ON commit_history (status)",
                "CREATE INDEX IF NOT EXISTS idx_asset_key ON commit_history (asset_key)",
                "CREATE INDEX IF NOT EXISTS idx_asset_path_branch ON commit_history (asset_path, branch)",
                "CREATE INDEX IF NOT EXISTS idx_asset_path_version_id_branch ON commit_history (asset_path, version_id, branch)",
                "CREATE INDEX IF NOT EXISTS idx_asset_key_branch ON commit_history (asset_key, branch)",
                "CREATE INDEX IF NOT EXISTS idx_audit_log ON audit_log (asset_path, version_id)",
                "CREATE INDEX IF NOT EXISTS idx_audit_log_branch ON audit_log (asset_path, version_id, branch)",
                "CREATE INDEX IF NOT EXISTS idx_checksum_branch ON commit_history (checksum, branch)",
                "CREATE INDEX IF NOT EXISTS idx_upload_date ON commit_history (upload_date)",
                "CREATE EXTENSION IF NOT EXISTS pg_trgm;",
                "CREATE INDEX IF NOT EXISTS trgm_idx_asset_path ON commit_history USING gin (asset_path gin_trgm_ops)",
                "CREATE INDEX IF NOT EXISTS trgm_idx_primary_filename ON commit_history USING gin (primary_filename gin_trgm_ops)"
            ]

            for idx_query in index_queries:
                await conn.execute(idx_query)
                           
            logger.info("PostgreSQL schema initialized")

    async def save_metadata(self, metadata: AssetMetadata):
        async with self.pool.acquire() as conn:
            query = """
                INSERT INTO commit_history (
                    asset_path, version_id, primary_filename, asset_key, associated_filenames,
                    upload_date, archive_date, destroy_date, updated_by, branch, status, checksum,
                    cloned_from, moderation_flags
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14::jsonb)
                ON CONFLICT (asset_path, version_id, branch) DO UPDATE SET
                    primary_filename = EXCLUDED.primary_filename,
                    associated_filenames = EXCLUDED.associated_filenames,
                    upload_date = EXCLUDED.upload_date,
                    archive_date = EXCLUDED.archive_date,
                    destroy_date = EXCLUDED.destroy_date,
                    updated_by = EXCLUDED.updated_by,
                    status = EXCLUDED.status,
                    checksum = EXCLUDED.checksum,
                    cloned_from = EXCLUDED.cloned_from,
                    -- CM-2: a re-save without a verdict keeps the one Module 10 wrote
                    moderation_flags = COALESCE(EXCLUDED.moderation_flags, commit_history.moderation_flags)
            """
            await conn.execute(query,
                metadata.asset_path,
                metadata.version_id,
                metadata.primary_filename,
                f"{metadata.asset_path}/{metadata.primary_filename}",
                json.dumps(metadata.associated_filenames),
                metadata.upload_date,
                metadata.archive_date if isinstance(metadata.archive_date, datetime) else None,
                metadata.destroy_date if isinstance(metadata.destroy_date, datetime) else None,
                metadata.user,
                metadata.branch,
                metadata.status,
                metadata.checksum,
                metadata.cloned_from,
                json.dumps(metadata.moderation_flags) if metadata.moderation_flags is not None else None,
            )

    async def _row_to_metadata(self, row) -> AssetMetadata:
        """Convert asyncpg.Record to AssetMetadata"""
        if not row:
            return None

        assoc_files = row["associated_filenames"]
        if isinstance(assoc_files, str):
            assoc_files = json.loads(assoc_files)

        clean_assoc_files = []
        if assoc_files and isinstance(assoc_files, list):
            for item in assoc_files:
                if isinstance(item, list) and len(item) == 2 and item[1] is not None:
                    clean_assoc_files.append(item)

        return AssetMetadata(
            asset_path=row["asset_path"],
            version_id=row["version_id"],
            primary_filename=row["primary_filename"],
            associated_filenames=clean_assoc_files or [],
            upload_date=row["upload_date"].astimezone(ZoneInfo(settings.timezone)),
            # archive_date=row["archive_date"].astimezone(ZoneInfo(settings.timezone)) if row["archive_date"] else "No expiration date",
            # destroy_date=row["destroy_date"].astimezone(ZoneInfo(settings.timezone)) if row["destroy_date"] else "No expiration date",
            archive_date=row["archive_date"].astimezone(ZoneInfo(settings.timezone)) if row["archive_date"] else None,
            destroy_date=row["destroy_date"].astimezone(ZoneInfo(settings.timezone)) if row["destroy_date"] else None,
            user=row["updated_by"],
            branch=row["branch"],
            status=row["status"],
            checksum=row["checksum"],
            cloned_from=row["cloned_from"],
            moderation_flags=_jsonb(row.get("moderation_flags")),
        )

    async def get_latest_asset(self, asset_path: str, branch: str) -> AssetMetadata | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM commit_history
                WHERE asset_path = $1 AND branch = $2
                ORDER BY upload_date DESC
                LIMIT 1
            """, asset_path, branch)
            return await self._row_to_metadata(row)

    async def check_path_existence(self, asset_path: str, checksum: str, branch: str) -> AssetMetadata | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM commit_history
                WHERE asset_path = $1 AND checksum = $2 AND branch = $3
                ORDER BY (CASE WHEN status = 'archived' THEN 1 ELSE 2 END) ASC, upload_date DESC
                LIMIT 1
            """, asset_path, checksum, branch)
            return await self._row_to_metadata(row)

    async def get_reusable_content(self, checksum: str, branch: str, status: str) -> AssetMetadata | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM commit_history
                WHERE checksum = $1 AND branch = $2 AND status = $3
                ORDER BY upload_date DESC LIMIT 1
            """, checksum, branch, status)
            return await self._row_to_metadata(row)

    async def get_asset_by_path_and_version(self, asset_path: str, version_id: str, branch: str) -> AssetMetadata | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM commit_history
                WHERE asset_path = $1 AND version_id = $2 AND branch = $3
            """, asset_path, version_id, branch)
            return await self._row_to_metadata(row)

    async def get_versions_by_key(self, key: str, branch: str) -> list[dict]:
        key = os.path.normpath(key).replace('\\', '/')
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT asset_path, version_id, primary_filename, upload_date, status
                FROM commit_history
                WHERE asset_key = $1 AND branch = $2
                ORDER BY upload_date DESC
            """, key, branch)
            return [
                {
                    "asset_path": row["asset_path"],
                    "version_id": row["version_id"],
                    "primary_filename": row["primary_filename"],
                    "upload_date": row["upload_date"].astimezone(ZoneInfo(settings.timezone)),
                    "status": row["status"]
                } for row in rows
            ]

    async def update_status(self, asset_path: str, version_id: str, status: str, branch: str):
        async with self.pool.acquire() as conn:
            if branch:
                await conn.execute("""
                    UPDATE commit_history SET status = $1
                    WHERE asset_path = $2 AND version_id = $3 AND branch = $4
                """, status, asset_path, version_id, branch)
            else:
                await conn.execute("""
                    UPDATE commit_history SET status = $1
                    WHERE asset_path = $2 AND version_id = $3
                """, status, asset_path, version_id)

    async def update_moderation(self, asset_path: str, version_id: str, branch: str, flags: dict) -> bool:
        """CM-2: store the asset's moderation_flags. False when no such row."""
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                UPDATE commit_history SET moderation_flags = $1::jsonb
                WHERE asset_path = $2 AND version_id = $3 AND branch = $4
            """, json.dumps(flags), asset_path, version_id, branch)
            return result != "UPDATE 0"

    async def claim_processing(self, asset_path: str, version_id: str, branch: str, correlation_id: str) -> dict | None:
        """VIE01-215: atomic conditional claim for a processing job.

        Succeeds (returns the updated row) if the asset is currently
        'uploaded' or 'failed' (first claim), or already 'processing' under
        this exact correlation_id (idempotent re-claim by the same caller
        after a retry/redelivery). Returns None otherwise -- claimed by a
        different correlation_id, or in a status that disallows claiming
        (e.g. 'active', 'archived', not found) -- the caller maps that to 409.
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                UPDATE commit_history
                SET status = 'processing', correlation_id = $4
                WHERE asset_path = $1 AND version_id = $2 AND branch = $3
                  AND (status IN ('uploaded', 'failed')
                       OR (status = 'processing' AND correlation_id = $4))
                RETURNING status, correlation_id
            """, asset_path, version_id, branch, correlation_id)
            return dict(row) if row else None

    async def delete_metadata(self, asset_path: str, version_id: str, branch: str):
        async with self.pool.acquire() as conn:
            if branch:
                await conn.execute("DELETE FROM commit_history WHERE asset_path = $1 AND version_id = $2 AND branch = $3", 
                                   asset_path, version_id, branch)
            else:
                await conn.execute("DELETE FROM commit_history WHERE asset_path = $1 AND version_id = $2", 
                                   asset_path, version_id)

    async def log_access(self, username: str, asset_path: str, version_id: str, branch: str, operation: str, success: bool, details: str | None = None):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO audit_log (username, asset_path, version_id, branch, operation, timestamp, success, details)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """, username, asset_path, version_id, branch, operation, 
               datetime.now(tz=ZoneInfo(settings.timezone)), success, details)

    async def cleanup_old_logs(self, current_date: datetime, days: int):
        cutoff = current_date - timedelta(days=days)
        async with self.pool.acquire() as conn:
            result = await conn.execute("DELETE FROM audit_log WHERE timestamp < $1", cutoff)
            logger.info(f"Cleanup logs: {result}")

    async def get_assets_to_archive(self, current_date: datetime) -> list[AssetMetadata]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM commit_history
                WHERE status NOT IN ('archived', 'destroyed')
                AND archive_date IS NOT NULL
                AND archive_date <= $1
            """, current_date)
            return [await self._row_to_metadata(row) for row in rows]

    async def get_assets_to_destroy(self, current_date: datetime) -> list[AssetMetadata]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM commit_history
                WHERE status = $1
                AND destroy_date IS NOT NULL
                AND destroy_date <= $2
            """, "archived", current_date)
            return [await self._row_to_metadata(row) for row in rows]

    async def get_assets_by_branch(self, branch: str) -> list[AssetMetadata]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM commit_history
                WHERE branch = $1
            """, branch)
            return [await self._row_to_metadata(row) for row in rows]

    async def get_gdpr_file_summary(self, branch: str) -> dict:
        """GD-4 / VIE01-249: counts-only summary of a user's files, for
        Module 33's DSAR. Same row scope as gdpr_erase_user (branch ==
        user_id, every status) -- i.e. exactly what an erasure would delete.
        Selects only the two filename columns, never full metadata rows."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT primary_filename, associated_filenames FROM commit_history
                WHERE branch = $1
            """, branch)
        return summarize_file_rows(rows)

    async def get_head_version(self, asset_path: str, branch: str) -> str | None:
        async with self.pool.acquire() as conn:
            val = await conn.fetchval("""
                SELECT version_id FROM commit_history
                WHERE asset_path = $1 AND branch = $2
                ORDER BY upload_date DESC
                LIMIT 1
            """, asset_path, branch)
            return val

    async def check_file_existence(self, checksum: str, asset_path: str, branch: str) -> ExistenceInfo:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT asset_path, version_id, primary_filename FROM commit_history
                WHERE checksum = $1 AND branch = $2
                ORDER BY upload_date DESC
                LIMIT 1
            """, checksum, branch)
            
            if not row:
                return ExistenceInfo(exists=False, message="The primary file is a new file")
            
            if row["asset_path"] == asset_path:
                return ExistenceInfo(
                    exists=True, 
                    message=f"Exists with path: {row['asset_path']} and ID: {row['version_id']}", 
                    existing_asset_path=row["asset_path"], 
                    existing_version_id=row["version_id"]
                )
            return ExistenceInfo(
                exists=True, 
                message=f"Exists with different name {row['primary_filename']} at {row['asset_path']}", 
                existing_asset_path=row["asset_path"], 
                existing_version_id=row["version_id"]
            )

    async def scan_assets(self, files: list[tuple[str, str]], branch: str, concurrency: int = 10) -> list[dict]:
        """MTR-8 / VIE01-307: batch duplicate-content check. One lookup per
        (path, checksum) pair -- same checksum+branch scoping as
        check_file_existence, just without a status filter so any match
        (whatever its current status) counts -- bounded by `concurrency`
        concurrent connections, not a single batched query, per the
        ticket's own concurrency-cap requirement."""
        semaphore = asyncio.Semaphore(concurrency)

        async def _lookup(path: str, checksum: str) -> dict:
            async with semaphore, self.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT asset_path, version_id, status FROM commit_history
                    WHERE checksum = $1 AND branch = $2
                    ORDER BY upload_date DESC
                    LIMIT 1
                """, checksum, branch)
            if not row:
                return {"path": path, "checksum": checksum, "status": "not_found"}
            return {
                "path": path,
                "checksum": checksum,
                "status": row["status"],
                "existing_asset_path": row["asset_path"],
                "existing_version_id": row["version_id"],
            }

        return await asyncio.gather(*[_lookup(path, checksum) for path, checksum in files])

    async def update_asset_by_checksum(
        self,
        checksum: str,
        branch: str,
        new_user: str,
        new_archive_date: datetime, 
        new_destroy_date: datetime
    ) -> AssetMetadata:
        """
        Update the earliest uploaded asset with the same checksum.
        Parameters:
            checksum (str): The checksum of the asset to update.
            branch (str): The branch of the asset.
            new_user (str): The new username to set.
            new_archive_date (datetime): The new archive date to set.
            new_destroy_date (datetime): The new destroy date to set.
        Returns:
            AssetMetadataResponse: The updated asset metadata.
        Raises:
            HTTPException: If no asset is found with the given checksum.
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM commit_history
                WHERE checksum = $1 AND branch = $2
                ORDER BY upload_date ASC
                LIMIT 1
            """, checksum, branch)

            if not row:
                return None

            await conn.execute("""
                UPDATE commit_history
                SET archive_date = $1,
                    destroy_date = $2,
                    status = 'active'
                WHERE asset_path = $3 AND version_id = $4
            """, new_archive_date, new_destroy_date, row["asset_path"], row["version_id"])

            logger.info(f"Updated asset {row['asset_path']}/{row['version_id']} by user {new_user}")

            updated_metadata = await self._row_to_metadata(row)
            updated_metadata.archive_date = new_archive_date
            updated_metadata.destroy_date = new_destroy_date
            updated_metadata.status = 'active'
            return updated_metadata

    async def get_user_commits(
        self,
        username: str,
        branch: str | None = None,
        keyword: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        page: int = 1,
        page_size: int = 20,
        statuses: list[str] | None = None,
    ):
        async with self.pool.acquire() as conn:
            # 1. construct WHERE clauses
            where_clauses = ["updated_by = $1"]
            params = [username]

            if branch:
                params.append(branch)
                where_clauses.append(f"branch = ${len(params)}")

            # MTR-4 / VIE01-304: repeatable ?status= query param -> status =
            # ANY($n). Omitted (None/empty) means all 6 statuses, matching
            # VIE01-214's requirement that this endpoint never defaults to
            # filtering on "active" only.
            if statuses:
                params.append(list(statuses))
                where_clauses.append(f"status = ANY(${len(params)})")

            if keyword:
                params.append(f"%{keyword}%")
                # where_clauses.append(f"(asset_path ILIKE ${len(params)} OR primary_filename ILIKE ${len(params)})")
                where_clauses.append(f"(primary_filename ILIKE ${len(params)})")

            # 2. Add date range filters (for upload_date)
            if start_date:
                params.append(start_date)
                where_clauses.append(f"upload_date >= ${len(params)}")
            
            if end_date:
                params.append(end_date)
                where_clauses.append(f"upload_date <= ${len(params)}")

            where_str = " WHERE " + " AND ".join(where_clauses)

            # 3. Get total count
            count_query = f"SELECT COUNT(*) FROM commit_history {where_str}"  # nosec B608 -- where_str only ever has fixed column names + $N placeholders, every value goes through params below
            total_count = await conn.fetchval(count_query, *params) or 0
            
            total_pages = math.ceil(total_count / page_size) if total_count > 0 else 0
            page = max(1, min(page, total_pages)) if total_pages > 0 else 1

            # 4. Get commits
            limit = page_size
            offset = (page - 1) * page_size

            # Copy parameter list and add limit/offset
            query_params = list(params)
            query_params.extend([limit, offset])
            
            query = f"""
                SELECT * FROM commit_history
                {where_str}
                ORDER BY upload_date DESC
                LIMIT ${len(query_params)-1} OFFSET ${len(query_params)}
            """  # nosec B608 -- same where_str as count_query above, all params
            
            rows = await conn.fetch(query, *query_params)
            commits = [await self._row_to_metadata(row) for row in rows]
            commits = [model_to_response(c, AssetMetadataResponse) for c in commits]
            
            return {
                "total_count": total_count,
                "total_pages": total_pages,
                "page": page,
                "page_size": page_size,
                "commits": commits
            }

    async def update_asset_expiration(
        self, 
        asset_path: str, 
        version_id: str, 
        branch: str, 
        new_archive_date: datetime, 
        new_destroy_date: datetime,
        target_status: str
    ) -> bool:
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                UPDATE commit_history 
                SET archive_date = $1, 
                    destroy_date = $2, 
                    status = $3
                WHERE asset_path = $4 AND version_id = $5 AND branch = $6
            """, new_archive_date, new_destroy_date, target_status, asset_path, version_id, branch)
            return result == "UPDATE 1"

    async def close(self):
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL connection pool closed")