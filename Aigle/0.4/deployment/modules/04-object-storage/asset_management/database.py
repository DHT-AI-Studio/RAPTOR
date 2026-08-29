import asyncpg
import json
import logging
import math
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from zoneinfo import ZoneInfo

from .models import AssetMetadata, AssetMetadataResponse, ExistenceInfo, model_to_response
from .config import settings

logger = logging.getLogger(__name__)

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
                    PRIMARY KEY (asset_path, version_id, branch)
                )
                """,
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
                    upload_date, archive_date, destroy_date, updated_by, branch, status, checksum
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (asset_path, version_id, branch) DO UPDATE SET
                    primary_filename = EXCLUDED.primary_filename,
                    associated_filenames = EXCLUDED.associated_filenames,
                    upload_date = EXCLUDED.upload_date,
                    archive_date = EXCLUDED.archive_date,
                    destroy_date = EXCLUDED.destroy_date,
                    updated_by = EXCLUDED.updated_by,
                    status = EXCLUDED.status,
                    checksum = EXCLUDED.checksum
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
                metadata.checksum
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
                if isinstance(item, list) and len(item) == 2:
                    if item[1] is not None:
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
            checksum=row["checksum"]
        )

    async def get_latest_asset(self, asset_path: str, branch: str) -> Optional[AssetMetadata]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM commit_history
                WHERE asset_path = $1 AND branch = $2
                ORDER BY upload_date DESC
                LIMIT 1
            """, asset_path, branch)
            return await self._row_to_metadata(row)

    async def check_path_existence(self, asset_path: str, checksum: str, branch: str) -> Optional[AssetMetadata]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM commit_history
                WHERE asset_path = $1 AND checksum = $2 AND branch = $3
                ORDER BY (CASE WHEN status = 'archived' THEN 1 ELSE 2 END) ASC, upload_date DESC
                LIMIT 1
            """, asset_path, checksum, branch)
            return await self._row_to_metadata(row)

    async def get_reusable_content(self, checksum: str, branch: str, status: str) -> Optional[AssetMetadata]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM commit_history
                WHERE checksum = $1 AND branch = $2 AND status = $3
                ORDER BY upload_date DESC LIMIT 1
            """, checksum, branch, status)
            return await self._row_to_metadata(row)

    async def get_asset_by_path_and_version(self, asset_path: str, version_id: str, branch: str) -> Optional[AssetMetadata]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM commit_history
                WHERE asset_path = $1 AND version_id = $2 AND branch = $3
            """, asset_path, version_id, branch)
            return await self._row_to_metadata(row)

    async def get_versions_by_key(self, key: str, branch: str) -> List[Dict]:
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

    async def delete_metadata(self, asset_path: str, version_id: str, branch: str):
        async with self.pool.acquire() as conn:
            if branch:
                await conn.execute("DELETE FROM commit_history WHERE asset_path = $1 AND version_id = $2 AND branch = $3", 
                                   asset_path, version_id, branch)
            else:
                await conn.execute("DELETE FROM commit_history WHERE asset_path = $1 AND version_id = $2", 
                                   asset_path, version_id)

    async def log_access(self, username: str, asset_path: str, version_id: str, branch: str, operation: str, success: bool, details: str = None):
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

    async def get_assets_to_archive(self, current_date: datetime) -> List[AssetMetadata]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM commit_history
                WHERE status = $1
                AND archive_date IS NOT NULL 
                AND archive_date <= $2
            """, "active", current_date)
            return [await self._row_to_metadata(row) for row in rows]

    async def get_assets_to_destroy(self, current_date: datetime) -> List[AssetMetadata]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM commit_history
                WHERE status = $1
                AND destroy_date IS NOT NULL
                AND destroy_date <= $2
            """, "archived", current_date)
            return [await self._row_to_metadata(row) for row in rows]

    async def get_assets_by_branch(self, branch: str) -> List[AssetMetadata]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM commit_history
                WHERE branch = $1
            """, branch)
            return [await self._row_to_metadata(row) for row in rows]

    async def get_head_version(self, asset_path: str, branch: str) -> Optional[str]:
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
        branch: Optional[str] = None, 
        keyword: Optional[str] = None, 
        start_date: Optional[datetime] = None, 
        end_date: Optional[datetime] = None, 
        page: int = 1, 
        page_size: int = 20
    ):
        async with self.pool.acquire() as conn:
            # 1. construct WHERE clauses
            where_clauses = ["updated_by = $1"]
            params = [username]

            if branch:
                params.append(branch)
                where_clauses.append(f"branch = ${len(params)}")
            
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
            count_query = f"SELECT COUNT(*) FROM commit_history {where_str}"
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
            """
            
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