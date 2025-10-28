import aiomysql
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import List, Dict, Optional
import json
from passlib.context import CryptContext
import logging
import os
from .models import AssetMetadata, User
from .config import settings

logger = logging.getLogger(__name__)

class Database:
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.pool = None

    async def init_db(self):
        """
        Initializes the database schema.

        This method creates the necessary tables in the MySQL database,
        including the commit history, users, access control, audit log, and
        file metadata tables.

        This method is idempotent, and can be safely called multiple times.
        """
        self.pool = await aiomysql.create_pool(
            host=settings.mysql_host,
            port=settings.mysql_port,
            user=settings.mysql_user,
            password=settings.mysql_password,
            db=settings.mysql_database,
            minsize=1,
            maxsize=10
        )
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("SET sql_notes = 0;") # Disable warnings
                await cursor.execute("""
                    CREATE TABLE IF NOT EXISTS commit_history (
                        asset_path VARCHAR(255),
                        version_id VARCHAR(255),
                        branch VARCHAR(255),
                        primary_filename VARCHAR(255),
                        asset_key VARCHAR(255),
                        associated_filenames JSON,
                        upload_date DATETIME,
                        archive_date DATETIME,
                        destroy_date DATETIME,
                        status VARCHAR(50),
                        PRIMARY KEY (asset_path, version_id, branch)
                    )
                """)
                await cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        username VARCHAR(255) PRIMARY KEY,
                        password_hash VARCHAR(255),
                        branch VARCHAR(255),
                        permissions JSON,
                        created_at DATETIME
                    )
                """)
                await cursor.execute("""
                    CREATE TABLE IF NOT EXISTS audit_log (
                        id BIGINT AUTO_INCREMENT PRIMARY KEY,
                        username VARCHAR(255),
                        asset_path VARCHAR(255),
                        version_id VARCHAR(255),
                        branch VARCHAR(255),
                        operation VARCHAR(50),
                        timestamp DATETIME,
                        success BOOLEAN,
                        details TEXT
                    )
                """)
                await cursor.execute("""
                    CREATE TABLE IF NOT EXISTS filemeta (
                        dirhash   BIGINT NOT NULL       COMMENT 'first 64 bits of MD5 hash value of directory field',
                        name      VARCHAR(766) NOT NULL COMMENT 'directory or file name',
                        directory TEXT NOT NULL         COMMENT 'full path to parent directory',
                        meta      LONGBLOB,
                        PRIMARY KEY (`dirhash`, `name`)
                    ) DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;
                """)
                try:
                    await cursor.execute("""CREATE INDEX idx_archive_date ON commit_history (archive_date)""")
                    await cursor.execute("""CREATE INDEX idx_destroy_date ON commit_history (destroy_date)""")
                    await cursor.execute("""CREATE INDEX idx_status ON commit_history (status)""")
                    await cursor.execute("""CREATE INDEX idx_asset_key ON commit_history (asset_key)""")
                    await cursor.execute("""CREATE INDEX idx_asset_path_branch ON commit_history (asset_path, branch)""")
                    await cursor.execute("""CREATE INDEX idx_asset_path_version_id_branch ON commit_history (asset_path, version_id, branch)""")
                    await cursor.execute("""CREATE INDEX idx_asset_key_branch ON commit_history (asset_key, branch)""")
                    await cursor.execute("""CREATE INDEX idx_audit_log ON audit_log (asset_path, version_id)""")
                    await cursor.execute("""CREATE INDEX idx_audit_log_branch ON audit_log (asset_path, version_id, branch)""")
                    await conn.commit()
                except Exception as e:
                    if e.args[0] == 1061:  # Duplicate key name
                        pass
                    else:
                        raise
                           
                logger.info("Database schema initialized")

    async def save_metadata(self, metadata: AssetMetadata):
        """
        Save the given AssetMetadata to the database.

        Args:
            metadata: the AssetMetadata to save.
        """
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("""
                    INSERT INTO commit_history (
                        asset_path, version_id, primary_filename, asset_key, associated_filenames,
                        upload_date, archive_date, destroy_date, branch, status
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) AS new_record
                    ON DUPLICATE KEY UPDATE
                        primary_filename = new_record.primary_filename,
                        associated_filenames = new_record.associated_filenames,
                        upload_date = new_record.upload_date,
                        archive_date = new_record.archive_date,
                        destroy_date = new_record.destroy_date,
                        branch = new_record.branch,
                        status = new_record.status
                """, (
                    metadata.asset_path,
                    metadata.version_id,
                    metadata.primary_filename,
                    metadata.asset_path + '/' + metadata.primary_filename,
                    json.dumps(metadata.associated_filenames),
                    metadata.upload_date,
                    metadata.archive_date,
                    metadata.destroy_date,
                    metadata.branch,
                    metadata.status
                ))
                await conn.commit()

    async def get_latest_active_asset(self, asset_path: str, branch: str) -> Optional[AssetMetadata]:
        """
        Retrieve the latest active version of an asset by its asset_path.
        
        Args:
            asset_path (str): The path of the asset to search for.
            branch (str): The branch of the asset.
        
        Returns:
            Optional[AssetMetadata]: The metadata of the latest active version, or None if not found.
        """
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute("""
                    SELECT asset_path, version_id, primary_filename, associated_filenames,
                        upload_date, archive_date, destroy_date, branch, status
                    FROM commit_history
                    WHERE asset_path = %s AND status = 'active' AND branch = %s
                    ORDER BY upload_date DESC
                    LIMIT 1
                """, (asset_path, branch))
                row = await cursor.fetchone()
                if not row:
                    return None
                
                return AssetMetadata(
                    asset_path=row["asset_path"],
                    version_id=row["version_id"],
                    primary_filename=row["primary_filename"],
                    associated_filenames=json.loads(row["associated_filenames"]) if row["associated_filenames"] else [],
                    upload_date=row["upload_date"],
                    archive_date=row["archive_date"],
                    destroy_date=row["destroy_date"],
                    branch=row["branch"],
                    status=row["status"]
                )

    async def get_asset_by_path_and_version(self, asset_path: str, version_id: str, branch: str) -> Optional[AssetMetadata]:
        """
        Retrieve the AssetMetadata with the given asset_path and version_id from the database.

        Args:
            asset_path (str): the asset path to retrieve.
            version_id (str): the version ID to retrieve.
            branch (str): the branch of the asset

        Returns:
            the AssetMetadata with the given asset_path and version_id, or None if not found.
        """
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute("""
                    SELECT asset_path, version_id, primary_filename, associated_filenames,
                           upload_date, archive_date, destroy_date, branch, status
                    FROM commit_history
                    WHERE asset_path = %s AND version_id = %s AND branch = %s
                """, (asset_path, version_id, branch))
                row = await cursor.fetchone()
                if not row:
                    return None
                return AssetMetadata(
                    asset_path=row["asset_path"],
                    version_id=row["version_id"],
                    primary_filename=row["primary_filename"],
                    associated_filenames=json.loads(row["associated_filenames"]) if row["associated_filenames"] else [],
                    upload_date=row["upload_date"],
                    archive_date=row["archive_date"],
                    destroy_date=row["destroy_date"],
                    branch=row["branch"],
                    status=row["status"]
                )

    async def get_versions_by_key(self, key: str, branch: str) -> List[Dict]:
        """
        Retrieve the versions of the active asset with the given key from the database.

        Args:
            key (str): the key of the asset to retrieve.
            branch (str): the branch of the asset

        Returns:
            a list of dictionaries containing information about each version of the asset.
            Each dictionary contains the keys "asset_path", "version_id", "primary_filename", and "last_modified".
        """
        key = os.path.normpath(key).replace('\\', '/')
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute("""
                    SELECT asset_path, version_id, primary_filename, upload_date, asset_key
                    FROM commit_history
                    WHERE asset_key = %s AND branch = %s AND status = 'active'
                    ORDER BY upload_date DESC
                """, (key, branch))
                rows = await cursor.fetchall()
                return [
                    {
                        "asset_path": row["asset_path"],
                        "version_id": row["version_id"],
                        "primary_filename": row["primary_filename"],
                        "last_modified": row["upload_date"].isoformat()
                    } for row in rows
                ]

    async def update_status(self, asset_path: str, version_id: str, status: str, branch: str):
        """
        Update the status of the asset with the given asset_path and version_id in the database.

        Args:
            asset_path (str): path of the asset to update.
            version_id (str): version_id of the asset to update.
            status (str): new status of the asset.
            branch (str): branch of the asset
        """
        if branch:
            async with self.pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute("""
                        UPDATE commit_history SET status = %s
                        WHERE asset_path = %s AND version_id = %s AND branch = %s
                    """, (status, asset_path, version_id, branch))
                    await conn.commit()
        else:
            async with self.pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute("""
                        UPDATE commit_history SET status = %s
                        WHERE asset_path = %s AND version_id = %s
                    """, (status, asset_path, version_id))
                    await conn.commit()

    async def delete_metadata(self, asset_path: str, version_id: str, branch: str):
        """
        Delete the metadata of the asset with the given asset_path and version_id from the database.

        Args:
            asset_path (str): path of the asset to delete.
            version_id (str): version_id of the asset to delete.
            branch (str): branch of the asset
        """
        if branch:
            async with self.pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    # Delete the commit_history
                    await cursor.execute("""
                        DELETE FROM commit_history WHERE asset_path = %s AND version_id = %s AND branch = %s
                    """, (asset_path, version_id, branch))
                    # Delete the audit_log
                    await cursor.execute("""
                        DELETE FROM audit_log WHERE asset_path = %s AND version_id = %s AND branch = %s
                    """, (asset_path, version_id, branch))
                    await conn.commit()
        else:
            async with self.pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    # Delete the commit_history
                    await cursor.execute("""
                        DELETE FROM commit_history WHERE asset_path = %s AND version_id = %s
                    """, (asset_path, version_id))
                    # Delete the audit_log
                    await cursor.execute("""
                        DELETE FROM audit_log WHERE asset_path = %s AND version_id = %s
                    """, (asset_path, version_id))
                    await conn.commit()

    async def get_user_by_name(self, username: str) -> Optional[Dict]:
        """
        Retrieve the user information of the specified user from the database.

        Args:
            username: The username of the user whose information is to be retrieved.

        Returns:
            A dictionary containing the user's username, password hash, and roles. Returns None if no user is found.
        """
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute("""
                    SELECT username, password_hash, branch, permissions FROM users WHERE username = %s
                """, (username,))
                row = await cursor.fetchone()
                if not row:
                    return None
                
                return User(
                    username=row["username"],
                    password_hash=row["password_hash"],
                    branch=row["branch"],
                    permissions=json.loads(row["permissions"]) if row["permissions"] else []
                )

    async def create_user(self, username: str, password: str, branch: str, permissions: List[str]):
        """
        Create a new user in the database and assign a specific branch and permissions to the user.

        Args:
            username: The username of the user to create.
            password: The plaintext password of the user.
            branch: The branch of the user can access.
            permissions: The permissions of the user can use.

        Raises:
            Exception if the username already exists.
        """
        available_permissions = ["upload", "download", "list", "archive", "destroy", "admin"]
        for permission in permissions:
            if permission not in available_permissions:
                raise Exception(f"Invalid permission: {permission}. Available permissions: {available_permissions}")
            
        permission = json.dumps(permissions)
        password_hash = self.pwd_context.hash(password)

        async with self.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                try:
                    await cursor.execute("""
                        INSERT INTO users (username, password_hash, branch, permissions, created_at)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (username, password_hash, branch, permission, datetime.now(tz=ZoneInfo(settings.timezone))))
                    await conn.commit()
                except aiomysql.IntegrityError as e:
                    # 1062 = Duplicate entry
                    if e.args[0] == 1062:
                        raise Exception(f"User '{username}' already exists. Please use a different username.") from e
                    else:
                        raise

    async def create_admin_user(self, username: str, password: str):
        """
        Create a new user in the database and assign a new branch to the user.

        Args:
            username: The username of the user to create.
            password: The plaintext password of the user.

        Raises:
            Exception if the username already exists.
        """
        branch = f"{username}_space" # Create a new branch for the user
        permission = ["admin"]

        # Create the admin user
        await self.create_user(username, password, branch, permission)

    async def delete_user_by_name(self, username: str):
        """
        Delete the specified user with the given username from the database.

        Args:
            username (str): The username of the user to delete.
        """
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("""
                    DELETE FROM users WHERE username = %s
                """, (username,))
                await conn.commit()

    async def change_shared_user_permission(self, username: str, permissions: list[str]):
        """
        Change the permission of a shared user.

        Args:
            username (str): The username of the user to change.
            permissions (list[str]): The new permission to set for the user.
        """
        available_permissions = ["upload", "download", "list", "archive", "destroy"]
        for permission in permissions:
            if permission == "admin":
                raise Exception("Shared users cannot have admin permissions. Available permissions: upload, download, list, archive, destroy")
            if permission not in available_permissions:
                raise Exception(f"Invalid permission: {permission}. Available permissions: {available_permissions}")

        permissions = json.dumps(permissions)
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("""
                    UPDATE users SET permissions = %s WHERE username = %s
                """, (permissions, username))
                await conn.commit()

    async def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify the given plaintext password matches the given hashed password.

        Args:
            plain_password: The plaintext password to verify.
            hashed_password: The hashed password to compare against.

        Returns:
            True if the plaintext password matches the hashed password, False otherwise.
        """
        return self.pwd_context.verify(plain_password, hashed_password)

    async def log_access(self, username: str, asset_path: str, version_id: str, branch: str, operation: str, success: bool, details: str = None):
        """
        Log access to the given asset in the audit_log table.

        Args:
            username (str): The username of the user who accessed the asset.
            asset_path (str): The path of the asset that was accessed.
            version_id (str): The version_id of the asset that was accessed.
            branch (str): The branch of the asset that was accessed.
            operation (str): A string describing the operation that was performed (e.g. read, write, delete).
            success (bool): A boolean indicating whether the operation was successful.
            details (str): An optional string containing additional details about the operation.

        This function will insert a new record into the audit_log table.
        """
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("""
                    INSERT INTO audit_log (username, asset_path, version_id, branch, operation, timestamp, success, details)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (username, asset_path, version_id, branch, operation, datetime.now(tz=ZoneInfo(settings.timezone)), success, details))
                await conn.commit()

    async def cleanup_old_logs(self, current_date: datetime, days: int):
        """
        Delete audit logs older than the specified number of days using Python time.

        Args:
            current_date: Current datetime for cutoff calculation.
            days: Number of days to keep logs. Logs older than this will be deleted.
        """
        cutoff = current_date - timedelta(days=days)

        async with self.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                batch_size = 10000
                deleted = 1
                total_deleted = 0
                while deleted:
                    await cursor.execute("""
                        DELETE FROM audit_log
                        WHERE timestamp < %s
                        LIMIT %s
                    """, (cutoff, batch_size))
                    deleted = cursor.rowcount
                    total_deleted += deleted
                    await conn.commit()

                logger.info(f"Deleted {total_deleted} audit logs older than {days} days")

    async def get_assets_to_archive(self, current_date: datetime) -> List[AssetMetadata]:
        """
        Retrieve a list of AssetMetadata objects for assets that are ready to be archived.
        
        Args:
            current_date: The current date and time.
        
        Returns:
            A list of AssetMetadata objects for assets that are ready to be archived. The list is empty if no assets are ready.
        """
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute("""
                    SELECT asset_path, version_id, primary_filename, associated_filenames,
                           upload_date, archive_date, destroy_date, branch, status
                    FROM commit_history
                    WHERE status = %s AND archive_date <= %s
                """, ("active", current_date))
                rows = await cursor.fetchall()
                return [
                    AssetMetadata(
                        asset_path=row["asset_path"],
                        version_id=row["version_id"],
                        primary_filename=row["primary_filename"],
                        associated_filenames=json.loads(row["associated_filenames"]) if row["associated_filenames"] else [],
                        upload_date=row["upload_date"],
                        archive_date=row["archive_date"],
                        destroy_date=row["destroy_date"],
                        branch=row["branch"],
                        status=row["status"]
                    ) for row in rows
                ]

    async def get_assets_to_destroy(self, current_date: datetime) -> List[AssetMetadata]:
        """
        Retrieve a list of AssetMetadata objects for assets that are ready to be destroyed.
        
        Args:
            current_date: The current date and time.
        
        Returns:
            A list of AssetMetadata objects for assets that are ready to be destroyed. The list is empty if no assets are ready.
        """
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute("""
                    SELECT asset_path, version_id, primary_filename, associated_filenames,
                           upload_date, archive_date, destroy_date, branch, status
                    FROM commit_history
                    WHERE status = %s AND destroy_date <= %s
                """, ("archived", current_date))
                rows = await cursor.fetchall()
                return [
                    AssetMetadata(
                        asset_path=row["asset_path"],
                        version_id=row["version_id"],
                        primary_filename=row["primary_filename"],
                        associated_filenames=json.loads(row["associated_filenames"]) if row["associated_filenames"] else [],
                        upload_date=row["upload_date"],
                        archive_date=row["archive_date"],
                        destroy_date=row["destroy_date"],
                        branch=row["branch"],
                        status=row["status"]
                    ) for row in rows
                ]

    async def get_head_version(self, asset_path: str, branch: str) -> str:
        """
        Retrieve the latest version of an asset by its asset_path.
        
        Args:
            asset_path (str): The path of the asset to search for.
            branch (str): The branch of the asset.
        Returns:
            Optional[AssetMetadata]: The metadata of the latest active version, or None if not found.
        """
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute("""
                    SELECT version_id FROM commit_history
                    WHERE asset_path = %s AND branch = %s
                    ORDER BY upload_date DESC
                    LIMIT 1
                """, (asset_path, branch))
                row = await cursor.fetchone()
                if not row:
                    return None
                
                return row["version_id"]


    async def close(self):
        """
        Close the database connection pool.
        """
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()
            logger.info("Database connection pool closed")
