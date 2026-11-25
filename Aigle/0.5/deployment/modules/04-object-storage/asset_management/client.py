import asyncio
import logging
import os
import re
import urllib.parse
from datetime import datetime, time
from typing import BinaryIO, ClassVar
from zoneinfo import ZoneInfo

import asyncpg
from fastapi import HTTPException

from .config import settings
from .database import Database
from .models import (
    AssetMetadata,
    AssetMetadataResponse,
    ExistenceInfo,
    model_to_response,
)
from .object_store import ObjectStore
from .search_sync import SearchSync
from .utils import (
    calculate_md5_bytes,
    detect_file_type,
    parse_absolute_datetime,
    parse_date_or_ttl,
)

logger = logging.getLogger(__name__)

# MTR-8 / VIE01-307: stored checksums are raw MD5 hex (32 lowercase hex chars).
_MD5_HEX_RE = re.compile(r"[0-9a-fA-F]{32}")


class AssetManager:
    def __init__(self, object_store: ObjectStore = None, search_sync: SearchSync = None, db: Database = None):
        self.object_store = ObjectStore() if object_store is None else object_store
        self.db = Database() if db is None else db
        self.search_sync = SearchSync() if search_sync is None else search_sync

    def _sanitize_path(self, path: str) -> str:
        """
        Sanitize a path by removing redundant slashes, stripping leading and trailing slashes, and preventing path traversal.
        
        Args:
            path (str): The path to be sanitized.
        Return: 
            The sanitized path.
        Raises:
            HTTPException: If the path is empty, or if path traversal is detected.
        """
        if not path:
            raise HTTPException(status_code=400, detail="Invalid empty path")
        path = re.sub(r'[\\/]+', '/', path.strip('/'))
        if '..' in path.split('/'):
            raise HTTPException(status_code=400, detail="Path traversal detected")
        return path

    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize a filename by unquoting it, removing any path information, replacing non-alphanumeric characters with underscores, and preventing filenames that are empty or equal to '.' or '..'.
        
        Args:
            filename (str): The filename to be sanitized.
        Returns:
            The sanitized filename.
        Raises:
            HTTPException: If the filename is empty, or if it is equal to '.' or '..' after sanitization.
        """
        if not filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        filename = urllib.parse.unquote(filename).split('/')[-1]
        filename = re.sub(r'[^\w\.\-]', '_', filename)
        if not filename or filename in {'.', '..'}:
            raise HTTPException(status_code=400, detail="Invalid filename")
        return filename

    async def _upload_assoc_batch(self, files, asset_path, branch=None):
        """Upload multiple associated files via object_store.batch_upload(),
        instead of one LakeFS commit per file. Uploads happen on isolated
        ephemeral branches (see object_store._sync_transact_write), so a
        concurrent upload to a different asset_path/filename on the same
        branch can never leak into this batch's commit or steal it -- unlike
        the old one-commit-per-file loop, which shared the branch's staging
        area with every other concurrent writer.

        A large batch may internally span more than one LakeFS transaction
        (see object_store._group_objects_by_size); a file whose group turned
        out byte-identical to what's already on the branch, or whose group
        genuinely failed, falls back to whatever version is already on
        record for that filename.

        Returns [(filename, version_id), ...] -- entries that couldn't be
        resolved at all (no successful upload, and no prior recorded version
        to fall back on) are dropped, never returned as (filename, None); a
        None here would have silently written a null version_id to Postgres.
        """
        if not files:
            return []
        sanitized = [(self._sanitize_filename(fn), data) for data, fn in files]
        objects = [
            {"key": f"{asset_path}/{filename}", "content": data, "content_type": detect_file_type(data, filename)["mime_type"]}
            for filename, data in sanitized
        ]
        key_to_filename = {o["key"]: filename for o, (filename, _) in zip(objects, sanitized)}

        try:
            response = await self.object_store.batch_upload(objects, branch)
            resolved = {f["key"]: f["version_id"] for f in response["files"]}
        except Exception as e:  # noqa: BLE001 -- batch_upload() raises HTTPException or malformed-response errors; fails open to per-file fallback below
            logger.warning(f"Failed to batch-upload {len(files)} associated files to {asset_path}: {e}")
            resolved = {}

        missing_keys = [key for key in key_to_filename if key not in resolved]
        if missing_keys:
            metadata = await self.db.get_latest_asset(asset_path, branch)
            existing = dict(metadata.associated_filenames) if metadata else {}
            dropped = []
            for key in missing_keys:
                filename = key_to_filename[key]
                version_id = existing.get(filename)
                if version_id is not None:
                    resolved[key] = version_id
                else:
                    dropped.append(filename)
            if dropped:
                logger.warning(f"No prior recorded version for {dropped} on {asset_path}, dropping from result")

        return [(key_to_filename[key], version_id) for key, version_id in resolved.items()]

    async def upload_files(
        self,
        username: str,
        branch: str,
        primary_file: tuple[str | BinaryIO, str],
        associated_files: list[tuple[str | BinaryIO, str]] | None = None,
        archive_date_or_ttl: str | int | None = None,
        destroy_date_or_ttl: str | int | None = None,
    ) -> AssetMetadata:
        """
        Upload files and associated files to the object store.
        If files have no change, it will just return the latest version.
        If associated files with the same name already exist and have change, they will be overwritten with the new version.

        Args:
            username (str): username of the user performing the upload.
            branch (str): branch of the asset.
            primary_file (Tuple[Union[str, BinaryIO], str]): primary file to upload.
            associated_files (Optional[List[Tuple[Union[str, BinaryIO], str]]]): list of associated files to upload.
            archive_date_or_ttl (Optional[Union[str, int]]): new archive date or TTL.
            destroy_date_or_ttl (Optional[Union[str, int]]): new destroy date or TTL.

        Returns:
            AssetMetadata: metadata of the uploaded asset.
        """
        associated_files = associated_files or []
        upload_time = datetime.now(tz=ZoneInfo(settings.timezone))

        if archive_date_or_ttl is None and destroy_date_or_ttl is not None:
            raise HTTPException(
                status_code=400, 
                detail="Error: Cannot set a destroy date if the asset is set to be permanently active (archive_date is None). Please set an archive_date or remove the destroy_date."
            )

        if archive_date_or_ttl:
            archive_date = parse_date_or_ttl(archive_date_or_ttl, upload_time, settings.timezone)
        else:
            archive_date = None
        
        if destroy_date_or_ttl:
            destroy_date = parse_date_or_ttl(destroy_date_or_ttl, archive_date, settings.timezone)
        else:
            destroy_date = None

        if archive_date and archive_date <= upload_time:
            raise HTTPException(
                status_code=400, 
                detail="Error: archive_date must be in the future."
            )
        
        if destroy_date and archive_date and destroy_date <= archive_date:
            raise HTTPException(
                status_code=400, 
                detail="Error: destroy_date must be later than archive_date."
            )
        
        # Process primary file
        primary_data, primary_filename = primary_file
        primary_filename = self._sanitize_filename(primary_filename)
        file_info = detect_file_type(primary_data, primary_filename)
        base_path = self._sanitize_path(file_info["base_path"])
        asset_path = self._sanitize_path(f"{base_path}/{os.path.splitext(primary_filename)[0]}")
        primary_asset_path = f"{asset_path}/{primary_filename}"

        # calculate MD5 checksum
        local_md5 = calculate_md5_bytes(primary_data)

        # Global Active Check within the same branch
        global_active = await self.db.get_reusable_content(local_md5, branch, status="active")

        if global_active:
            logger.info(f"Knowledge Hit: Content already active at {global_active.asset_path}")

            if associated_files:
                new_assocs = await self._upload_assoc_batch(associated_files, global_active.asset_path, branch)

                current_assoc_dict = dict(global_active.associated_filenames)
                current_assoc_dict.update(dict(new_assocs))
                global_active.associated_filenames = list(current_assoc_dict.items())
                global_active.user = username
                await self.db.log_access(username, global_active.asset_path, global_active.version_id, branch, "Upload", True, f"Upload existing active asset with associated files: {list(current_assoc_dict.keys())}")
            else:
                await self.db.log_access(username, global_active.asset_path, global_active.version_id, branch, "Upload", True, "Upload existing active asset, just return existing metadata")

            # get_reusable_content() already filtered on status="active" --
            # trust it directly (matches register_asset()'s same trust of
            # `status`) instead of re-verifying via check_indexed(), whose
            # Chunk-only definition of "indexed" false-negatived on
            # summary-only content (e.g. audio with no transcribable
            # speech), spuriously re-triggering analysis on every duplicate
            # upload.
            global_active.existence_info = ExistenceInfo(
                exists=True,
                message=f"Same content already exists at {global_active.asset_path} with version: {global_active.version_id}, reuse it and skipping upload.",
                existing_asset_path=global_active.asset_path,
                existing_version_id=global_active.version_id,
            )
            await self.db.save_metadata(global_active)
            return global_active
        
        # Check if the asset already exists at this path and is archived
        path_record = await self.db.check_path_existence(asset_path, local_md5, branch)
        if path_record and path_record.status == "archived":
            logger.warning(f"Upload blocked: {primary_asset_path} is already archived at this path.")
            await self.db.log_access(username, asset_path, path_record.version_id, branch, "Upload", False, "Upload blocked: Same content at this path already archived")
            raise HTTPException(
                status_code=400,
                detail=(
                    "Asset at this path within your branch is archived with identical content."
                    "Overwriting archived assets is forbidden."
                    "Please delete the archived record or change the file content/path."
                )
            )

        # Same path, same content, but not active and not archived (uploaded /
        # processing / failed / unregistered) -- identical bytes are already
        # committed at this exact LakeFS key, so attempting a real write below
        # would only ever fail with LakeFS's "commit: no changes" (nothing to
        # commit), which used to surface to the caller as a misleading 409
        # ("...has no DB record. Delete the file from LakeFS and re-upload.")
        # even though a DB record exists. Reuse the existing record instead,
        # same shape as the active-reuse branch above -- re-upload was never
        # the right way to retry any of these four states anyway; retrying
        # analysis is POST /processing/process-file's job (accepts uploaded/
        # failed directly; unregistered takes its own fast path; processing
        # is already in flight and just needs to be waited on).
        if path_record and path_record.status in ("uploaded", "processing", "failed", "unregistered"):
            logger.info(f"Knowledge Hit: Identical content already at {path_record.asset_path} with status={path_record.status}")

            if associated_files:
                new_assocs = await self._upload_assoc_batch(associated_files, path_record.asset_path, branch)
                current_assoc_dict = dict(path_record.associated_filenames)
                current_assoc_dict.update(dict(new_assocs))
                path_record.associated_filenames = list(current_assoc_dict.items())
                path_record.user = username
                await self.db.log_access(username, path_record.asset_path, path_record.version_id, branch, "Upload", True, f"Upload existing {path_record.status} asset with associated files: {list(current_assoc_dict.keys())}")
            else:
                await self.db.log_access(username, path_record.asset_path, path_record.version_id, branch, "Upload", True, f"Identical content already exists with status={path_record.status}, reuse it and skipping upload.")

            path_record.existence_info = ExistenceInfo(
                exists=True,
                message=(
                    f"Identical content already exists at {path_record.asset_path} with version: {path_record.version_id} "
                    f"(status: {path_record.status}) -- no need to re-upload; call POST /processing/process-file "
                    f"with this asset_path/version_id to (re)trigger analysis."
                ),
                existing_asset_path=path_record.asset_path,
                existing_version_id=path_record.version_id,
            )
            await self.db.save_metadata(path_record)
            return path_record

        # Globally no active content with the same checksum and locally no archived/reusable content with the same path, execute LakeFS upload.
        try:
            primary_response = await self.object_store.upload_file(
                primary_data, primary_asset_path, file_info["mime_type"], branch
            )
            version_id = primary_response.get("version_id")
            server_checksum = primary_response.get("checksum")
            if server_checksum != local_md5:
                logger.error(f"Checksum mismatch! Local: {local_md5}, Server: {server_checksum}")
                raise HTTPException(status_code=500, detail="Checksum mismatch after upload to LakeFS")
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001 -- upload_file() itself already converts failures to HTTPException; last-resort net for anything else
            logger.error(f"LakeFS Upload failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to upload file to storage")
        
        # Delete old associated files since primary file has changed
        await self.object_store.delete_associated_files(asset_path, primary_filename, branch)
        # Upload associated files
        associated_filenames = await self._upload_assoc_batch(associated_files, asset_path, branch)

        # Global Archive Check within the same branch
        global_archived = await self.db.get_reusable_content(local_md5, branch, status="archived")
        cloned_from = None
        moderation_flags = None
        if  global_archived and global_archived.asset_path != asset_path:
            # Different content but same checksum found globally, we can clone the indexed data to save analysis time
            logger.info(f"Deduplication: Cloning index from {global_archived.asset_path}")
            existence_info = ExistenceInfo(
                exists=True,
                message="Identical content found in another asset path and is archived, we can clone the indexed data and partially modify some fields to save analysis time.",
                existing_asset_path=global_archived.asset_path,
                existing_version_id=global_archived.version_id
            )
            try:
                await self.search_sync.clone_point(
                    source_path=global_archived.asset_path,
                    source_version=global_archived.version_id,
                    target_path=asset_path,
                    target_version=version_id,
                    branch=branch,
                    target_filename=primary_filename,
                )
                # clone_point() above already succeeded (raised otherwise) --
                # that success *is* the confirmation Module 25 has the data,
                # so register_asset() can trust this column directly instead
                # of re-checking Module 25 over the network at Register time.
                cloned_from = global_archived.version_id
                # CM-2: a cloned version never reaches Module 10, and Module 25's
                # clone already copied the Source's moderation_flags -- keep this
                # row in step with it.
                moderation_flags = global_archived.moderation_flags
            except Exception as e:  # noqa: BLE001 -- clone_point() can raise httpx errors in prod; unit tests also exercise generic failures here, must leave cloned_from unset
                logger.error(f"Failed to clone indexed data from {global_archived.asset_path}: {e}")
                existence_info = ExistenceInfo(
                    exists=False,
                    message="Failed to clone indexed data from archived asset."
                )
        else:
            existence_info = ExistenceInfo(
                exists=False,
                message="New asset content"
            )

        # VIE01-214: upload always leaves status="uploaded", no exceptions --
        # even when clone_point() above already cloned Chunk/Source data
        # (status="cloned" there, not "active"). register_asset() checks
        # cloned_from before publishing Kafka, so the saved analysis cost
        # isn't lost; it just isn't claimed here at upload time.
        initial_status = "uploaded"
        metadata = AssetMetadata(
            asset_path=asset_path,
            version_id=version_id,
            primary_filename=primary_filename,
            associated_filenames=associated_filenames,
            upload_date=upload_time,
            archive_date=archive_date,
            destroy_date=destroy_date,
            user=username,
            branch=branch,
            status=initial_status,
            checksum=local_md5,
            cloned_from=cloned_from,
            moderation_flags=moderation_flags,
            existence_info=existence_info
        )

        # Save to MySQL database
        try:
            await self.db.save_metadata(metadata)
        except (asyncpg.PostgresError, asyncpg.InterfaceError) as e:
            logger.error(f"Failed to save metadata for asset {asset_path}/{version_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to save metadata")

        # In the real system, this would not be saved immediately.
        # Instead, it would be combined with other metadata then saved.
        # For testing purposes, you can uncomment the following lines to save it immediately.

        # try:
        #     await self.search_sync.save_or_update_metadata(metadata)
        # except Exception as e:
        #     logger.warning(f"Failed to save metadata to vector store for asset {asset_path}/{version_id}: {e}")

        await self.db.log_access(username, asset_path, version_id, branch, "Upload", True, "Uploaded new asset")
        logger.info(f"Uploaded asset {asset_path}/{version_id} by user {username}")
        return metadata

    async def add_associated_files(
        self,
        username: str,
        branch: str,
        asset_path: str,
        associated_files: list[tuple[str | BinaryIO, str]],
        primary_version_id: str | None = None
    ) -> AssetMetadata:
        """
        Add multiple associated files to an existing asset version.
        If a associated file with the same name already exists, it will overwrite the old associated file.

        Args:
            username (str): Username of the user adding the files.
            branch (str): Branch of the asset.
            asset_path (str): Path of the existing asset (e.g., 'reports/annual_report').
            associated_files (List[Tuple[Union[str, BinaryIO], str]]): List of (file_data, filename) tuples.
            primary_version_id (Optional[str]): Specific version_id to update. 
                If None, uses the latest active version.

        Returns:
            AssetMetadata: Updated metadata including the new associated files.

        Raises:
            HTTPException: If asset not found, user lacks permission, or upload fails.
        """
        if not associated_files:
            raise HTTPException(status_code=400, detail="No associated files provided")

        asset_path = self._sanitize_path(asset_path)

        # Determine which version to update
        if primary_version_id:
            metadata = await self.db.get_asset_by_path_and_version(asset_path, primary_version_id, branch)
            if not metadata:
                raise HTTPException(status_code=404, detail=f"Asset not found for {asset_path}/{primary_version_id}")
        else:
            metadata = await self.db.get_latest_asset(asset_path, branch)
            if not metadata:
                raise HTTPException(
                    status_code=404,
                    detail=f"No active asset found for path: {asset_path}"
                )

        if metadata.status in ("archived", "destroyed"):
            raise HTTPException(
                status_code=400,
                detail=f"Target asset version is {metadata.status}, cannot add associated files"
            )
        
        # Upload associated files
        new_associated_files = await self._upload_assoc_batch(associated_files, asset_path, branch)

        if not new_associated_files:
            raise HTTPException(status_code=500, detail="All associated file uploads failed")

        # Update metadata with new associated files
        associated_filenames = dict(metadata.associated_filenames)
        associated_filenames.update(dict(new_associated_files))
        metadata.associated_filenames = [(k, v) for k, v in associated_filenames.items()]
        metadata.user = username  # Update user who made the change
        metadata.branch = branch

        # Save updated metadata to database
        try:
            await self.db.save_metadata(metadata)
        except (asyncpg.PostgresError, asyncpg.InterfaceError) as e:
            logger.error(f"Failed to update metadata for asset {asset_path}: {e}")  # nosec B608 -- log message, not SQL ("update" is plain English here)
            raise HTTPException(status_code=500, detail="Failed to update asset metadata in database")

        # update vector store metadata
        # In the real system, this would not be updated immediately.
        # Instead, it would be combined with other metadata then updated.
        # For testing purposes, you can uncomment the following lines to save it immediately.

        # try:
        #     await self.search_sync.update_metadata(metadata)
        # except Exception as e:
        #     logger.warning(f"Failed to update vector store metadata for {asset_path}: {e}")

        # Log access
        await self.db.log_access(
            username, asset_path, metadata.version_id, branch, "Add Associated Files", True,
            details=f"Added {len(new_associated_files)} associated files"
        )
        logger.info(f"User {username} added {len(new_associated_files)} associated files to {asset_path}/{metadata.version_id}")
        return metadata

    _IDENTITY_PHOTO_TYPES: ClassVar[dict[str, str]] = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}
    _IDENTITY_PHOTO_MAX_BYTES = 10 * 1024 * 1024
    _IDENTITY_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

    async def upload_identity_photo(self, username: str, branch: str, identity_id: str,
                                    data: bytes, filename: str | None = None) -> dict[str, str | None]:
        """Store a person's seed photo (dev plan v1.4 §4.5 step 2) at the fixed key
        identity/{branch}/{identity_id}.{ext} -- not an asset: no metadata row, no
        status, never indexed. Written through batch_upload() as one LakeFS commit.

        Returns {"image_path", "image_version_id"}: the commit id is the ref
        /presign-url takes. Re-uploading identical bytes makes no new commit
        (batch_upload() then returns nothing for the key); the branch itself is
        returned as the ref, which serves that same content.
        """
        if not self._IDENTITY_ID.match(identity_id or ""):
            raise HTTPException(status_code=400, detail="Invalid identity_id")
        if not data:
            raise HTTPException(status_code=400, detail="Empty photo")
        if len(data) > self._IDENTITY_PHOTO_MAX_BYTES:
            raise HTTPException(status_code=413, detail="Photo larger than 10 MB")
        mime = detect_file_type(data, filename or "")["mime_type"]   # the content decides, not the name
        ext = self._IDENTITY_PHOTO_TYPES.get(mime)
        if not ext:
            raise HTTPException(status_code=415, detail=f"Unsupported photo type {mime}; use JPEG, PNG or WebP")

        key = f"identity/{branch}/{identity_id}.{ext}"
        response = await self.object_store.batch_upload(
            [{"key": key, "content": data, "content_type": mime}], branch,
            commit_message=f"Identity photo {identity_id}")
        version = {f["key"]: f["version_id"] for f in response.get("files", [])}.get(key) or branch
        await self.db.log_access(username, key, version, branch, "Upload Identity Photo", True,
                                 details=f"{len(data)} bytes, {mime}")
        logger.info(f"User {username} stored identity photo {key}@{version}")
        return {"image_path": key, "image_version_id": version}

    async def retrieve_asset(
        self,
        username: str, 
        branch: str, 
        asset_path: str, 
        version_id: str, 
        return_file_content: bool = False
    ) -> dict:
        """
        Retrieve the asset with the given asset_path and version_id.

        Args:
            username (str): Username of the user performing the retrieval.
            branch (str): Branch of the asset.
            asset_path (str): Path of the asset to retrieve.
            version_id (str): Version ID of the asset to retrieve.
            return_file_content (bool, optional): Whether to return the file content. Defaults to False.

        Returns:
            Dict: A dictionary containing the metadata and content of the asset.
        """
        asset_path = self._sanitize_path(asset_path)

        metadata = await self.db.get_asset_by_path_and_version(asset_path, version_id, branch)
        if not metadata:
            await self.db.log_access(username, asset_path, version_id, branch, "retrieve", False, "Asset not found")
            raise HTTPException(status_code=404, detail=f"Asset with asset_path {asset_path} and version_id {version_id} not found")

        metadata = model_to_response(metadata, AssetMetadataResponse)

        # Primary file is always index 0; everything else is an associated
        # file -- same (filename, version_id) shape, fetched the same way,
        # so there's one list and one fetch path instead of separate
        # primary/associated-file handling.
        items: list[tuple[str, str]] = [(metadata.primary_filename, metadata.version_id)] + [
            (filename, item_version_id) for (filename, item_version_id) in metadata.associated_filenames if filename
        ]

        # Resolve as many presigned URLs as possible in one batch (grouped by
        # version_id -- see batch_generate_presigned_urls()) instead of one
        # LakeFS round trip per file; fetch() below falls back to its usual
        # per-file lookup for anything not covered here. Content downloads
        # (return_file_content=True) read actual bytes per file regardless,
        # so batching URLs alone wouldn't help that path.
        presigned_urls: dict[tuple[str, str], dict[str, str | None]] = {}
        if not return_file_content:
            presigned_urls = await self.object_store.batch_generate_presigned_urls(metadata.asset_path, items)

        async def fetch(filename: str, item_version_id: str) -> dict | None:
            # batch_generate_presigned_urls() is best-effort -- a version_id
            # group's LakeFS list call can fail independently of the others,
            # so even under not return_file_content some items may still be
            # unresolved here. Falling through to a real per-file get_file()
            # below (rather than treating "not in presigned_urls" as "doesn't
            # exist") is what keeps a partial batch failure from silently
            # dropping files, same as the original bug this fixes.
            batch_result = presigned_urls.get((filename, item_version_id))
            if batch_result:
                return {"filename": filename, "content_type": batch_result["content_type"], "version_id": item_version_id, "url": batch_result["url"]}
            path = f'{metadata.asset_path}/{filename}'
            try:
                file = await self.object_store.get_file(path, version_id=item_version_id, return_file_content=return_file_content)
            except HTTPException as e:
                logger.error(f"Failed to retrieve file {path}: {e}")
                return None
            result = {"filename": filename, "content_type": file["content_type"], "version_id": file["version_id"], "url": file["url"]}
            if return_file_content:
                result["content"] = file["content"]
            return result

        fetched = await asyncio.gather(*[fetch(filename, item_version_id) for filename, item_version_id in items])

        primary_file_data = fetched[0]
        if not primary_file_data:
            await self.db.log_access(username, asset_path, version_id, branch, "retrieve", False, f"Primary file {metadata.primary_filename} not found in object store")
            raise HTTPException(status_code=404, detail=f"Primary file {metadata.primary_filename} not found")

        results = {"metadata": metadata.model_dump(), "primary_file": primary_file_data}
        for idx, file in enumerate(fetched[1:], 1):
            if file:
                results[f"associated_file_{idx}"] = file

        await self.db.log_access(username, asset_path, version_id, branch, "retrieve", True, "Retrieved asset")
        logger.info(f"Retrieved asset {asset_path}/{version_id} by user {username}")
        return results

    async def register_asset(self, branch: str, asset_path: str, version_id: str) -> dict:
        """VIE01-215: the Register API's status-branching logic.

        Looks up the asset's current status and returns one of:
        - {"outcome": "not_found"} -- caller (module 13) returns 404.
        - {"outcome": "conflict", "status": ...} -- processing/active/
          archived/destroyed, none of which should be (re-)dispatched for
          analysis; caller returns 409.
        - {"outcome": "registered", "metadata": ...} -- either was
          "unregistered", or was "uploaded"/"failed" with a clone_point()
          clone already waiting in Module 25; either way Module 25 confirms
          the content is indexed and the asset flips to "active"
          synchronously, no Kafka needed.
        - {"outcome": "needs_processing", "metadata": ...} -- "uploaded" or
          "failed" with no clone waiting; status is left unchanged (module 13
          publishes Kafka; the claiming orchestrator is what actually writes
          "processing", via POST /claim-processing).
        """
        metadata = await self.db.get_asset_by_path_and_version(asset_path, version_id, branch)
        if metadata is None:
            return {"outcome": "not_found"}

        if metadata.status in ("processing", "active", "archived", "destroyed"):
            return {"outcome": "conflict", "status": metadata.status}

        if metadata.status == "unregistered":
            # Module 25 is the source of truth that this content is actually
            # indexed under this asset_path/version_id -- only flip to
            # "active" once it confirms, rather than assuming the earlier
            # clone_point() at upload time is still valid.
            await self.search_sync._personal_db_set_status(asset_path, version_id, "active", branch_id=branch)
            await self.db.update_status(asset_path, version_id, "active", branch)
            metadata.status = "active"
            return {"outcome": "registered", "metadata": metadata}

        # "uploaded" / "failed" -- cloned_from is set only when clone_point()
        # itself already succeeded at upload time (it raises otherwise), so
        # that success is the confirmation; no need to ask Module 25 again
        # here. Flip straight to "active" instead of paying for a full
        # re-analysis of content that's already indexed.
        if metadata.cloned_from:
            await self.search_sync._personal_db_set_status(asset_path, version_id, "active", branch_id=branch)
            await self.db.update_status(asset_path, version_id, "active", branch)
            metadata.status = "active"
            return {"outcome": "registered", "metadata": metadata}

        # Otherwise leave status untouched; the orchestrator that actually
        # claims the job (POST /claim-processing) is what writes
        # "processing", not Register itself.
        return {"outcome": "needs_processing", "metadata": metadata}

    async def unregister_asset(self, username: str, branch: str, asset_path: str, version_id: str) -> AssetMetadata:
        """MTR-5 / VIE01-305: the reverse of register_asset's "unregistered"
        fast path -- active -> unregistered. Only legal from "active" (409
        otherwise, including from "unregistered" itself -- not idempotent,
        same as archive()'s guard).

        Both steps run synchronously and in this order, matching the
        ticket's own requirement: Module 25's index status flips first (set
        status="unregistered" on the existing Chunk/Source/TemporalFact
        rows), then Postgres -- so a concurrent reader never observes
        "active" in Postgres while Module 25 already thinks it's
        unregistered. Deliberately calls the same status-set endpoint
        register_asset's fast path already uses (_personal_db_set_status),
        never the index-delete endpoint -- no embedding/vector/graph data is
        touched, only a status column on rows that stay fully indexed.
        """
        asset_path = self._sanitize_path(asset_path)
        metadata = await self.db.get_asset_by_path_and_version(asset_path, version_id, branch)

        if not metadata:
            await self.db.log_access(username, asset_path, version_id, branch, "Unregister", False, "Asset not found")
            raise HTTPException(status_code=404, detail=f"Asset with asset_path {asset_path} and version_id {version_id} not found")
        if metadata.status != "active":
            await self.db.log_access(username, asset_path, version_id, branch, "Unregister", False, f"Asset is {metadata.status}, not active")
            raise HTTPException(status_code=409, detail=f"Asset {asset_path}/{version_id} is {metadata.status}, not active -- cannot unregister")

        await self.search_sync._personal_db_set_status(asset_path, version_id, "unregistered", branch_id=branch)
        await self.db.update_status(asset_path, version_id, "unregistered", branch)
        metadata.status = "unregistered"

        await self.db.log_access(username, asset_path, version_id, branch, "Unregister", True, "Unregistered asset successfully")
        logger.info(f"Unregistered asset {asset_path}/{version_id} by user {username}")
        return metadata

    async def archive(self, username: str, branch: str, asset_path: str, version_id: str) -> AssetMetadata:
        """
        Archive the asset with the given asset_path and version_id.

        Args:
            username (str): Username of the user performing the archiving.
            branch (str): Branch of the asset.
            asset_path (str): Path of the asset to archive.
            version_id (str): Version ID of the asset to archive.

        Returns:
            AssetMetadata: The archived asset metadata.

        Raises:
            HTTPException: If the asset is not found or is not active.
        """
        asset_path = self._sanitize_path(asset_path)
        metadata = await self.db.get_asset_by_path_and_version(asset_path, version_id, branch)

        if not metadata:
            await self.db.log_access(username, asset_path, version_id, branch, "Archive", False, "Asset not found")
            raise HTTPException(status_code=404, detail=f"Asset with asset_path {asset_path} and version_id {version_id} not found")
        if metadata.status in ("archived", "destroyed"):
            await self.db.log_access(username, asset_path, version_id, branch, "Archive", False, f"Asset is {metadata.status}")
            raise HTTPException(status_code=400, detail=f"Asset {asset_path}/{version_id} is already {metadata.status}")

        await self.db.update_status(asset_path, version_id, "archived", branch)
        try:
            await self.search_sync.archive_metadata(asset_path, version_id, branch)
        except Exception as e:  # noqa: BLE001 -- archive_metadata() gathers calls whose own handlers already log+swallow; net for anything unexpected
            logger.error(f"Vector store archive failed for asset {asset_path}/{version_id}: {e}")

        await self.db.log_access(username, asset_path, version_id, branch, "Archive", True, "Archived asset successfully")
        logger.info(f"Archived asset {asset_path}/{version_id} by user {username}")
        
        # Check if the asset is archived completely
        while True:
            metadata = await self.db.get_asset_by_path_and_version(asset_path, version_id, branch)
            if metadata.status == "archived":
                break
        return metadata

    async def update_status(self, username: str, branch: str, asset_path: str, version_id: str, status: str) -> str:
        """VIE01-255: write back a claimed job's final outcome (active/failed).
        Called by the 09-12 orchestrators via POST /update-status once
        processing finishes, success or failure.
        """
        asset_path = self._sanitize_path(asset_path)
        metadata = await self.db.get_asset_by_path_and_version(asset_path, version_id, branch)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Asset with asset_path {asset_path} and version_id {version_id} not found")

        await self.db.update_status(asset_path, version_id, status, branch)
        logger.info(f"Updated status={status} for asset {asset_path}/{version_id} by {username}")
        return status

    async def update_moderation(self, username: str, branch: str, asset_path: str, version_id: str, flags: dict) -> None:
        """CM-2: store the image moderation verdict Module 10 reports via
        POST /update-moderation."""
        asset_path = self._sanitize_path(asset_path)
        if not await self.db.update_moderation(asset_path, version_id, branch, flags):
            raise HTTPException(status_code=404, detail=f"Asset with asset_path {asset_path} and version_id {version_id} not found")
        logger.info(f"Updated moderation_flags ({flags.get('verdict')}) for asset {asset_path}/{version_id} by {username}")

    async def auto_archive(self, current_date: datetime | None = None) -> list[AssetMetadata]:
        """
        Archive assets whose archive_date has passed.

        This function is called automatically by the scheduler daily.

        Args:
            current_date (datetime): The current date and time in the Asia/Taipei timezone.
                Defaults to the current UTC time.

        Returns:
            List[AssetMetadata]: A list of AssetMetadata objects that represent the archived assets.
        """
        current_date = current_date or datetime.now(tz=ZoneInfo(settings.timezone))
        
        assets = await self.db.get_assets_to_archive(current_date)
        if not assets:
            logger.info("No assets to archive")
            return []

        async def safe_archive(asset):
            try:
                return await self.archive("scheduler", asset.branch, asset.asset_path, asset.version_id)
            except Exception as e:  # noqa: BLE001 -- archive() can raise HTTPException or any underlying db/store error; batch job logs and skips this asset
                logger.info(f"Auto-archive failed for {asset.asset_path}/{asset.version_id}: {e}")
                return None
        
        tasks = [safe_archive(asset) for asset in assets]
        archived_assets = await asyncio.gather(*tasks)
        return [asset for asset in archived_assets if asset is not None]
    
    async def destroy_v2(self, username: str, branch: str, asset_path: str, version_id: str) -> AssetMetadata:
        """
        Destroy the asset with the given asset_path and version_id.

        Args:
            username (str): Username of the user performing the destruction.
            asset_path (str): Path of the asset to destroy.
            version_id (str): Version ID of the asset to destroy.

        Returns:
            AssetMetadata: The destroyed asset metadata.

        Raises:
            HTTPException: If the asset is not found or is not archived.
        """
        asset_path = self._sanitize_path(asset_path)

        metadata = await self.db.get_asset_by_path_and_version(asset_path, version_id, branch)
        if not metadata:
            await self.db.log_access(username, asset_path, version_id, branch, "Destroy", False, "Asset not found")
            raise HTTPException(status_code=404, detail=f"Asset with asset_path {asset_path} and version_id {version_id} not found")
        if metadata.status != "archived":
            await self.db.log_access(username, asset_path, version_id, branch, "Destroy", False, f"Asset is {metadata.status}")
            raise HTTPException(status_code=400, detail=f"Asset {asset_path}/{version_id} is not archived")

        head_version = await self.db.get_head_version(asset_path, branch)
        is_head = head_version == metadata.version_id

        primary_path = f"{asset_path}/{metadata.primary_filename}"
        files_to_delete = [(primary_path, metadata.version_id)] + [
            (f"{asset_path}/{filename}", assoc_version_id)
            for filename, assoc_version_id in metadata.associated_filenames
            if filename
        ]
        try:
            await self.object_store.bulk_delete_files_v2(files_to_delete, is_head=is_head, branch=branch, asset_path=asset_path)
        except Exception as e:  # noqa: BLE001 -- can raise botocore errors in prod; unit tests also exercise generic storage failures here, must not block metadata deletion below
            logger.error(f"Failed to bulk-delete files for {asset_path}/{version_id}: {e}")

        await self.db.delete_metadata(asset_path, version_id, branch)
        try:
            await self.search_sync.delete_metadata(asset_path, version_id, branch)
        except Exception as e:  # noqa: BLE001 -- delete_metadata()'s own calls already log+swallow httpx errors; net for anything unexpected
            logger.error(f"Vector store delete failed for asset {asset_path}/{version_id}: {e}")

        await self.db.log_access(username, asset_path, version_id, branch, "Destroy", True, "Destroyed asset successfully")
        logger.info(f"Destroyed asset {asset_path}/{version_id} by user {username}")
        metadata.status = "destroyed"
        return metadata

    async def gdpr_summarize_user(self, branch: str) -> dict:
        """GD-4 / VIE01-249: counts-only file summary for Module 33's DSAR --
        Postgres metadata only, no LakeFS/S3 access at all."""
        return await self.db.get_gdpr_file_summary(branch)

    async def gdpr_list_files(self, branch: str, expires_in: int, concurrency: int = 8) -> list[dict]:
        """GD-2 / VIE01-247: one entry per stored file (primary and associated)
        of every asset on this branch, any status -- metadata plus a download
        link valid for expires_in seconds, never the bytes themselves (the
        ticket keeps file content out of the export ZIP). A file whose
        LakeFS stat fails is still listed, with size/type/download_url null
        and `error` set, so the index never silently drops a file."""
        assets = await self.db.get_assets_by_branch(branch)
        semaphore = asyncio.Semaphore(concurrency)

        async def _entry(metadata: AssetMetadata, filename: str, version_id: str, role: str) -> dict:
            entry = {
                "name": filename, "asset_path": metadata.asset_path, "version_id": version_id, "role": role,
                "status": metadata.status, "uploaded_at": metadata.upload_date,
                "size": None, "type": None, "download_url": None,
            }
            async with semaphore:
                try:
                    entry.update(await self.object_store.gdpr_file_link(
                        f"{metadata.asset_path}/{filename}", version_id, expires_in,
                    ))
                except Exception as e:  # noqa: BLE001 -- lakefs stat()/S3 presign errors aren't one type; the file stays listed with the error
                    logger.error(f"GDPR export: no download link for {metadata.asset_path}/{filename}: {e}")
                    entry["error"] = "file not found in object storage"
            return entry

        jobs = []
        for metadata in assets:
            jobs.append(_entry(metadata, metadata.primary_filename, metadata.version_id, "primary"))
            for name, version_id in metadata.associated_filenames or []:
                if name:
                    jobs.append(_entry(metadata, name, version_id or metadata.version_id, "associated"))
        return await asyncio.gather(*jobs)

    async def scan_assets(self, files: list[tuple[str, str]], branch: str, concurrency: int = 10) -> list[dict]:
        """MTR-8 / VIE01-307: batch duplicate-content check -- for each
        (path, checksum) pair, reports whether that checksum already
        exists for this branch (any status), without uploading anything.

        Checksum format: stored checksums are raw MD5 hex (calculate_md5_bytes,
        see upload_files) -- the ticket's own "sha256:..." example doesn't
        match that. Accepts an optional "algo:" prefix for forward
        compatibility (stripped, never checked against the actual
        algorithm name) and validates/compares the remaining hex against
        the stored MD5 value. 400 if the hex part is missing or malformed.
        """
        for _, checksum in files:
            hex_part = checksum.split(":", 1)[-1]
            if not _MD5_HEX_RE.fullmatch(hex_part):
                raise HTTPException(status_code=400, detail=f"Malformed checksum: {checksum!r}")

        normalized = [(path, checksum.split(":", 1)[-1].lower()) for path, checksum in files]
        return await self.db.scan_assets(normalized, branch, concurrency=concurrency)

    async def gdpr_erase_user(self, branch: str) -> int:
        """GD-1 / VIE01-246: unconditionally purge every asset belonging to
        this branch (= user_id), regardless of status.

        Deliberately does not reuse destroy_v2 -- that method requires the
        asset already be "archived" first (a safety guard appropriate for
        normal user-initiated deletion, where archive-then-destroy is a
        two-step workflow). GDPR erasure must delete an asset no matter its
        lifecycle state (uploaded, processing, active, failed, unregistered,
        archived), so this skips that guard entirely.

        Note this codebase has no per-user LakeFS branch to simply drop --
        every user's files live on one shared branch (settings.lakefs_branch,
        "main"); `branch` here is purely the Postgres-level tenant tag. So
        erasure means iterating this user's individual asset rows and
        deleting each one's objects, not a branch-level operation.
        """
        assets = await self.db.get_assets_by_branch(branch)
        deleted_count = 0
        for metadata in assets:
            head_version = await self.db.get_head_version(metadata.asset_path, branch)
            is_head = head_version == metadata.version_id

            primary_path = f"{metadata.asset_path}/{metadata.primary_filename}"
            files_to_delete = [(primary_path, metadata.version_id)] + [
                (f"{metadata.asset_path}/{filename}", assoc_version_id)
                for filename, assoc_version_id in metadata.associated_filenames
                if filename
            ]
            try:
                await self.object_store.bulk_delete_files_v2(
                    files_to_delete, is_head=is_head, branch=branch, asset_path=metadata.asset_path
                )
            except Exception as e:  # noqa: BLE001 -- can raise botocore errors in prod; must not block metadata deletion below
                logger.error(f"[gdpr] Failed to bulk-delete files for {metadata.asset_path}/{metadata.version_id}: {e}")

            await self.db.delete_metadata(metadata.asset_path, metadata.version_id, branch)
            try:
                await self.search_sync.delete_metadata(metadata.asset_path, metadata.version_id, branch)
            except Exception as e:  # noqa: BLE001 -- delete_metadata()'s own calls already log+swallow httpx errors; net for anything unexpected
                logger.error(f"[gdpr] Vector store delete failed for {metadata.asset_path}/{metadata.version_id}: {e}")

            await self.db.log_access(
                "gdpr-erasure", metadata.asset_path, metadata.version_id, branch,
                "Destroy", True, "Destroyed via GDPR erasure request",
            )
            deleted_count += 1

        logger.info(f"[gdpr] Erased {deleted_count} asset(s) for branch {branch}")
        return deleted_count

    async def auto_destroy(self, current_date: datetime | None = None) -> list[AssetMetadata]:
        """
        Destroy assets whose destroy_date has passed.

        This function is called automatically by the scheduler daily.

        Args:
            current_date (datetime): The current date and time in the Asia/Taipei timezone.
                Defaults to the current UTC time.

        Returns:
            List[AssetMetadata]: A list of AssetMetadata objects that represent the destroyed assets.
        """
        current_date = current_date or datetime.now(tz=ZoneInfo(settings.timezone))

        # Destroy logs older than 120 days (not recommended in real system, just for testing purpose)
        # await self.db.cleanup_old_logs(current_date, 120)

        assets = await self.db.get_assets_to_destroy(current_date)
        if not assets:
            logger.info("No assets to destroy")
            return []
        
        async def safe_destroy(asset):
            try:
                return await self.destroy_v2("scheduler", asset.branch, asset.asset_path, asset.version_id)
            except Exception as e:  # noqa: BLE001 -- destroy_v2() can raise HTTPException or any underlying db/store error; batch job logs and skips this asset
                logger.info(f"Auto-destroy failed for {asset.asset_path}/{asset.version_id}: {e}")
                return None
            
        tasks = [safe_destroy(asset) for asset in assets]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]

    async def list_file_versions(self, username: str, key: str, branch: str) -> list[dict]:
        """
        List all versions of a file in the bucket.

        Args:
            username (str): The username of the user performing the operation.
            key (str): The key of the object in the bucket.
            branch (str): The branch of the object in the bucket.

        Returns:
            List[Dict]: A list of dictionaries containing information about each version of the file.
                Each dictionary contains the keys "key", "version_id", "last_modified", and "url".
        """
        key = self._sanitize_path(key)
        base_path = key.rsplit("/", 1)[0] if "/" in key else key
        logger.info(f"Listing versions for key: {key} by user {username}")

        versions_info = await self.db.get_versions_by_key(key, branch)
        versions = []

        async def process_version(info):
            try:
                url = await self.object_store.generate_presigned_url(key, version_id=info["version_id"])
                return {
                    "key": key,
                    "version_id": info["version_id"],
                    "upload_date": info["upload_date"],
                    "status": info["status"],
                    "url": url
                }

            except HTTPException as e:
                await self.db.log_access(username, info["asset_path"], info["version_id"], branch, "List File Versions", False, str(e))
                logger.warning(f"Access denied for version {key}/{info['version_id']}: {e.detail}")
                return None

        tasks = [process_version(info) for info in versions_info]
        results = await asyncio.gather(*tasks)
        versions = [r for r in results if r is not None]

        await self.db.log_access(username, base_path, "", branch, "List File Versions", True, f"Found {len(versions)} versions")
        logger.info(f"Found {len(versions)} versions for {key} by user {username}")
        return versions

    async def get_user_commits(
        self,
        username: str,
        branch: str,
        keyword: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        page: int = 1,
        page_size: int = 50,
        statuses: list[str] | None = None,
    ) -> list[AssetMetadataResponse]:
        """
        Get the commit history for a user.

        Args:
            username (str): The username of the user.
            branch (str): The branch to filter the history.
            page (int, optional): The page number for pagination. Defaults to 1.
            page_size (int, optional): The number of records per page. Defaults to 50.

        Returns:
            List[AssetMetadataResponse]: A list of AssetMetadataResponse objects representing the user's commit history.
        """
        if start_date:
            start_date = parse_absolute_datetime(start_date, settings.timezone)
            start_date = datetime.combine(start_date.date(), time.min).replace(tzinfo=ZoneInfo(settings.timezone))

        if end_date:
            end_date = parse_absolute_datetime(end_date, settings.timezone)
            end_date = datetime.combine(end_date.date(), time.max).replace(tzinfo=ZoneInfo(settings.timezone))

        commits = await self.db.get_user_commits(
            username=username,
            branch=branch,
            keyword=keyword,
            start_date=start_date,
            end_date=end_date,
            page=page,
            page_size=page_size,
            statuses=statuses,
        )
        await self.db.log_access(username, "", "", branch, "Get User Commits", True, f"Requested page {page} with size {page_size}")
        
        return commits

    async def update_expiration(
        self,
        username: str,
        branch: str,
        asset_path: str,
        version_id: str,
        archive_date_or_ttl: int | str | None,
        destroy_date_or_ttl: int | str | None,
    ) -> AssetMetadata:
        """
        Update the expiration dates for an asset.

        Args:
            username (str): The username of the user performing the update.
            branch (str): The branch of the asset.
            asset_path (str): The path of the asset.
            version_id (str): The version ID of the asset.
            archive_date_or_ttl (Union[int, str]): New archive date or TTL.
            destroy_date_or_ttl (Union[int, str]): New destroy date or TTL.

        Returns:
            AssetMetadata: The updated asset metadata.
        """
        asset_path = self._sanitize_path(asset_path)

        metadata = await self.db.get_asset_by_path_and_version(asset_path, version_id, branch)
        if not metadata:
            raise HTTPException(status_code=404, detail="Asset not found")

        if metadata.status == "archived":
            # Reactivation is not allowed
            logger.warning(f"Reactivation attempt blocked for {asset_path}/{version_id}")
            raise HTTPException(
                status_code=400, 
                detail="Asset is already archived and cannot be reactivated or modified. "
                        "You can only delete the asset."
            )

        now = datetime.now(tz=ZoneInfo(settings.timezone))

        if archive_date_or_ttl == "no_change" and destroy_date_or_ttl == "no_change":
            raise HTTPException(status_code=400, detail="No changes specified for archive or destroy date, please specify at least one.")

        # Process Archive Date
        if archive_date_or_ttl is not None:
            if archive_date_or_ttl == "no_change":
                new_archive_date = metadata.archive_date
            else:
                new_archive_date = parse_date_or_ttl(archive_date_or_ttl, now, settings.timezone)
        else:
            new_archive_date = None

        # Process Destroy Date
        if destroy_date_or_ttl is not None:
            if destroy_date_or_ttl == "no_change":
                new_destroy_date = metadata.destroy_date
            else:
                reference_date = new_archive_date or now
                new_destroy_date = parse_date_or_ttl(destroy_date_or_ttl, reference_date, settings.timezone)
        else:
            new_destroy_date = None
        
        # Validate
        if new_archive_date is None and new_destroy_date is not None:
            raise HTTPException(status_code=400, detail="Permanent active assets cannot have a destroy date. Please specify an archive date or remove the setting for destroy date.")

        if new_archive_date:
            if new_archive_date <= now:
                raise HTTPException(status_code=400, detail="New Archive date must be in the future")

            if new_destroy_date and new_destroy_date <= new_archive_date:
                raise HTTPException(status_code=400, detail="Destroy date must be after archive date")

        success = await self.db.update_asset_expiration(
            asset_path=asset_path,
            version_id=version_id,
            branch=branch,
            new_archive_date=new_archive_date,
            new_destroy_date=new_destroy_date,
            target_status=metadata.status
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to update expiration dates")

        # Update the metadata
        metadata.archive_date = new_archive_date
        metadata.destroy_date = new_destroy_date

        await self.db.log_access(username, asset_path, version_id, branch, "Update Expiration", True, f"Updated archive date to {new_archive_date}, destroy date to {new_destroy_date}")
        return metadata