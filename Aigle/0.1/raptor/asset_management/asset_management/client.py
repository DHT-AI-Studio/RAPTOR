import asyncio
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo
from typing import List, Optional, Tuple, Union, Dict, BinaryIO
from fastapi import HTTPException
import urllib.parse
import os
import logging
import re

from .models import AssetMetadata, AssetMetadataResponse, ExistenceInfo, model_to_response
from .database import Database
from .vector_store import VectorStore
from .object_store import ObjectStore
from .utils import detect_file_type, parse_date_or_ttl, parse_absolute_datetime, calculate_md5_bytes
from .config import settings

logger = logging.getLogger(__name__)


class AssetManager:
    def __init__(self, object_store: ObjectStore = None, vector_store: VectorStore = None, db: Database = None):
        self.object_store = ObjectStore() if object_store is None else object_store
        self.db = Database() if db is None else db
        self.vector_store = VectorStore() if vector_store is None else vector_store

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

    async def _upload_assoc_task(self, data, asset_path, filename, branch=None):
        filename = self._sanitize_filename(filename)
        file_info = detect_file_type(data, filename)
        assoc_path = f"{asset_path}/{filename}"
        try:
            response = await self.object_store.upload_file(
                data, assoc_path, file_info["mime_type"], branch
            )
            assoc_version_id = response.get("version_id")
            return (filename, assoc_version_id)
        except Exception as e:
            if e.status_code == 400: # File no changed, return latest version
                metadata = await self.db.get_latest_asset(asset_path, branch)
                if metadata:
                    associated_filenames = dict(metadata.associated_filenames)
                    return (filename, associated_filenames.get(filename))
            logger.warning(f"Failed to upload associated file {filename}: {e}")
            return None

    async def upload_files(
        self,
        username: str,
        branch: str,
        primary_file: Tuple[Union[str, BinaryIO], str],
        associated_files: Optional[List[Tuple[Union[str, BinaryIO], str]]] = None,
        archive_date_or_ttl: Optional[Union[str, int]] = None,
        destroy_date_or_ttl: Optional[Union[str, int]] = None,
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
                tasks = [self._upload_assoc_task(data, global_active.asset_path, filename, branch) 
                         for data, filename in associated_files]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                new_assocs = [r for r in results if r is not None and not isinstance(r, Exception)]

                current_assoc_dict = dict(global_active.associated_filenames)
                current_assoc_dict.update(dict(new_assocs))
                global_active.associated_filenames = list(current_assoc_dict.items())
                global_active.user = username
                await self.db.log_access(username, global_active.asset_path, global_active.version_id, branch, "Upload", True, f"Upload existing active asset with associated files: {list(current_assoc_dict.keys())}")
            else:
                await self.db.log_access(username, global_active.asset_path, global_active.version_id, branch, "Upload", True, "Upload existing active asset, just return existing metadata")

            global_active.existence_info = ExistenceInfo(
                exists=True, 
                message=f"Same content already exists at {global_active.asset_path} with version: {global_active.version_id}, reuse it and skipping upload.",
                existing_asset_path=global_active.asset_path,
                existing_version_id=global_active.version_id
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

        # Globally no active content with the same checksum and locally no archived content with the same path, execute LakeFS upload.
        try:
            primary_response = await self.object_store.upload_file(
                primary_data, primary_asset_path, file_info["mime_type"], branch
            )
            version_id = primary_response.get("version_id")
            server_checksum = primary_response.get("checksum")
            if server_checksum != local_md5:
                logger.error(f"Checksum mismatch! Local: {local_md5}, Server: {server_checksum}")
                raise HTTPException(status_code=500, detail="Checksum mismatch after upload to LakeFS")
        except Exception as e:
            logger.error(f"LakeFS Upload failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to upload file to storage")        
        
        # Delete old associated files since primary file has changed
        await self.object_store.delete_associated_files(asset_path, primary_filename, branch)
        # Upload associated files
        tasks = [self._upload_assoc_task(data, asset_path, filename, branch) for data, filename in associated_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        associated_filenames = [r for r in results if r is not None]

        # Global Archive Check within the same branch
        global_archived = await self.db.get_reusable_content(local_md5, branch, status="archived")
        if  global_archived and global_archived.asset_path != asset_path:
            # Different content but same checksum found globally, we can clone the vector data to save analysis time
            logger.info(f"Deduplication: Cloning vector from {global_archived.asset_path}")
            existence_info = ExistenceInfo(
                exists=True,
                message="Identical content found in another asset path and is archived, we can clone the qdrant data and partially modify some fields to save analysis time.",
                existing_asset_path=global_archived.asset_path,
                existing_version_id=global_archived.version_id
            )
            try:
                await self.vector_store.clone_point(
                    source_path=global_archived.asset_path,
                    source_version=global_archived.version_id,
                    target_path=asset_path,
                    target_version=version_id,
                    branch=branch
                )
            except Exception as e:
                logger.error(f"Failed to clone vector data from {global_archived.asset_path}: {e}")
                existence_info = ExistenceInfo(
                    exists=False,
                    message="Failed to clone vector data from archived asset."
                )
        else:
            existence_info = ExistenceInfo(
                exists=False,
                message="New asset content"
            )

        # Save metadata
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
            status="active",
            checksum=local_md5,
            existence_info=existence_info
        )

        # Save to MySQL database
        try:
            await self.db.save_metadata(metadata)
        except Exception as e:
            logger.error(f"Failed to save metadata for asset {asset_path}/{version_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to save metadata")

        # In the real system, this would not be saved immediately.
        # Instead, it would be combined with other metadata then saved.
        # For testing purposes, you can uncomment the following lines to save it immediately.

        # try:
        #     await self.vector_store.save_or_update_metadata(metadata)
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
        associated_files: List[Tuple[Union[str, BinaryIO], str]],
        primary_version_id: Optional[str] = None
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

        if metadata.status != "active":
            raise HTTPException(
                status_code=400,
                detail=f"Target asset version is not active (status: {metadata.status})"
            )
        
        # Upload all associated files concurrently
        tasks = [self._upload_assoc_task(file_data, asset_path, filename, branch) for file_data, filename in associated_files]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        new_associated_files = [r for r in results if r is not None]

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
        except Exception as e:
            logger.error(f"Failed to update metadata for asset {asset_path}: {e}")
            raise HTTPException(status_code=500, detail="Failed to update asset metadata in database")

        # update vector store metadata
        # In the real system, this would not be updated immediately.
        # Instead, it would be combined with other metadata then updated.
        # For testing purposes, you can uncomment the following lines to save it immediately.

        # try:
        #     await self.vector_store.update_metadata(metadata)
        # except Exception as e:
        #     logger.warning(f"Failed to update vector store metadata for {asset_path}: {e}")

        # Log access
        await self.db.log_access(
            username, asset_path, metadata.version_id, branch, "Add Associated Files", True,
            details=f"Added {len(new_associated_files)} associated files"
        )
        logger.info(f"User {username} added {len(new_associated_files)} associated files to {asset_path}/{metadata.version_id}")
        return metadata

    async def retrieve_asset(
        self,
        username: str, 
        branch: str, 
        asset_path: str, 
        version_id: str, 
        return_file_content: bool = False
    ) -> Dict:
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
        results = {"metadata": metadata.model_dump()}

        async def fetch_file(filename: str, version_id: str, return_file_content: bool = False) -> Optional[Dict]:
            path = f'{metadata.asset_path}/{filename}'
            try:
                file = await self.object_store.get_file(path, version_id=version_id, return_file_content=return_file_content)
                if return_file_content:
                    return {
                        "filename": filename,
                        "content": file["content"],
                        "content_type": file["content_type"],
                        "version_id": file["version_id"],
                        "url": file["url"]
                    }
                else:
                    return {
                        "filename": filename,
                        "content_type": file["content_type"],
                        "version_id": file["version_id"],
                        "url": file["url"]
                    }
            except Exception as e:
                logger.error(f"Failed to retrieve file {path}: {e}")
                return None

        primary_file_data = await fetch_file(metadata.primary_filename, metadata.version_id, return_file_content)
        if not primary_file_data:
            await self.db.log_access(username, asset_path, version_id, branch, "retrieve", False, f"Primary file {metadata.primary_filename} not found in object store")
            raise HTTPException(status_code=404, detail=f"Primary file {metadata.primary_filename} not found")
        results["primary_file"] = primary_file_data

        # Retrieve associated files
        assoc_tasks = [
            fetch_file(filename, version_id, return_file_content)
            for (filename, version_id) in metadata.associated_filenames
            if filename
        ]
        assoc_results = await asyncio.gather(*assoc_tasks)
        for idx, file in enumerate(assoc_results, 1):
            if file:
                results[f"associated_file_{idx}"] = file

        await self.db.log_access(username, asset_path, version_id, branch, "retrieve", True, "Retrieved asset")
        logger.info(f"Retrieved asset {asset_path}/{version_id} by user {username}")
        return results

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
        if metadata.status != "active":
            await self.db.log_access(username, asset_path, version_id, branch, "Archive", False, f"Asset is {metadata.status}")
            raise HTTPException(status_code=400, detail=f"Asset {asset_path}/{version_id} is already {metadata.status}")

        await self.db.update_status(asset_path, version_id, "archived", branch)
        try:
            await self.vector_store.archive_metadata(asset_path, version_id, branch)
        except Exception as e:
            logger.error(f"Vector store archive failed for asset {asset_path}/{version_id}: {e}")

        await self.db.log_access(username, asset_path, version_id, branch, "Archive", True, "Archived asset successfully")
        logger.info(f"Archived asset {asset_path}/{version_id} by user {username}")
        
        # Check if the asset is archived completely
        while True:
            metadata = await self.db.get_asset_by_path_and_version(asset_path, version_id, branch)
            if metadata.status == "archived":
                break   
        return metadata

    async def auto_archive(self, current_date: Optional[datetime] = None) -> List[AssetMetadata]:
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
            except Exception as e:
                logger.info(f"Auto-archive failed for {asset.asset_path}/{asset.version_id}: {e}")
                return None
        
        tasks = [safe_archive(asset) for asset in assets]
        archived_assets = await asyncio.gather(*tasks)
        return [asset for asset in archived_assets if asset is not None]
    
    async def destroy(self, username: str, branch: str, asset_path: str, version_id: str) -> AssetMetadata:
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
        async def delete_assoc(filename, version_id, branch=None):
            path = f"{asset_path}/{filename}"
            try:
                await self.object_store.delete_file(path, version_id=version_id, branch=branch)
            except Exception as e:
                logger.error(f"Failed to delete associated file {path}: {e}")

        asset_path = self._sanitize_path(asset_path)

        metadata = await self.db.get_asset_by_path_and_version(asset_path, version_id, branch)
        if not metadata:
            await self.db.log_access(username, asset_path, version_id, branch, "Destroy", False, "Asset not found")
            raise HTTPException(status_code=404, detail=f"Asset with asset_path {asset_path} and version_id {version_id} not found")
        if metadata.status != "archived":
            await self.db.log_access(username, asset_path, version_id, branch, "Destroy", False, f"Asset is {metadata.status}")
            raise HTTPException(status_code=400, detail=f"Asset {asset_path}/{version_id} is not archived")

        # Since lakefs is immutable, we can not delete old committed versions. Just let garbage collection do its job
        # If the version is the head, we can delete the primary and associated files and commit the deletion
        head_version = await self.db.get_head_version(asset_path, branch) 
        # Delete the primary and associated files if the version is the head
        if head_version == metadata.version_id:
            primary_path = f"{asset_path}/{metadata.primary_filename}"
            try:
                await self.object_store.delete_file(primary_path, version_id=metadata.version_id, branch=branch)
            except Exception as e:
                logger.error(f"Failed to delete primary file {primary_path}: {e}")

            delete_tasks = [
                delete_assoc(filename, version_id, branch)
                for filename, version_id in metadata.associated_filenames
                if filename
            ]
            await asyncio.gather(*delete_tasks)
        else:
            logger.info(f"Asset {asset_path}/{version_id} is not the head version, skipping deletion in lakefs")

        await self.db.delete_metadata(asset_path, version_id, branch)
        try:
            await self.vector_store.delete_metadata(asset_path, version_id, branch)
        except Exception as e:
            logger.error(f"Vector store delete failed for asset {asset_path}/{version_id}: {e}")

        await self.db.log_access(username, asset_path, version_id, branch, "Destroy", True, "Destroyed asset successfully")
        logger.info(f"Destroyed asset {asset_path}/{version_id} by user {username}")
        metadata.status = "destroyed"
        return metadata

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
        async def delete_assoc(filename, version_id, branch=None, is_head=False):
            path = f"{asset_path}/{filename}"
            try:
                await self.object_store.delete_file_v2(path, version_id=version_id, branch=branch, is_head=is_head)
            except Exception as e:
                logger.error(f"Failed to delete associated file {path}: {e}")

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
        try:
            await self.object_store.delete_file_v2(primary_path, version_id=metadata.version_id, branch=branch, is_head=is_head)
        except Exception as e:
            logger.error(f"Failed to delete primary file {primary_path}: {e}")

        delete_tasks = [
            delete_assoc(filename, version_id, branch, is_head=is_head)
            for filename, version_id in metadata.associated_filenames
            if filename
        ]
        await asyncio.gather(*delete_tasks)

        await self.db.delete_metadata(asset_path, version_id, branch)
        try:
            await self.vector_store.delete_metadata(asset_path, version_id, branch)
        except Exception as e:
            logger.error(f"Vector store delete failed for asset {asset_path}/{version_id}: {e}")

        await self.db.log_access(username, asset_path, version_id, branch, "Destroy", True, "Destroyed asset successfully")
        logger.info(f"Destroyed asset {asset_path}/{version_id} by user {username}")
        metadata.status = "destroyed"
        return metadata

    async def auto_destroy(self, current_date: Optional[datetime] = None) -> List[AssetMetadata]:
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
                return await self.destroy("scheduler", asset.branch, asset.asset_path, asset.version_id)
            except Exception as e:
                logger.info(f"Auto-destroy failed for {asset.asset_path}/{asset.version_id}: {e}")
                return None
            
        tasks = [safe_destroy(asset) for asset in assets]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]

    async def list_file_versions(self, username: str, key: str, branch: str) -> List[Dict]:
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
        keyword: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None, 
        page: int = 1, 
        page_size: int = 50
    ) -> List[AssetMetadataResponse]:
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
            page_size=page_size
        )
        await self.db.log_access(username, "", "", branch, "Get User Commits", True, f"Requested page {page} with size {page_size}")
        
        return commits

    async def update_expiration(
        self,
        username: str,
        branch: str,
        asset_path: str,
        version_id: str,
        archive_date_or_ttl: Union[int, str, None],
        destroy_date_or_ttl: Union[int, str, None],
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