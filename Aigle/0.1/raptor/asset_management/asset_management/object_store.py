import lakefs
import lakefs_sdk
import aioboto3
import botocore.exceptions
from typing import Union, BinaryIO, Dict, Optional
import io
import os
from fastapi import HTTPException
import base64
from urllib.parse import urlencode
import logging
import requests
from urllib.parse import urlparse, urlunparse

from .config import settings


logger = logging.getLogger(__name__)


class ObjectStore:
    def __init__(self):
        self.client = lakefs.Client(
            host=settings.lakefs_endpoint,
            username=settings.lakefs_access_key,
            password=settings.lakefs_secret_key
        )
        self.repository = settings.lakefs_repository
        self.branch = settings.lakefs_branch
        self._initialized = False
        self.session = aioboto3.Session()

    async def _ensure_bucket(self, bucket: str = settings.s3_bucket):
        """
        Ensure the specified bucket exists in the S3 storage.

        This method checks if the bucket exists by listing all buckets.
        If the bucket does not exist, it creates the bucket. 

        Raises:
            HTTPException: If there is a failure in ensuring the bucket exists, such as network 
            issues or insufficient permissions.
        """
        s3_client = await self.session.client(
            "s3",
            endpoint_url=settings.s3_endpoint,
            aws_access_key_id=settings.aws_access_key,
            aws_secret_access_key=settings.aws_secret_key
        ).__aenter__()

        try:
            # Check if bucket exists by listing buckets
            await s3_client.head_bucket(Bucket=bucket)
            logger.debug(f"Bucket already exists: {bucket}")

        except botocore.exceptions.ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code in ("404", "NoSuchBucket"):
                try:
                    await s3_client.create_bucket(Bucket=bucket)
                    logger.debug(f"Created bucket: {bucket}")
                except botocore.exceptions.ClientError as ce:
                    ce_code = ce.response["Error"]["Code"]
                    if ce_code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
                        logger.debug(f"Bucket already exists (caught during create): {bucket}")
                    else:
                        raise HTTPException(status_code=500, detail=f"S3 create error: {str(ce)}")

            elif error_code in ("403", "301"):
                # 403: The bucket already exists and you don't have permission to access it.
                # 301: The bucket already exists in another region.
                raise HTTPException(
                    status_code=500,
                    detail=f"Bucket '{bucket}' already exists and is not accessible."
                )
            else:
                raise HTTPException(status_code=500, detail=f"S3 error: {str(e)}")
        finally:
            await s3_client.__aexit__(None, None, None)
        
    def list_repositories(self):
        """List all repositories"""
        try:
            return [
                {
                    "id": repo.properties.id, 
                    "creation_date": repo.properties.creation_date, 
                    "default_branch": repo.properties.default_branch, 
                    "storage_namespace": repo.properties.storage_namespace
                }
                for repo in lakefs.repositories(client=self.client)
            ]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to list repositories: {str(e)}")

    def list_branches(self, repository_id: str):
        """List all repositories"""
        try:
            return [branch.id for branch in lakefs.repository(repository_id=repository_id, client=self.client).branches()]

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to list repositories: {str(e)}")

    def set_gc_rules(self, repository: str):
        try:
            garbage_collection_rules = lakefs_sdk.GarbageCollectionRules(
                default_retention_days=settings.lakefs_default_retention_days,
                branches=[
                    lakefs_sdk.GarbageCollectionRule(
                        branch_id="main",
                        retention_days=settings.lakefs_main_branch_retention_days
                    )
                ]
            )
            self.client.sdk_client.repositories_api.set_gc_rules(
                repository=repository, 
                garbage_collection_rules=garbage_collection_rules
            )
            logger.debug(f"GC rules set for repository '{repository}'")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to set GC rules: {str(e)}")

    async def create_repository(self, repository_id: str, branch: str, s3_bucket: str):
        """Create lakeFS repository if it does not exist"""
        try:
            await self._ensure_bucket(s3_bucket)
            lakefs.repository(repository_id=repository_id, client=self.client).create(
                # storage_namespace=f"s3://{s3_bucket}/{repository_id}",
                storage_namespace=f"s3://{s3_bucket}",
                default_branch=branch,
                exist_ok=True
            )
            self.set_gc_rules(repository_id)
            logger.debug(f"Repository '{repository_id}' created successfully")

        except Exception as e:
            if hasattr(e, "status_code") and e.status_code == 400:
                msg = e.body.get("message", "") if isinstance(e.body, dict) else str(e.body)
                if "already exists" in msg:
                    logger.debug(f"Repository '{repository_id}' already exists: {msg}")
                    return
                elif "failed to access storage" in msg:
                    raise HTTPException(
                        status_code=500, 
                        detail=f"""Storage access error: {msg} 
                        💡 Tip: If your S3 storage is running in a Docker container, clearing the data volume and restarting the container may solve this issue.
                        If your volumes are mounted on NFS and you have run `docker-compose down -v`, make sure that any existing data on the NFS mount is **completely deleted** before restarting the container. 
                        Failing to do so may still cause bucket conflicts or storage access errors in lakeFS.
                        """
                    )
            elif isinstance(e, botocore.exceptions.EndpointConnectionError):
                logger.error(f"Cannot connect to S3 endpoint: {e}")
                raise HTTPException(status_code=503, detail="S3 service temporarily unavailable")
            else:
                raise HTTPException(status_code=500, detail=f"{str(e)}")

    def create_branch(self, repository_id: str, branch: str, source_branch: str = "main"):
        """Create a branch"""
        try:           
            lakefs.repository(repository_id=repository_id, client=self.client).branch(branch).create(
                source_reference=source_branch,
                exist_ok=True
            )
            logger.debug(f"Branch '{branch}' created successfully or already exists")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create branch: {str(e)}")

    def get_branch(self, repository_id: str, branch: str):
        try:
            return lakefs.repository(repository_id=repository_id, client=self.client).branch(branch)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get branch: {str(e)}")

    async def initialize(self):
        """
        Initialize the object store.

        This method sets up the object store by creating a LakeFS repository and branch
        if they don't exist.

        Returns:
            None

        Raises:
            HTTPException: If the repository cannot be created or accessed.
        """
        if self._initialized:
            return
        try:
            # Ensure repository exists
            await self.create_repository(self.repository, self.branch, settings.s3_bucket)
            self._initialized = True
            logger.info(f"ObjectStore initialized successfully for repository: {self.repository}, branch: {self.branch}")
        except Exception as e:
            logger.error(f"Failed to initialize ObjectStore: {e}")
            raise

    async def close(self):
        """
        Close the object store.

        This method marks the object store as uninitialized.
        """
        if self._initialized:
            self._initialized = False
            logger.info("ObjectStore closed.")

    async def upload_file(self, data: Union[str, BinaryIO, bytes], key: str, content_type: str, metadata: Dict, branch: Optional[str] = None) -> Dict:
        """
        Upload a file to the LakeFS repository and return the version ID.

        Args:
            data: The file to upload. It can be a string (file path), a BinaryIO (file-like object)
                or bytes (file contents).
            key: The key to use for the uploaded file.
            content_type: The content type of the file.
            metadata: A dictionary of metadata to associate with the file.
            branch: The branch to upload the file to. If not provided, the default branch will be used.

        Returns:
            A dictionary containing the key and version ID of the uploaded file.

        Raises:
            HTTPException: If the file cannot be uploaded.
        """
        try:
            if branch is None:
                branch = self.branch
            else:
                self.create_branch(self.repository, branch)

            branch_ref = lakefs.repository(repository_id=self.repository, client=self.client).branch(branch)

            # Handle data input
            if isinstance(data, str):
                if not os.path.exists(data):
                    raise HTTPException(status_code=400, detail=f"File not found: {data}")
                with open(data, "rb") as f:
                    branch_ref.object(path=key).upload(
                        data=f.read(),
                        content_type=content_type,
                        pre_sign=True
                    )
            else:
                if isinstance(data, bytes):
                    data = io.BytesIO(data)
                data.seek(0)
                branch_ref.object(path=key).upload(
                    data=data.read(),
                    content_type=content_type,
                    pre_sign=True
                )
                data.seek(0)

            # Commit the upload
            commit = branch_ref.commit(
                message=f"Upload file {key}",
                metadata=metadata
            )
            commit_id = commit.get_commit().id

            logger.info(f"Uploaded file {key} with version ID {commit_id} to repository {self.repository}")
            return {"key": key, "version_id": commit_id}
        
        except Exception as e:
            if "commit: no changes" in str(e).lower():
                raise HTTPException(status_code=400, detail=f"File already exists: {key}")
            raise HTTPException(status_code=500, detail=f"Failed to upload file {key}: {str(e)}")

    async def get_file(self, key: str, version_id: Optional[str] = None, return_file_content: bool = False) -> Dict:
        """
        Retrieve a file or a specific version from the LakeFS repository, including its URL.

        Args:
            key (str): The key of the object in the repository.
            version_id (Optional[str]): The version ID of the file to retrieve. If not specified,
                the latest version will be retrieved.

        Returns:
            Dict: A dictionary containing the content of the file, its content type, its version ID,
                and its public URL.

        Raises:
            HTTPException: If the file cannot be retrieved.
        """
        try:
            # Generate pre-signed URL
            download_url = await self.generate_presigned_url(key, version_id)

            commit_ref = lakefs.repository(repository_id=self.repository, client=self.client).ref(version_id)
            obj = commit_ref.object(path=key)

            if return_file_content:
                content = io.BytesIO()
                with obj.reader(mode="rb", pre_sign=True) as reader:
                    content.write(reader.read())
                content.seek(0)

                return {
                    "content": base64.b64encode(content.read()).decode("utf-8"),
                    "content_type": obj.stat().content_type,
                    "version_id": version_id,
                    "url": download_url
                }

            return {
                "content_type": obj.stat().content_type,
                "version_id": version_id,
                "url": download_url
            }
            
        except Exception as e:
            if "not found" in str(e).lower():
                raise HTTPException(status_code=404, detail=f"File {key} or commit {version_id} not found")
            raise HTTPException(status_code=500, detail=f"Failed to retrieve file {key}: {str(e)}")

    async def generate_presigned_url(self, key: str, version_id: Optional[str] = None) -> str:
        """
        Generate a presigned URL for accessing a file or a specific version from the repository.

        Args:
            key (str): The key of the object in the repository.
            version_id (Optional[str]): The version ID of the file to generate the URL for.

        Returns:
            str: A presigned URL for accessing the specified file or version.

        Raises:
            HTTPException: If the repository does not exist or if there is a failure in generating
                the presigned URL.
        """
        try:
            endpoint = f"{settings.lakefs_endpoint}/api/v1/repositories/{self.repository}/refs/{version_id}/objects"
            query_params = {"path": key, "presign": "true"}
            url = f"{endpoint}?{urlencode(query_params)}"
            auth = (settings.lakefs_access_key, settings.lakefs_secret_key)
            response = requests.get(url, auth=auth)
            response.raise_for_status()
            parsed_internal = urlparse(response.url)
            parsed_public = parsed_internal._replace(
                scheme=urlparse(settings.s3_public_url).scheme,
                netloc=urlparse(settings.s3_public_url).netloc
            )
            external_url = urlunparse(parsed_public)
            return external_url

        except Exception as e:
            if "not found" in str(e).lower():
                raise HTTPException(status_code=404, detail=f"File {key} or commit {version_id} not found")
            raise HTTPException(status_code=500, detail=f"Failed to generate presigned URL for {key}: {str(e)}")

    async def delete_file(self, key: str, version_id: Optional[str] = None, branch: Optional[str] = None):
        """
        Delete a file or a specific version from the repository.

        Args:
            key (str): The key of the object in the repository.
            version_id (Optional[str]): The version ID of the file to delete. If not specified,
                the latest version will be deleted.
            branch (Optional[str]): The branch to delete the file from. If not specified,
                the default branch will be used.

        Raises:
            HTTPException: If the repository does not exist or if there is a failure in deleting the file.
        """
        try:
            if branch is None:
                branch = self.branch
            else:
                self.create_branch(self.repository, branch)

            branch_ref = lakefs.repository(repository_id=self.repository, client=self.client).branch(branch)
            
            # LakeFS doesn't support deleting specific versions directly, so we delete the file in a new commit
            branch_ref.object(path=key).delete()
            branch_ref.commit(
                message=f"Delete file {key}",
            )
            logger.info(f"Deleted file {key} from repository {self.repository}")
        except Exception as e:
            if "not found" in str(e).lower():
                pass  # Ignore if file doesn't exist
            else:
                raise HTTPException(status_code=500, detail=f"Failed to delete file {key}: {str(e)}")


    async def delete_associated_files(self, prefix: str, primary_file: str, branch: Optional[str] = None):
        """
        Delete all old associated files.

        Args:
            key (str): The key of the object in the repository.
            version_id (Optional[str]): The version ID of the file to delete. If not specified,
                the latest version will be deleted.
            branch (Optional[str]): The branch to delete the file from. If not specified,
                the default branch will be used.

        Raises:
            HTTPException: If the repository does not exist or if there is a failure in deleting the file.
        """
        try:
            if branch is None:
                branch = self.branch
            else:
                self.create_branch(self.repository, branch)

            branch_ref = lakefs.repository(repository_id=self.repository, client=self.client).branch(branch)
            objs = branch_ref.objects(prefix=prefix)
            associated_files = [o.path for o in objs if primary_file not in o.path]
            logger.info(f"Found associated files: {associated_files}")
            if not associated_files:
                return
            branch_ref.delete_objects(associated_files)
            branch_ref.commit(
                message=f"Delete associated files in {prefix}",
            )
            logger.info(f"Deleted associated files in {prefix} from repository {self.repository}")
        except Exception as e:
            if "not found" in str(e).lower():
                pass  # Ignore if file doesn't exist
            else:
                raise HTTPException(status_code=500, detail=f"Failed to delete associated files: {str(e)}")
