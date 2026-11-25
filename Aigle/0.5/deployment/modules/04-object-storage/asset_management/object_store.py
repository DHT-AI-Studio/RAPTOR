import asyncio
import base64
import io
import logging
import os
import uuid
from typing import BinaryIO
from urllib.parse import quote, urlencode, urlparse, urlunparse

import aioboto3
import botocore.exceptions
import lakefs
import redis.asyncio as aioredis
import requests
from botocore.auth import S3SigV4QueryAuth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials
from fastapi import HTTPException

# import lakefs_sdk
from lakefs.exceptions import ConflictException
from redis.exceptions import RedisError

from .config import settings
from .utils import mime_from_filename

logger = logging.getLogger(__name__)

# GD-2 export links: SigV4, path-style (bucket in the path, as SeaweedFS serves
# it). SeaweedFS ignores the region, but SigV4 needs one to sign with.
_SIGNING_REGION = "us-east-1"


def _presign_path_style_get(base_url: str, bucket: str, key: str, params: dict[str, str], expires_in: int) -> str:
    """Sign a path-style S3 GET URL directly against base_url with SigV4
    query auth, without ever constructing a boto3/aioboto3 client against
    it. `boto3.client(..., endpoint_url=base_url)` validates base_url's
    hostname (botocore.endpoint.EndpointCreator) and rejects one with an
    underscore in it -- a real constraint when base_url is settings.s3_public_url,
    which isn't ours to rename. That validation only guards actually opening
    a connection; signing a request is a separate, lower-level botocore
    primitive that never goes through it."""
    request = AWSRequest(
        method="GET",
        url=f"{base_url.rstrip('/')}/{quote(bucket, safe='')}/{quote(key, safe='/')}",
        params=params,
    )
    S3SigV4QueryAuth(
        Credentials(settings.aws_access_key, settings.aws_secret_key), "s3", _SIGNING_REGION, expires=expires_in,
    ).add_auth(request)
    return request.prepare().url


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
        # Persistent session for LakeFS presigned URL requests — reuses TCP connections.
        self._lakefs_session = requests.Session()
        self._lakefs_session.auth = (settings.lakefs_access_key, settings.lakefs_secret_key)
        # Async Redis cache for presigned URLs (TTL = 15 min, LakeFS expiry = 20 min).
        # Use redis.asyncio to avoid blocking the event loop during concurrent URL resolution.
        # Set PRESIGN_URL_CACHE_ENABLED=false in .env to disable caching (forces fresh URL on every request).
        if not settings.presign_url_cache_enabled:
            logger.info("Presigned URL cache disabled via PRESIGN_URL_CACHE_ENABLED=false")
            self._redis = None
        else:
            try:
                # requirements.txt floats redis>=5.0.0; the installed 8.x defaults
                # ConnectionPool.max_connections to 100 (was effectively unbounded
                # on 5.x), raising "Too many connections" under concurrent presign
                # lookups -- generate_presigned_url() treats that as a cache miss.
                self._redis = aioredis.Redis(
                    host=settings.redis_host,
                    port=settings.redis_port,
                    db=settings.redis_db,
                    password=settings.redis_password or None,
                    socket_connect_timeout=1,
                    decode_responses=True,
                )
                logger.info("Redis cache configured for presigned URLs")
            except Exception as _e:  # noqa: BLE001 -- Redis() ctor failures aren't typed, fail-open to no cache
                logger.warning(f"Redis unavailable, presigned URL caching disabled: {_e}")
                self._redis = None

    async def _ensure_bucket(self, bucket: str | None = None):
        """
        Ensure the specified bucket exists in the S3 storage.

        This method checks if the bucket exists by listing all buckets.
        If the bucket does not exist, it creates the bucket.

        Raises:
            HTTPException: If there is a failure in ensuring the bucket exists, such as network
            issues or insufficient permissions.
        """
        bucket = bucket or settings.s3_bucket
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
                        raise HTTPException(status_code=500, detail=f"S3 create error: {ce!s}")

            elif error_code in ("403", "301"):
                # 403: The bucket already exists and you don't have permission to access it.
                # 301: The bucket already exists in another region.
                raise HTTPException(
                    status_code=500,
                    detail=f"Bucket '{bucket}' already exists and is not accessible."
                )
            else:
                raise HTTPException(status_code=500, detail=f"S3 error: {e!s}")
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
        except Exception as e:  # noqa: BLE001 -- lakefs errors aren't all ApiException-wrapped; converts any failure to 500
            raise HTTPException(status_code=500, detail=f"Failed to list repositories: {e!s}")

    def list_branches(self, repository_id: str):
        """List all repositories"""
        try:
            return [branch.id for branch in lakefs.repository(repository_id=repository_id, client=self.client).branches()]

        except Exception as e:  # noqa: BLE001 -- lakefs errors aren't all ApiException-wrapped; converts any failure to 500
            raise HTTPException(status_code=500, detail=f"Failed to list repositories: {e!s}")

    # def set_gc_rules(self, repository: str):
    #     try:
    #         garbage_collection_rules = lakefs_sdk.GarbageCollectionRules(
    #             default_retention_days=settings.lakefs_default_retention_days,
    #             branches=[
    #                 lakefs_sdk.GarbageCollectionRule(
    #                     branch_id="main",
    #                     retention_days=settings.lakefs_main_branch_retention_days
    #                 )
    #             ]
    #         )
    #         self.client.sdk_client.repositories_api.set_gc_rules(
    #             repository=repository,
    #             garbage_collection_rules=garbage_collection_rules
    #         )
    #         logger.debug(f"GC rules set for repository '{repository}'")
    #     except Exception as e:
    #         raise HTTPException(status_code=500, detail=f"Failed to set GC rules: {str(e)}")

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

            # # Set GC rules (deprecated)
            # # Since the effective period of different assets is different,
            # # and there is no fixed effective period, so currently it is directly deleted from s3 storage
            # self.set_gc_rules(repository_id)

            logger.debug(f"Repository '{repository_id}' created successfully")

        except Exception as e:  # noqa: BLE001 -- spans lakeFS (status_code/body) and botocore errors, type-branched below
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
                else:
                    raise HTTPException(status_code=400, detail=f"Failed to create repository: {msg}")
            elif isinstance(e, botocore.exceptions.EndpointConnectionError):
                logger.error(f"Cannot connect to S3 endpoint: {e}")
                raise HTTPException(status_code=503, detail="S3 service temporarily unavailable")
            else:
                raise HTTPException(status_code=500, detail=f"{e!s}")

    def create_branch(self, repository_id: str, branch: str, source_branch: str = "main"):
        """Create a branch"""
        try:
            lakefs.repository(repository_id=repository_id, client=self.client).branch(branch).create(
                source_reference=source_branch,
                exist_ok=True
            )
            logger.debug(f"Branch '{branch}' created successfully or already exists")
        except Exception as e:  # noqa: BLE001 -- lakefs errors aren't all ApiException-wrapped; converts any failure to 500
            raise HTTPException(status_code=500, detail=f"Failed to create branch: {e!s}")

    def get_branch(self, repository_id: str, branch: str):
        try:
            return lakefs.repository(repository_id=repository_id, client=self.client).branch(branch)
        except Exception as e:  # noqa: BLE001 -- lakefs errors aren't all ApiException-wrapped; converts any failure to 500
            raise HTTPException(status_code=500, detail=f"Failed to get branch: {e!s}")

    def delete_branch(self, repository_id: str, branch: str):
        try:
            return lakefs.repository(repository_id=repository_id, client=self.client).branch(branch).delete()
        except Exception as e:  # noqa: BLE001 -- lakefs errors aren't all ApiException-wrapped; converts any failure to 500
            raise HTTPException(status_code=500, detail=f"Failed to delete branch: {e!s}")

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

    @staticmethod
    def _as_bytes(data: str | BinaryIO | bytes) -> bytes:
        if isinstance(data, str):
            if not os.path.exists(data):
                raise HTTPException(status_code=400, detail=f"File not found: {data}")
            with open(data, "rb") as f:
                return f.read()
        if isinstance(data, bytes):
            return data
        data.seek(0)
        return data.read()

    def _sync_transact_write(
        self,
        branch: str,
        commit_message: str,
        uploads: list[tuple] = (),
        deletes: list[str] = (),
    ):
        """N uploads and/or deletes as one atomic commit, via an
        ephemeral branch merged back into `branch` -- manual replication of
        Branch.transact()'s steps (not the convenience wrapper, which
        discards merge_into()'s return value and we need that as our
        version_id). The ephemeral branch isolates this call's staging from
        any other concurrent writer on `branch`, so our commit can never
        sweep up someone else's uncommitted objects (and vice versa) --
        unlike a direct branch_ref.commit(), which commits everything
        currently staged on the branch, not just this call's own writes.
        Must run in a thread pool via asyncio.to_thread.

        uploads: list of (data, key, content_type). deletes: list of key.
        Returns (merge_commit_id, {key: checksum}).
        """
        repo = lakefs.repository(repository_id=self.repository, client=self.client)
        last_error = None
        for attempt in range(settings.lakefs_tx_max_retries):
            tx_name = f"tx-{uuid.uuid4()}"
            tx_branch = repo.branch(tx_name).create(branch, hidden=True)
            try:
                checksums = {}
                for data, key, content_type in uploads:
                    obj = tx_branch.object(path=key)
                    obj.upload(data=self._as_bytes(data), content_type=content_type, pre_sign=False)
                    checksums[key] = obj.stat().checksum
                if deletes:
                    tx_branch.delete_objects(list(deletes))
                tx_branch.commit(message=commit_message)
                merge_id = tx_branch.merge_into(branch, message=f"Merge {tx_name} into {branch}")
                return merge_id, checksums
            except ConflictException as e:
                # Narrow case: another transaction's merge landed on an
                # overlapping path between our ephemeral branch's creation
                # and our own merge attempt. Retry on a fresh ephemeral
                # branch -- lakeFS guarantees nothing we already committed
                # on the old ephemeral branch is lost, we just need a new
                # merge attempt.
                last_error = e
                continue
            finally:
                tx_branch.delete()
        raise last_error

    async def upload_file(self, data: str | BinaryIO | bytes, key: str, content_type: str, branch: str | None = None) -> dict:
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
                await asyncio.to_thread(self.create_branch, self.repository, branch)

            commit_id, checksums = await asyncio.to_thread(
                self._sync_transact_write, branch, f"Upload file {key}", [(data, key, content_type)]
            )

            logger.info(f"Uploaded file {key} with version ID {commit_id} to repository {self.repository}")
            return {"key": key, "version_id": commit_id, "checksum": checksums[key]}

        except Exception as e:  # noqa: BLE001 -- lakefs SDK errors from the thread aren't all ApiException-wrapped; string-matched below
            if "commit: no changes" in str(e).lower():
                raise HTTPException(status_code=409, detail=f"LakeFS/DB inconsistency: file '{key}' is already committed in LakeFS but has no DB record. Delete the file from LakeFS and re-upload.")
            raise HTTPException(status_code=500, detail=f"Failed to upload file {key}: {e!s}")

    def _group_objects_by_size(self, objects: list[dict]) -> list[list[dict]]:
        """Split objects into groups, each within lakefs_batch_group_max_bytes
        / lakefs_batch_group_max_files, so a single LakeFS transaction never
        grows unboundedly -- not a reject limit, callers can send a batch of
        any size. Each object's content is read in full before the size
        check runs, so one object is always placed entirely into one group,
        never split across two."""
        groups: list[list[dict]] = []
        current: list[dict] = []
        current_bytes = 0
        for o in objects:
            data = o["content"] if isinstance(o["content"], bytes) else self._as_bytes(o["content"])
            if current and (
                current_bytes + len(data) > settings.lakefs_batch_group_max_bytes
                or len(current) >= settings.lakefs_batch_group_max_files
            ):
                groups.append(current)
                current, current_bytes = [], 0
            current.append({**o, "content": data})
            current_bytes += len(data)
        if current:
            groups.append(current)
        return groups

    async def batch_upload(self, objects: list[dict], branch: str | None = None, commit_message: str | None = None) -> dict:
        """
        Upload multiple objects, split internally into one or more LakeFS
        transactions (see _sync_transact_write and _group_objects_by_size)
        so no single commit grows unboundedly. Callers never need to
        pre-split a batch themselves -- there's no size or count limit here.

        Args:
            objects: list of {"key": str, "content": bytes|str|BinaryIO, "content_type": str}.
            branch: The branch to upload to. If not provided, the default branch will be used.
            commit_message: Optional override for the commit message (a
                "(part N/M)" suffix is appended automatically when more than
                one group is needed).

        Returns:
            {"files": [{"key", "checksum", "version_id"}, ...]} -- version_id
            is per file, not per batch: files that landed in different groups
            get different commit ids. A file whose group turned out to be
            byte-identical to what's already on the branch (LakeFS "commit:
            no changes") is simply omitted here, not returned with a null
            version_id -- callers fall back to whatever version is already
            on record for it.

        Raises:
            HTTPException: If the batch is empty or the upload genuinely fails.
        """
        if not objects:
            raise HTTPException(status_code=400, detail="batch_upload() called with no objects")

        try:
            if branch is None:
                branch = self.branch
            else:
                await asyncio.to_thread(self.create_branch, self.repository, branch)

            groups = await asyncio.to_thread(self._group_objects_by_size, objects)

            base_message = commit_message or f"Batch upload {len(objects)} files"
            checksums: dict[str, str] = {}
            version_ids: dict[str, str] = {}
            for i, group in enumerate(groups):
                message = base_message if len(groups) == 1 else f"{base_message} (part {i + 1}/{len(groups)})"
                uploads = [(o["content"], o["key"], o.get("content_type")) for o in group]
                try:
                    commit_id, group_checksums = await asyncio.to_thread(
                        self._sync_transact_write, branch, message, uploads
                    )
                except Exception as e:
                    if "commit: no changes" in str(e).lower():
                        continue  # this group is byte-identical to what's already on the branch
                    raise
                checksums.update(group_checksums)
                for o in group:
                    version_ids[o["key"]] = commit_id

            logger.info(f"Batch-uploaded {len(objects)} files across {len(groups)} commit(s) to repository {self.repository}")
            return {
                "files": [
                    {"key": o["key"], "checksum": checksums[o["key"]], "version_id": version_ids[o["key"]]}
                    for o in objects if o["key"] in version_ids
                ],
            }

        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001 -- lakefs SDK errors from the thread aren't all ApiException-wrapped; converts any failure to 500
            raise HTTPException(status_code=500, detail=f"Failed to batch-upload {len(objects)} files: {e!s}")

    async def get_file(self, key: str, version_id: str | None = None, return_file_content: bool = False) -> dict:
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
            download_url = await self.generate_presigned_url(key, version_id)

            if not return_file_content:
                # Skip obj.stat() -- an extra LakeFS round-trip that doubles
                # latency -- and guess content_type from the filename
                # extension instead (free, no I/O); None only when the
                # extension isn't in EXTENSION_TO_MIME.
                return {
                    "content_type": mime_from_filename(os.path.basename(key)),
                    "version_id": version_id,
                    "url": download_url
                }

            commit_ref = lakefs.repository(repository_id=self.repository, client=self.client).ref(version_id)
            obj = commit_ref.object(path=key)

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

        except Exception as e:  # noqa: BLE001 -- lakefs read + presign errors aren't all ApiException-wrapped; string-matched below
            if "not found" in str(e).lower():
                raise HTTPException(status_code=404, detail=f"File {key} or commit {version_id} not found")
            raise HTTPException(status_code=500, detail=f"Failed to retrieve file {key}: {e!s}")

    async def batch_resolve_physical_addresses(
        self, asset_path: str, items: list[tuple[str, str]], concurrency: int = 10
    ) -> dict[tuple[str, str], str]:
        """Resolve many (key, version_id) pairs under one asset_path to their physical_address
        in as few LakeFS calls as possible -- same one-objects/ls-call-per-commit batching as
        batch_generate_presigned_urls(), reused here by bulk_delete_files_v2()'s destroy-lifecycle
        hard delete instead of one blocking stat() per file. Deliberately omits `presign` --
        unlike batch_generate_presigned_urls(), which wants a download link and so asks LakeFS
        to overwrite `physical_address` with a presigned URL, this needs the raw `s3://bucket/key`
        value to hard-delete the real object; passing presign=true here would silently turn every
        delete into a no-op (the "address" handed to S3 wouldn't match any real key, and S3's
        DeleteObjects doesn't error on a key that doesn't exist).

        One LakeFS call per distinct commit, bounded to `concurrency` at a time (same pattern as
        gdpr_list_files()/scan_assets()) instead of one at a time -- a destroy with files spread
        across many small commits (e.g. associated files attached in several separate batches)
        would otherwise pay each commit's network round trip serially.

        Returns {(key, version_id): physical_address} (the s3://bucket/ prefix already
        stripped) for whatever was resolved; silently omits entries it couldn't resolve --
        callers fall back to a per-file stat() for anything missing.
        """
        if not items:
            return {}
        by_version: dict[str, set[str]] = {}
        for key, version_id in items:
            by_version.setdefault(version_id, set()).add(key)

        resolved: dict[tuple[str, str], str] = {}
        semaphore = asyncio.Semaphore(concurrency)

        async def _resolve_one_commit(version_id: str, wanted: set[str]) -> None:
            async with semaphore:
                try:
                    after = ""
                    while True:
                        endpoint = f"{settings.lakefs_endpoint}/api/v1/repositories/{self.repository}/refs/{version_id}/objects/ls"
                        query_params = {"prefix": f"{asset_path}/", "amount": 1000}
                        if after:
                            query_params["after"] = after
                        url = f"{endpoint}?{urlencode(query_params)}"
                        response = await asyncio.to_thread(
                            lambda u=url: self._lakefs_session.get(u, timeout=8)
                        )
                        response.raise_for_status()
                        body = response.json()
                        for obj in body.get("results", []):
                            path = obj.get("path")
                            if path not in wanted:
                                continue
                            resolved[(path, version_id)] = obj.get("physical_address", "").replace(
                                f"s3://{settings.s3_bucket}/", "")
                        pagination = body.get("pagination") or {}
                        if not pagination.get("has_more"):
                            break
                        after = pagination.get("next_offset", "")
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Batch physical-address list failed for version {version_id}, falling back per-file: {e}")

        await asyncio.gather(*(_resolve_one_commit(version_id, wanted) for version_id, wanted in by_version.items()))
        return resolved

    def _to_public_url(self, internal_url: str) -> str:
        """Rewrite a LakeFS-internal presigned URL's scheme+host to the public S3 endpoint."""
        parsed_internal = urlparse(internal_url)
        parsed_public = parsed_internal._replace(
            scheme=urlparse(settings.s3_public_url).scheme,
            netloc=urlparse(settings.s3_public_url).netloc
        )
        return urlunparse(parsed_public)

    async def batch_generate_presigned_urls(self, asset_path: str, items: list[tuple[str, str]]) -> dict[tuple[str, str], dict[str, str | None]]:
        """Resolve presigned URLs (and content_type) for many (filename,
        version_id) pairs under one asset_path in as few LakeFS calls as
        possible.

        Groups items by version_id and makes one `objects/ls?presign=true`
        call per distinct version_id instead of one `generate_presigned_url()`
        per file -- batch_upload() usually lands every associated file of one
        asset in a single LakeFS commit (see _group_objects_by_size), so in
        the common case this is a single LakeFS round trip for an entire
        asset's files, not O(file count). list_objects()'s own ObjectStats
        already carries content_type -- unlike the single-file presign path
        (generate_presigned_url(), a redirect response with no metadata),
        this gets the real value for free, no separate obj.stat() needed.

        Returns {(filename, version_id): {"url":..., "content_type":...}} for
        whatever was resolved; silently omits entries it couldn't resolve (a
        failed version_id group, or a file list_objects didn't return) --
        callers fall back to generate_presigned_url() per-file for anything
        missing here, so this is a best-effort accelerator, never the only
        path.
        """
        if not items:
            return {}
        by_version: dict[str, dict[str, str]] = {}
        for filename, version_id in items:
            by_version.setdefault(version_id, {})[f"{asset_path}/{filename}"] = filename

        resolved: dict[tuple[str, str], dict[str, str | None]] = {}
        for version_id, wanted in by_version.items():
            try:
                after = ""
                while True:
                    endpoint = f"{settings.lakefs_endpoint}/api/v1/repositories/{self.repository}/refs/{version_id}/objects/ls"
                    query_params = {"prefix": f"{asset_path}/", "presign": "true", "amount": 1000}
                    if after:
                        query_params["after"] = after
                    url = f"{endpoint}?{urlencode(query_params)}"
                    response = await asyncio.to_thread(
                        lambda u=url: self._lakefs_session.get(u, timeout=8)
                    )
                    response.raise_for_status()
                    body = response.json()
                    for obj in body.get("results", []):
                        path = obj.get("path")
                        filename = wanted.get(path)
                        if not filename:
                            continue
                        external_url = self._to_public_url(obj.get("physical_address", ""))
                        resolved[(filename, version_id)] = {"url": external_url, "content_type": obj.get("content_type")}
                        if self._redis:
                            try:
                                await self._redis.setex(f"presign:{version_id}:{path}", settings.presign_url_ttl, external_url)
                            except RedisError as e:
                                logger.debug(f"Presign cache write failed for {path}: {e}")
                    pagination = body.get("pagination") or {}
                    if not pagination.get("has_more"):
                        break
                    after = pagination.get("next_offset", "")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Batch presign list failed for version {version_id}, falling back per-file: {e}")
                continue
        return resolved

    async def batch_generate_presigned_urls_by_key(
        self, items: list[tuple[str, str]]
    ) -> dict[tuple[str, str], dict[str, str | None]]:
        """Like batch_generate_presigned_urls(), but for callers with full LakeFS keys from several
        different assets in one batch (e.g. Module 13's face-image URLs on a people-search response
        spanning many videos, plus identity seed photos under their own identity/ prefix) -- no single
        shared asset_path to pass in.

        Groups by (key's own parent directory, version_id) instead of a caller-supplied asset_path:
        items under the same directory in the same commit still resolve with one objects/ls call (the
        common case -- one video's representative-image batch lands in one commit under its own
        asset_path, see video_face_detection_service/personal_db_client.py), while a seed photo under
        identity/ or anything else elsewhere just becomes its own, still-one-call group.

        Returns {(key, version_id): {"url":..., "content_type":...}} for whatever was resolved;
        silently omits entries it couldn't resolve, same best-effort contract as
        batch_generate_presigned_urls() -- callers fall back to generate_presigned_url() per-key.
        """
        if not items:
            return {}
        groups: dict[tuple[str, str], set[str]] = {}
        for key, version_id in items:
            prefix = key.rsplit("/", 1)[0] if "/" in key else ""
            groups.setdefault((prefix, version_id), set()).add(key)

        resolved: dict[tuple[str, str], dict[str, str | None]] = {}
        for (prefix, version_id), wanted in groups.items():
            try:
                after = ""
                while True:
                    endpoint = f"{settings.lakefs_endpoint}/api/v1/repositories/{self.repository}/refs/{version_id}/objects/ls"
                    query_params = {"prefix": f"{prefix}/" if prefix else "", "presign": "true", "amount": 1000}
                    if after:
                        query_params["after"] = after
                    url = f"{endpoint}?{urlencode(query_params)}"
                    response = await asyncio.to_thread(
                        lambda u=url: self._lakefs_session.get(u, timeout=8)
                    )
                    response.raise_for_status()
                    body = response.json()
                    for obj in body.get("results", []):
                        path = obj.get("path")
                        if path not in wanted:
                            continue
                        external_url = self._to_public_url(obj.get("physical_address", ""))
                        resolved[(path, version_id)] = {"url": external_url, "content_type": obj.get("content_type")}
                        if self._redis:
                            try:
                                await self._redis.setex(f"presign:{version_id}:{path}", settings.presign_url_ttl, external_url)
                            except RedisError as e:
                                logger.debug(f"Presign cache write failed for {path}: {e}")
                    pagination = body.get("pagination") or {}
                    if not pagination.get("has_more"):
                        break
                    after = pagination.get("next_offset", "")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Batch presign list failed for {prefix!r}@{version_id}, falling back per-file: {e}")
                continue
        return resolved

    async def generate_presigned_url(self, key: str, version_id: str | None = None) -> str:
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
        cache_key = f"presign:{version_id}:{key}"
        try:
            if self._redis:
                try:
                    cached = await self._redis.get(cache_key)
                    if cached:
                        return cached
                except RedisError as e:
                    logger.warning(f"Presign cache read failed, using LakeFS directly: {e}")

            endpoint = f"{settings.lakefs_endpoint}/api/v1/repositories/{self.repository}/refs/{version_id}/objects"
            query_params = {"path": key, "presign": "true"}
            url = f"{endpoint}?{urlencode(query_params)}"
            response = await asyncio.to_thread(
                lambda: self._lakefs_session.get(url, timeout=8, allow_redirects=False)
            )
            if response.status_code in (301, 302, 303, 307, 308):
                presigned = response.headers.get("Location", "")
            else:
                response.raise_for_status()
                presigned = response.url
            external_url = self._to_public_url(presigned)

            if self._redis:
                try:
                    await self._redis.setex(cache_key, settings.presign_url_ttl, external_url)
                except RedisError as e:
                    logger.debug(f"Presign cache write failed for {cache_key}: {e}")

            return external_url

        except requests.exceptions.RequestException as e:
            if "not found" in str(e).lower():
                raise HTTPException(status_code=404, detail=f"File {key} or commit {version_id} not found")
            raise HTTPException(status_code=500, detail=f"Failed to generate presigned URL for {key}: {e!s}")

    async def gdpr_file_link(self, key: str, version_id: str, expires_in: int) -> dict:
        """GD-2 / VIE01-247: size, content type and a download link for one
        file version, for Module 33's export files/index.json.

        The link is presigned here, on the object's physical address, for
        the export's own lifetime (expires_in) -- not via LakeFS's
        ?presign=true, whose lifetime is lakefs_pre_signed_expiry (20m),
        global to every download on the platform. Same physical-address
        resolution bulk_delete_files_v2 uses. Content-Disposition carries the
        original filename, since the physical key is an opaque LakeFS id.

        Signed with SigV4 for the public endpoint directly: SigV4 signs the
        host, so the link can't be signed internally and rewritten by
        _to_public_url afterwards -- and SeaweedFS rejects SigV2 links that
        carry response-content-disposition (403). Signed via
        _presign_path_style_get(), not a real client, since settings.s3_public_url
        can be a hostname boto3's own client construction refuses (see its
        docstring)."""
        stat = await asyncio.to_thread(
            lambda: lakefs.repository(repository_id=self.repository, client=self.client)
            .ref(version_id).object(path=key).stat()
        )
        physical_key = stat.physical_address.replace(f"s3://{settings.s3_bucket}/", "")
        filename = quote(os.path.basename(key))
        url = _presign_path_style_get(
            settings.s3_public_url or settings.s3_endpoint, settings.s3_bucket, physical_key,
            {"response-content-disposition": f"attachment; filename*=UTF-8''{filename}"},
            expires_in,
        )
        return {
            "size": stat.size_bytes,
            "type": stat.content_type or mime_from_filename(key),
            "download_url": url,
        }

    async def delete_file(self, key: str, version_id: str | None = None, branch: str | None = None):
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
                await asyncio.to_thread(self.create_branch, self.repository, branch)

            # LakeFS doesn't support deleting specific versions directly, so we delete the file in a new commit.
            # Routed through _sync_transact_write for the same
            # concurrent-writer isolation as uploads (see its docstring).
            await asyncio.to_thread(
                self._sync_transact_write, branch, f"Delete file {key}", (), [key]
            )
            logger.info(f"Deleted file {key} from repository {self.repository}")
        except Exception as e:  # noqa: BLE001 -- lakefs SDK errors from the thread aren't all ApiException-wrapped; string-matched below
            if "not found" in str(e).lower() or "commit: no changes" in str(e).lower():
                pass  # Ignore if file doesn't exist / nothing to delete
            else:
                raise HTTPException(status_code=500, detail=f"Failed to delete file {key}: {e!s}")


    async def delete_associated_files(self, prefix: str, primary_file: str, branch: str | None = None):
        """
        Delete all old associated files. When a primary file is changed, all other old associated files with the same prefix
        (except the primary file) will be deleted.

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
                await asyncio.to_thread(self.create_branch, self.repository, branch)

            def _list_associated(prefix, primary_file):
                branch_ref = lakefs.repository(repository_id=self.repository, client=self.client).branch(branch)
                return [o.path for o in branch_ref.objects(prefix=prefix) if primary_file not in o.path]

            associated_files = await asyncio.to_thread(_list_associated, prefix, primary_file)
            logger.info(f"Found associated files: {associated_files}")
            if not associated_files:
                return

            # Routed through _sync_transact_write for the same
            # concurrent-writer isolation as uploads (see its docstring).
            await asyncio.to_thread(
                self._sync_transact_write, branch, f"Delete associated files in {prefix}", (), associated_files
            )
            logger.info(f"Deleted associated files in {prefix} from repository {self.repository}")
        except Exception as e:  # noqa: BLE001 -- lakefs SDK errors from the thread aren't all ApiException-wrapped; string-matched below
            if "not found" in str(e).lower() or "commit: no changes" in str(e).lower():
                pass  # Ignore if file doesn't exist / nothing to delete
            else:
                raise HTTPException(status_code=500, detail=f"Failed to delete associated files: {e!s}")

    # S3's bulk DeleteObjects caps at 1000 keys per request -- same limit
    # lakeFS's own GC sweep step chunks by (io.treeverse.clients.StorageUtils.S3MaxBulkSize).
    S3_MAX_BULK_DELETE_SIZE = 1000

    async def _bulk_delete_from_s3_bucket(
        self, keys: list[str], bucket: str | None = None, concurrency: int = 10
    ) -> dict[str, int]:
        """
        Delete many keys directly from the S3 bucket in chunked bulk
        DeleteObjects calls, instead of one delete_object round-trip per
        key -- the same batching lakeFS's own GC sweep step uses. Chunks
        are dispatched concurrently (bounded by `concurrency`, same pattern
        as batch_resolve_physical_addresses()/gdpr_list_files()) instead of
        one at a time: measured locally against SeaweedFS, firing 10 chunks
        of 100 keys concurrently instead of sequentially was ~3x faster --
        the backing store does process multiple DeleteObjects calls at
        once, cost does not simply serialize per key regardless of
        batching.

        Returns {"deleted": n, "errors": n}.
        """
        bucket = bucket or settings.s3_bucket
        s3_client = await self.session.client(
            "s3",
            endpoint_url=settings.s3_endpoint,
            aws_access_key_id=settings.aws_access_key,
            aws_secret_access_key=settings.aws_secret_key
        ).__aenter__()

        deleted = 0
        errors = 0
        semaphore = asyncio.Semaphore(concurrency)

        async def _delete_chunk(chunk: list[str]) -> tuple[int, int]:
            async with semaphore:
                response = await s3_client.delete_objects(
                    Bucket=bucket,
                    Delete={"Objects": [{"Key": k} for k in chunk], "Quiet": True},
                )
            chunk_errors = response.get("Errors", [])
            for e in chunk_errors:
                logger.warning(f"Bulk delete failed for {e.get('Key')}: {e.get('Message')}")
            return len(chunk) - len(chunk_errors), len(chunk_errors)

        try:
            chunks = [keys[i:i + self.S3_MAX_BULK_DELETE_SIZE] for i in range(0, len(keys), self.S3_MAX_BULK_DELETE_SIZE)]
            results = await asyncio.gather(*(_delete_chunk(chunk) for chunk in chunks))
            for chunk_deleted, chunk_errors in results:
                deleted += chunk_deleted
                errors += chunk_errors
        finally:
            await s3_client.__aexit__(None, None, None)

        logger.info(f"Bulk-deleted {deleted}/{len(keys)} objects from bucket {bucket} ({errors} errors)")
        return {"deleted": deleted, "errors": errors}

    async def bulk_delete_files_v2(
        self, files: list[tuple[str, str]], is_head: bool = False, branch: str | None = None,
        asset_path: str | None = None,
    ) -> dict[str, int]:
        """
        Destroy-lifecycle bulk delete for a primary file plus its
        associated files together, e.g. client.py's destroy_v2(). Resolves
        each (key, version_id) pair to its physical_address via
        batch_resolve_physical_addresses() when asset_path is given -- one
        objects/ls call per distinct commit, not one stat() per file -- then
        falls back to a per-file stat() (off the event loop, via
        asyncio.to_thread) for anything that didn't resolve, since there's
        no bulk-stat API. All of them are then hard-deleted from S3 in one
        chunked bulk DeleteObjects call instead of one delete_object
        round-trip per file (see _bulk_delete_from_s3_bucket). If is_head,
        also removes every key from the branch's live tree in a single
        LakeFS transaction instead of one delete_file() commit per file. A
        file whose physical_address can't be resolved is logged and
        skipped, not fatal to the rest of the batch.

        files: list of (key, version_id) pairs.
        Returns {"deleted": n, "errors": n} from the S3 bulk delete.
        """
        resolved = await self.batch_resolve_physical_addresses(asset_path, files) if asset_path else {}
        physical_addresses = list(resolved.values())
        for key, version_id in files:
            if (key, version_id) in resolved:
                continue
            try:
                commit_ref = lakefs.repository(repository_id=self.repository, client=self.client).ref(version_id)
                stat = await asyncio.to_thread(commit_ref.object(path=key).stat)
                physical_addresses.append(stat.physical_address.replace(f"s3://{settings.s3_bucket}/", ""))
            except Exception as e:  # noqa: BLE001 -- lakefs stat() errors aren't all ApiException-wrapped; skips this file, logged
                logger.error(f"Failed to resolve physical address for {key}: {e}")

        if is_head and files:
            try:
                await asyncio.to_thread(
                    self._sync_transact_write,
                    branch or self.branch,
                    f"Destroy {len(files)} files",
                    (),
                    [key for key, _ in files],
                )
            except Exception as e:  # noqa: BLE001 -- lakefs SDK errors from the thread aren't all ApiException-wrapped; string-matched below
                if "commit: no changes" not in str(e).lower():
                    logger.error(f"Failed to remove {len(files)} files from branch: {e}")

        return await self._bulk_delete_from_s3_bucket(physical_addresses)
