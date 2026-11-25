"""GD-2 / VIE01-247: the export state machine.

Sequence: create job -> gather each store's exportable content -> lay it out
as the ticket's ZIP structure -> split into parts at GD_EXPORT_MAX_ZIP_BYTES
-> upload each part to SeaweedFS (gdpr-exports/{job_id}/export.zip, or
export-part-N.zip) -> presign a GET URL per part (GD_EXPORT_TTL_HOURS) ->
write the job row with download_urls/size_bytes/expires_at.

    manifest.json                    schema version, GDPR/CCPA references, stores, parts
    identity/keycloak_profile.json   Module 06
    files/index.json                 Module 04 -- metadata + a download link per
                                     file (valid as long as the ZIP), never bytes
    content/{documents,entities,moments}.json   Module 25
    memory/long_term.json            Module 26
    memory/sessions/{session_id}.json           Module 26, one per session
    memory/multimedia.json           Module 26 (not in the ticket's tree; kept
                                     rather than dropped -- it's the user's data)
    benchmark/runs.json              benchmark runs (results) + schemas the
                                     user owns (test cases)
    moderation/flags.json            Module 23 -- this user's own moderation_flags
                                     rows (CM-5 / VIE01-408)
    voice/recordings.json            Module 34 -- metadata per stored session
                                     recording, never the audio (GD-10)

audit/violations.json (guardrail_violations -- a different Module 23 table,
with no owner column at all) has nothing behind it here (see
_OMITTED_SECTIONS); manifest.json says why instead of shipping an empty or
fabricated file.
voice/recordings.json is omitted the same way when GD_MODULE_34_URL is
empty, and manifest.json's speaker_embeddings_cached is Module 35's count.

A store that fails to answer doesn't abort the whole export -- its section
is simply omitted and recorded in manifest.json's stores_failed, since a
partial export beats no export. Unlike erasure, this never touches
gdpr:erased_users or disables/deletes anything -- it's read-only end to end.

V05-10 / VIE01-364: every finished job leaves a CF-4 `gdpr.export` event --
`allow` when completed, `error` on a partial failure or a failed upload.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
import uuid
import zipfile
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote

import aioboto3
import asyncpg
import botocore.exceptions
import httpx
from botocore.auth import S3SigV4QueryAuth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials

from app.core.config import settings
from app.models.job import ExportJob, JobStatus
from app.services.audit_events import audit_writer, privacy_event

logger = logging.getLogger("privacy_compliance.export")

_INTERNAL_HEADERS = {"X-Internal-Token": settings.internal_token}

EXPORT_SCHEMA_VERSION = "1.0"
GDPR_ARTICLE = "Art. 20 Right to Data Portability"
CCPA_SECTION = "§ 1798.100"

# Sections of the ticket's ZIP structure that this codebase has no real
# backing for -- omitted from every export rather than faked, same policy
# GD-1's erasure orchestrator follows for its own per-store deletions.
_OMITTED_SECTIONS = {
    "audit/violations.json": (
        "guardrail_violations (Module 23's guardrails database) has no "
        "user_id/branch_id column -- not scopable to this user"
    ),
}
_VOICE_NOT_CONFIGURED = (
    "Module 34 is not wired here (GD_MODULE_34_URL empty) -- its voice "
    "recordings can't be listed"
)
# Written into voice/recordings.json beside the list (GD-10 / VIE01-416).
_VOICE_NOTE = (
    "Metadata of each stored voice session recording (WAV, kept until "
    "expires_at). The audio itself is not in this export. "
    "pending_sessions counts sessions still "
    "being recorded. Voice-clone speaker embeddings are counted in "
    "manifest.json (speaker_embeddings_cached): they are model tensors, not "
    "readable data."
)
_SPEAKER_EMBEDDINGS_NOT_DEPLOYED = (
    "Module 35 is not deployed here (GD_MODULE_35_URL empty) -- no voice-clone "
    "speaker embeddings can exist"
)

Blobs = list[tuple[str, bytes]]

# Download links: SigV4, path-style (bucket in the path, as SeaweedFS serves it).
# SeaweedFS ignores the region, but SigV4 needs one to sign with.
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


class ExportOrchestrator:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool
        self.http = httpx.AsyncClient(timeout=120.0)
        self.s3_session = aioboto3.Session()
        self.audit = audit_writer(pool)

    async def _audit(self, job: ExportJob, ok: bool, metadata: dict[str, Any]) -> None:
        await self.audit.emit(privacy_event(
            "gdpr.export", user_id=job.user_id, action="export", ok=ok, requested_by=job.requested_by,
            resource_id=job.job_id, metadata=metadata))

    async def close(self) -> None:
        await self.http.aclose()

    # ── Job lookup / creation (gdpr_erasure_jobs, type='export') ───────────

    async def create_job(self, user_id: str, requested_by: str) -> ExportJob:
        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO gdpr_erasure_jobs
                    (job_id, user_id, requested_at, requested_by, status, store_statuses, type, completed_at, notes)
                VALUES ($1, $2, $3, $4, $5, $6, 'export', NULL, NULL)
                """,
                job_id, user_id, now, requested_by, JobStatus.PENDING.value, json.dumps({}),
            )
        return ExportJob(job_id=job_id, user_id=user_id, requested_at=now, requested_by=requested_by, status=JobStatus.PENDING)

    async def get_job(self, job_id: str) -> ExportJob | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM gdpr_erasure_jobs WHERE job_id = $1 AND type = 'export'", job_id,
            )
        return _row_to_export_job(row) if row else None

    async def _update_job(
        self, job_id: str, *, status: JobStatus | None = None,
        completed_at: datetime | None = None, notes: str | None = None,
        download_urls: list[str] | None = None, size_bytes: int | None = None,
        expires_at: datetime | None = None,
    ) -> None:
        sets, args = [], []
        for col, val in (
            ("status", status.value if status is not None else None),
            ("completed_at", completed_at), ("notes", notes),
            ("download_urls", json.dumps(download_urls) if download_urls is not None else None),
            ("size_bytes", size_bytes), ("expires_at", expires_at),
        ):
            if val is not None:
                args.append(val)
                sets.append(f"{col} = ${len(args)}")
        if not sets:
            return
        args.append(job_id)
        async with self.pool.acquire() as conn:
            await conn.execute(f"UPDATE gdpr_erasure_jobs SET {', '.join(sets)} WHERE job_id = ${len(args)}", *args)  # nosec B608 -- sets built above from a hardcoded ("col", val) tuple, column names never request-derived

    # ── Orchestration entry point (run as a background task) ───────────────

    async def run(self, job_id: str) -> None:
        job = await self.get_job(job_id)
        if job is None:
            logger.error("run() called for unknown export job_id=%s", job_id)
            return

        await self._update_job(job_id, status=JobStatus.IN_PROGRESS)
        user_id = job.user_id
        exported_at = datetime.now(timezone.utc)

        stores_included: list[str] = []
        stores_failed: list[str] = []
        blobs: Blobs = []

        async def collect(store: str, fetch: Callable[[str], Awaitable[Any]], layout: Callable[[Any], Blobs]) -> None:
            try:
                blobs.extend(layout(await fetch(user_id)))
                stores_included.append(store)
            except Exception as exc:  # noqa: BLE001 -- a store fetch can fail for any reason; recorded in stores_failed and logged, partial export continues
                logger.error("export job=%s: %s failed: %s", job_id, store, exc)
                stores_failed.append(store)

        await collect("module_06", self._fetch_module_06, lambda p: [("identity/keycloak_profile.json", _dump(p))])
        await collect("module_04", self._fetch_module_04, lambda files: [("files/index.json", _dump(files))])
        await collect("module_25", self._fetch_module_25, _content_layout)
        await collect("module_26", self._fetch_module_26, _memory_layout)
        await collect("module_03", self._fetch_module_03, lambda b: [("benchmark/runs.json", _dump(b))])
        await collect("module_23", self._fetch_module_23, lambda f: [("moderation/flags.json", _dump(f))])

        omitted = dict(_OMITTED_SECTIONS)
        if settings.module_34_url:
            await collect("module_34", self._fetch_module_34, lambda v: [("voice/recordings.json", _dump(v))])
        else:
            omitted["voice/recordings.json"] = _VOICE_NOT_CONFIGURED
        speaker_embeddings: int | None = None
        if not settings.module_35_url:
            omitted["manifest.speaker_embeddings_cached"] = _SPEAKER_EMBEDDINGS_NOT_DEPLOYED
        else:
            try:
                speaker_embeddings = await self._fetch_module_35(user_id)
                stores_included.append("module_35")
            except Exception as exc:  # noqa: BLE001 -- same partial-export policy as collect()
                logger.error("export job=%s: module_35 failed: %s", job_id, exc)
                stores_failed.append("module_35")

        parts = _split(blobs, settings.export_max_zip_bytes)
        part_names = _part_names(len(parts))
        manifest = {
            "export_schema_version": EXPORT_SCHEMA_VERSION,
            "user_id": user_id,
            "exported_at": exported_at.isoformat(),
            "gdpr_article": GDPR_ARTICLE,
            "ccpa_section": CCPA_SECTION,
            "stores_included": stores_included,
            "stores_failed": stores_failed,
            "parts": part_names,
            # Count only -- the embeddings are opaque model tensors, not
            # exportable personal data in any readable form (per the ticket).
            "speaker_embeddings_cached": speaker_embeddings,
            "omitted": omitted,
            "job_id": job_id,
        }
        parts[0].insert(0, ("manifest.json", _dump(manifest)))

        try:
            download_urls, size_bytes = await self._upload_parts(job_id, user_id, exported_at, parts, part_names)
        except Exception as exc:  # noqa: BLE001 -- ZIP build (file-IO) and S3 upload (botocore ClientError/BotoCoreError) span multiple error types; already logged and recorded as job failure
            logger.error("export job=%s: zip build/upload failed: %s", job_id, exc)
            await self._update_job(job_id, status=JobStatus.PARTIAL_FAILURE, notes=f"ZIP build/upload failed: {exc}")
            await self._audit(job, False, {"stores_failed": stores_failed, "upload_failed": True})
            return

        status = JobStatus.COMPLETED if not stores_failed else JobStatus.PARTIAL_FAILURE
        notes = f"stores_failed: {', '.join(stores_failed)}" if stores_failed else None
        await self._update_job(
            job_id, status=status, completed_at=datetime.now(timezone.utc),
            download_urls=download_urls, size_bytes=size_bytes,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=settings.export_ttl_hours),
            notes=notes,
        )
        await self._audit(job, not stores_failed, {"stores_failed": stores_failed, "parts": len(part_names)})

    # ── Per-store fetchers ───────────────────────────────────────────────

    async def _fetch_module_04(self, user_id: str) -> list[dict]:
        """files/index.json: metadata and a download link per stored file,
        valid for the export's own TTL -- Module 04 presigns them, so the
        ZIP never carries file bytes (the ticket's guard against huge ZIPs
        from large video uploads)."""
        resp = await self.http.get(
            f"{settings.module_04_url}/internal/gdpr/user/{user_id}/files", headers=_INTERNAL_HEADERS,
            params={"expires_in": int(settings.export_ttl_hours * 3600)},
        )
        resp.raise_for_status()
        return resp.json()["files"]

    async def _fetch_module_25(self, user_id: str) -> dict:
        resp = await self.http.get(
            f"{settings.module_25_url}/internal/gdpr/user/{user_id}/export", headers=_INTERNAL_HEADERS,
        )
        resp.raise_for_status()
        return resp.json()

    async def _fetch_module_26(self, user_id: str) -> dict:
        resp = await self.http.get(
            f"{settings.module_26_url}/internal/gdpr/user/{user_id}/export", headers=_INTERNAL_HEADERS,
        )
        resp.raise_for_status()
        return resp.json()

    async def _fetch_module_34(self, user_id: str) -> dict:
        resp = await self.http.get(
            f"{settings.module_34_url}/internal/gdpr/user/{user_id}/recordings", headers=_INTERNAL_HEADERS,
        )
        resp.raise_for_status()
        body = resp.json()
        return {"note": _VOICE_NOTE, "recordings": body["recordings"], "pending_sessions": body["pending_sessions"]}

    async def _fetch_module_35(self, user_id: str) -> int:
        resp = await self.http.get(
            f"{settings.module_35_url}/internal/gdpr/user/{user_id}/summary", headers=_INTERNAL_HEADERS,
        )
        resp.raise_for_status()
        return int(resp.json()["speaker_embeddings"])

    async def _keycloak_admin_token(self) -> str:
        resp = await self.http.post(
            f"{settings.keycloak_url}/realms/master/protocol/openid-connect/token",
            data={
                "grant_type": "password", "client_id": "admin-cli",
                "username": settings.keycloak_admin_username, "password": settings.keycloak_admin_password,
            },
        )
        resp.raise_for_status()
        return resp.json()["access_token"]

    async def _fetch_module_06(self, user_id: str) -> dict | None:
        token = await self._keycloak_admin_token()
        resp = await self.http.get(
            f"{settings.keycloak_url}/admin/realms/{settings.keycloak_realm}/users/{user_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        profile = resp.json()
        profile.pop("access", None)  # admin-console UI flags, not the user's own data
        return profile

    async def _fetch_module_03(self, user_id: str) -> dict:
        """benchmark/runs.json -- the ticket's "test cases + results owned by
        user": runs (results) by branch_id, the same scope GD-1's erasure
        deleter uses, and schemas (test cases) by GD-5's owner_id. Schemas
        created before GD-5 have no owner and can't be attributed. Direct
        SQL against Module 03's `benchmark` database, as before."""
        conn = await asyncpg.connect(
            host=settings.postgres_host, port=settings.postgres_port,
            user=settings.postgres_user, password=settings.postgres_password,
            database=settings.benchmark_postgres_db,
        )
        try:
            runs = await conn.fetch("SELECT * FROM benchmark_runs WHERE branch_id = $1", user_id)
            schemas = await conn.fetch("SELECT * FROM benchmark_schemas WHERE owner_id = $1", user_id)
        finally:
            await conn.close()
        return {"runs": [dict(r) for r in runs], "schemas": [dict(s) for s in schemas]}

    async def _fetch_module_23(self, user_id: str) -> list[dict]:
        """moderation/flags.json -- this user's own moderation_flags rows
        (Module 23, CM-5 / VIE01-408). Distinct from audit/violations.json
        (guardrail_violations, still omitted -- see _OMITTED_SECTIONS):
        moderation_flags has a branch_id column, guardrail_violations has
        no owner column at all. Direct SQL against the `guardrails`
        database, same connect-directly pattern as _fetch_module_03."""
        conn = await asyncpg.connect(
            host=settings.postgres_host, port=settings.postgres_port,
            user=settings.postgres_user, password=settings.postgres_password,
            database=settings.guardrails_postgres_db,
        )
        try:
            rows = await conn.fetch("SELECT * FROM moderation_flags WHERE branch_id = $1", user_id)
        finally:
            await conn.close()
        return [dict(r) for r in rows]

    # ── ZIP build + S3 upload ───────────────────────────────────────────────

    async def _ensure_bucket(self, s3_client, bucket: str) -> None:
        try:
            await s3_client.head_bucket(Bucket=bucket)
        except botocore.exceptions.ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code not in ("404", "NoSuchBucket"):
                raise
            try:
                await s3_client.create_bucket(Bucket=bucket)
            except botocore.exceptions.ClientError as ce:
                if ce.response["Error"]["Code"] not in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
                    raise

    async def _upload_parts(
        self, job_id: str, user_id: str, exported_at: datetime, parts: list[Blobs], part_names: list[str],
    ) -> tuple[list[str], int]:
        """Stored as gdpr-exports/{job_id}/<part name>; downloaded as the
        ticket's gdpr-export-{user_id}-{timestamp}.zip (-part-N for a split
        export) via Content-Disposition on the presigned URL.

        Uploads go to the internal endpoint; links are signed directly
        against the public endpoint with SigV4 via _presign_path_style_get()
        (not a real client -- settings.s3_public_url can be a hostname
        boto3's own client construction refuses, see its docstring). SigV4
        signs the host, so the link must be signed for the host the user
        actually downloads from, not rewritten afterwards -- and SeaweedFS
        rejects SigV2 links that carry response-content-disposition (403)."""
        stamp = exported_at.strftime("%Y%m%dT%H%M%SZ")
        async with self.s3_session.client(
            "s3", endpoint_url=settings.s3_endpoint,
            aws_access_key_id=settings.aws_access_key, aws_secret_access_key=settings.aws_secret_key,
        ) as s3_client:
            await self._ensure_bucket(s3_client, settings.s3_bucket)

            download_urls: list[str] = []
            total_size = 0
            for part, name in zip(parts, part_names):
                # mkstemp, not mktemp -- atomically creates the file so there's
                # no window between name generation and open for a symlink/
                # race attack to land in. _write_zip's zipfile.ZipFile(path, "w")
                # truncates on open either way, so a pre-existing empty file is fine.
                fd, tmp_path = tempfile.mkstemp(suffix=".zip")
                os.close(fd)
                try:
                    total_size += await asyncio.to_thread(_write_zip, tmp_path, part)
                    s3_key = f"gdpr-exports/{job_id}/{name}"
                    await s3_client.upload_file(tmp_path, settings.s3_bucket, s3_key)
                    download_name = f"gdpr-export-{user_id}-{stamp}{name.removeprefix('export')}"
                    presigned = _presign_path_style_get(
                        settings.s3_public_url or settings.s3_endpoint, settings.s3_bucket, s3_key,
                        {"response-content-disposition": f'attachment; filename="{download_name}"'},
                        int(settings.export_ttl_hours * 3600),
                    )
                    download_urls.append(presigned)
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
            return download_urls, total_size

    # ── Cleanup sweep (expired export ZIPs) ─────────────────────────────────

    async def cleanup_expired(self) -> int:
        """Deletes S3 objects for export jobs past their expires_at, and
        clears download_urls/size_bytes so a stale link is never returned by
        a subsequent poll. Returns the number of jobs cleaned."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT job_id FROM gdpr_erasure_jobs
                WHERE type = 'export' AND expires_at IS NOT NULL
                  AND expires_at < now() AND download_urls IS NOT NULL
                """,
            )
        if not rows:
            return 0

        async with self.s3_session.client(
            "s3", endpoint_url=settings.s3_endpoint,
            aws_access_key_id=settings.aws_access_key, aws_secret_access_key=settings.aws_secret_key,
        ) as s3_client:
            for row in rows:
                job_id = str(row["job_id"])
                prefix = f"gdpr-exports/{job_id}/"
                try:
                    listing = await s3_client.list_objects_v2(Bucket=settings.s3_bucket, Prefix=prefix)
                    for obj in listing.get("Contents", []):
                        await s3_client.delete_object(Bucket=settings.s3_bucket, Key=obj["Key"])
                except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as exc:
                    logger.error("cleanup: failed to delete S3 objects for export job=%s: %s", job_id, exc)
                    continue
                async with self.pool.acquire() as conn:
                    await conn.execute(
                        "UPDATE gdpr_erasure_jobs SET download_urls = NULL, size_bytes = NULL WHERE job_id = $1",
                        row["job_id"],
                    )
        logger.info("cleanup: expired %d export job(s)", len(rows))
        return len(rows)


# ── ZIP layout ──────────────────────────────────────────────────────────────

def _content_layout(m25: dict) -> Blobs:
    return [(f"content/{kind}.json", _dump(m25.get(kind, []))) for kind in ("documents", "entities", "moments")]


def _memory_layout(m26: dict) -> Blobs:
    blobs: Blobs = [("memory/long_term.json", _dump(m26.get("longterm", [])))]
    used: set[str] = set()
    for session in m26.get("sessions", []):
        name = _safe_name(str(session.get("session_id") or "session"))
        while name in used:  # two ids that sanitize alike must not overwrite each other
            name += "_"
        used.add(name)
        blobs.append((f"memory/sessions/{name}.json", _dump(session)))
    blobs.append(("memory/multimedia.json", _dump(m26.get("multimedia", []))))
    return blobs


def _safe_name(name: str) -> str:
    """A session id as a single ZIP path segment -- no separators, no '..'."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name).strip(".") or "session"


def _split(blobs: Blobs, max_bytes: int) -> list[Blobs]:
    """Split at max_bytes on raw (pre-compression) content size -- an
    approximation, but compression only ever makes the real ZIP smaller
    than this estimate, never larger, so it's a safe bound. Always at
    least one part, so the manifest has somewhere to go."""
    parts: list[Blobs] = [[]]
    part_size = 0
    for path, content in blobs:
        if part_size + len(content) > max_bytes and parts[-1]:
            parts.append([])
            part_size = 0
        parts[-1].append((path, content))
        part_size += len(content)
    return parts


def _part_names(count: int) -> list[str]:
    return ["export.zip"] if count == 1 else [f"export-part-{i}.zip" for i in range(1, count + 1)]


def _write_zip(path: str, entries: Blobs) -> int:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for entry_path, content in entries:
            zf.writestr(entry_path, content)
    return os.path.getsize(path)


def _dump(obj: Any) -> bytes:
    return json.dumps(obj, default=str, indent=2, ensure_ascii=False).encode("utf-8")


def _row_to_export_job(row: Any) -> ExportJob:
    download_urls = row["download_urls"]
    if isinstance(download_urls, str):
        download_urls = json.loads(download_urls)
    return ExportJob(
        job_id=str(row["job_id"]),
        user_id=row["user_id"],
        requested_at=row["requested_at"],
        requested_by=row["requested_by"],
        status=JobStatus(row["status"]),
        notes=row["notes"],
        completed_at=row["completed_at"],
        download_url=download_urls[0] if download_urls else None,
        download_urls=download_urls,
        size_bytes=row["size_bytes"],
        expires_at=row["expires_at"],
    )
