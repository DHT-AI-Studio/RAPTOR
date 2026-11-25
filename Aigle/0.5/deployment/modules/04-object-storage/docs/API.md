# Asset Management Service API Documentation

This service provides comprehensive lifecycle management for assets, including uploading, version control, associated file management, and automated archiving or destruction. It integrates **LakeFS** (Version Control), **SeaweedFS** (Object Storage), **PostgreSQL** (Metadata), and **Qdrant** (Vector Database).

---

## 1. General Information

### 1.1 Authentication & Context
Authentication is handled by an API Gateway. The Gateway must inject these mandatory headers:

| Header | Description | Example |
| :--- | :--- | :--- |
| `X-User-ID` | Unique ID of the user. | `550e8400-e29b-41d4-a716-446655440000` |
| `X-Branch-ID` | The storage namespace identifier. | `group1` or `user-uuid` |

### 1.2 Workspace & Branch Isolation
`X-Branch-ID` determines the storage environment:
*   **Shared Workspace:** `X-Branch-ID` = Group ID (e.g., `group1`). Users share the same space.
*   **Private Workspace:** `X-Branch-ID` and `X-User-ID` are both set to the **User's UUID**. Space is isolated to that user.

### 1.3 Asset Lifecycle Status
- **`uploaded`**: Committed to storage, not yet analyzed (upload and analysis are decoupled -- see `/fileupload` vs `/process-file` on Module 13).
- **`processing`**: A worker (Modules 09-12) has claimed the job and is actively analyzing it.
- **`active`**: Analysis finished successfully; asset is live and accessible.
- **`failed`**: Analysis finished unsuccessfully. Re-registering for analysis (Module 13's `/process-file`) is allowed from this state.
- **`unregistered`**: Taken out of search and index results by Unregister (2.5) -- nothing is deleted. Register (2.12) fast-paths it straight back to `active` with no re-analysis.
- **`archived`**: Read-only state, pending destruction.
- **`destroyed`**: Metadata purged and physical files deleted (this status is only ever returned in an API response, never persisted -- see `/delfile`).

### 1.4 Date & TTL Formats
Fields ending in **`_date_or_ttl`** support:
- Relative TTL: e.g., **`30d`** (days), **`2w`** (weeks), **`6m`** (months), **`1y`** (years).
- Absolute Date: **`YYYY-MM-DD`** or **`YYYY/MM/DD`**.
- ISO Datetime: **`YYYY-MM-DDTHH:MM:SS`**.

---

## 2. API Endpoints

### 2.1 Upload Asset

Uploads a primary file and optional associated files.
The system identifies duplicates based on the combination of **File Content (Checksum)** and **Asset Path** within the specific **Branch ID** (Storage Space).

#### Upload Logic Matrix

| Scenario | Content (Checksum) | Asset Path | Branch DB Status (Latest) | System Action | Qdrant/Analysis Handling |
| --- | --- | --- | --- | --- | --- |
| **1. New Upload** | New in Branch | New Path | No Record | **Upload & Analyze** | Create new **Active** Point |
| **2. Version Update** | **Different** | Same Path | `active` or `archived` | **Upload & Analyze (New Version)** | Create new **Active** Points |
| **3. Global Active Hit** | **Same** in Branch | Any Path | `active` | **Merge/Return Info** | No Change (Return active metadata) |
| **4. Archive Lock** | **Same** in Branch | Same Path | **`archived`** | **Reject (400 Error)** | Forbidden (Cannot reactivate archived records, Integrity protection) |
| **5. Cross-Path Reuse** | **Same** in Branch | **Different Path** | **`archived`** | **Upload & Reuse** | **Clone Point**: Reuse vectors/analysis for new **Active** Point |

> **Business Rule**: Within a specific **Branch**, if an asset at the asset path is already `archived` and the uploaded content is identical, the operation is rejected. This ensures the integrity of the archival state. To re-use the same path with the same content, the archived record must be explicitly deleted first.



- **URL**: `POST /fileupload`
- **Content-Type**: `multipart/form-data`

**cURL Example:**
```bash
curl -X POST "http://<host>:8000/fileupload" \
     -H "X-User-ID: user_01" \
     -H "X-Branch-ID: workspace_abc" \
     -F "primary_file=@test123.txt" \
     -F "associated_files=@test.jpg" \
     -F "archive_date_or_ttl=90d" \
     -F "destroy_date_or_ttl=1y"
```

**Response Example:**
```json
{
  "asset_path": "document/txt/test123",
  "version_id": "5da85b77b4f7c8651f33989ba141bd4c7fa74ec9a5ca5c30a72812de24aea3ec",
  "primary_filename": "test123.txt",
  "associated_filenames": [
    ["test.jpg", "5ae40aed0d2aa3038dcd7d22a2e052a2881ead115f387701a83473b77cf819fe"]
  ],
  "upload_date": "2025-12-31T13:18:05.166625+08:00",
  "archive_date": "2026-03-31T13:18:05.166625+08:00",
  "destroy_date": "2027-03-31T13:18:05.166625+08:00",
  "user": "user_01",
  "branch": "workspace_abc",
  "status": "active",
  "checksum": "31e49720a568e64dd4669e9a08b536bc",
  "existence_info": {
    "exists": false,
    "message": "The primary file is a new file",
    "existing_asset_path": null,
    "existing_version_id": null
  }
}
```

**Response Example (Error - Scenario 4):**

```json
{
  "detail": "Asset at this path within your branch is archived with identical content. Overwriting archived assets is forbidden. Please delete the archived record or change the file content/path."
}
```

---

### 2.2 Add Associated Files
Add extra files to an existing asset version.

- **URL**: `POST /add-associated-files/{asset_path:path}`

**cURL Example:**
```bash
curl -X POST "http://<host>:8000/add-associated-files/document/txt/test123" \
     -H "X-User-ID: user_01" \
     -H "X-Branch-ID: workspace_abc" \
     -F "associated_files=@98042.pdf"
```

**Response Example:**
```json
{
  "asset_path": "document/txt/test123",
  "version_id": "5da85b77b4f7c8651f33989ba141bd4c7fa74ec9a5ca5c30a72812de24aea3ec",
  "primary_filename": "test123.txt",
  "associated_filenames": [
    ["test.jpg", "5ae40aed0d2aa3038dcd7d22a2e052a2881ead115f387701a83473b77cf819fe"],
    ["98042.pdf", "2a9e33108b4047ffa7155aa88467c0355361691bc3e4fdbae0eb5ae751bb5c39"]
  ],
  "upload_date": "2025-12-31T13:18:05.166625+08:00",
  "archive_date": "2026-03-31T13:18:05.166625+08:00",
  "destroy_date": "2027-03-31T13:18:05.166625+08:00",
  "status": "active"
}
```

---

### 2.3 Download/Retrieve Asset
Fetch the metadata and generated presigned URLs for all files in an asset.

- **URL**: `GET /filedownload/{asset_path:path}/{version_id}`
- **Query Params**: `return_file_content` (bool)

**cURL Example:**
```bash
curl -X GET "http://<host>:8000/filedownload/document/txt/test123/5da85b77b4f7c8651f33989ba141bd4c7fa74ec9a5ca5c30a72812de24aea3ec?return_file_content=false" \
     -H "X-User-ID: user_01" \
     -H "X-Branch-ID: workspace_abc"
```

**Response Example:**
```json
{
  "metadata": {
    "asset_path": "document/txt/test123",
    "version_id": "5da85b77b4f7c8651f33989ba141bd4c7fa74ec9a5ca5c30a72812de24aea3ec",
    "primary_filename": "test123.txt",
    "associated_filenames": [
      ["test.jpg", "..."],
      ["98042.pdf", "..."]
    ],
    "status": "active"
  },
  "primary_file": {
    "filename": "test123.txt",
    "content_type": "text/plain",
    "version_id": "5da85b...",
    "url": "http://192.168.157.123:8333/lakefs/data/..."
  },
  "associated_file_1": {
    "filename": "test.jpg",
    "content_type": "image/jpeg",
    "version_id": "...",
    "url": "..."
  },
  "associated_file_2": {
    "filename": "98042.pdf",
    "content_type": "application/pdf",
    "version_id": "...",
    "url": "..."
  }
}
```

---

### 2.4 Archive Asset
Manually transition an asset to `archived` status. Callable from any pre-archive state -- `uploaded`, `processing`, `active`, `failed`, or `unregistered` -- not just `active`. `400` if the asset is already `archived` or `destroyed` (the latter has had its underlying files physically deleted by Destroy, so there is nothing left to archive).

Archive is one-way: there is no unarchive endpoint anywhere on this service. Once archived, an asset's only further transition is Destroy (`archived` -> `destroyed`, see 2.6). Both Register (`POST /processing/process-file` on Module 13's gateway -- undocumented in markdown, see its own OpenAPI description) and Unregister (2.5) return `409` on an archived asset, same as on any other state they don't accept.

- **URL**: `POST /filearchive/{asset_path:path}/{version_id}`

**cURL Example:**
```bash
curl -X POST "http://<host>:8000/filearchive/document/txt/test123/5da85b...ec" \
     -H "X-User-ID: user_01" \
     -H "X-Branch-ID: workspace_abc"
```

**Response Example:**
```json
{
  "asset_path": "document/txt/test123",
  "version_id": "5da85b77b4f7c8651f33989ba141bd4c7fa74ec9a5ca5c30a72812de24aea3ec",
  "primary_filename": "test123.txt",
  "associated_filenames": [
    ["test.jpg", "5ae40aed0d2aa3038dcd7d22a2e052a2881ead115f387701a83473b77cf819fe"],
    ["98042.pdf", "2a9e33108b4047ffa7155aa88467c0355361691bc3e4fdbae0eb5ae751bb5c39"]
  ],
  "upload_date": "2025-12-31T13:18:05.166625+08:00",
  "archive_date": "2026-03-31T13:18:05.166625+08:00",
  "destroy_date": "2027-03-31T13:18:05.166625+08:00",
  "status": "archived"
}
```

---

### 2.5 Unregister Asset
Filter-only: transitions an `active` asset to `unregistered`, dropping it out of search/index results **without deleting any embedding, vector, or graph data**. Only legal from `active` -- `409` otherwise, including when already `unregistered` (not idempotent). Reversible via `POST /register` (2.12; users call it through Module 13's `POST /processing/process-file`), which fast-paths straight back to `active` with no re-analysis and no Kafka dispatch.

- **URL**: `POST /unregister/{asset_path:path}/{version_id}`

**cURL Example:**
```bash
curl -X POST "http://<host>:8000/unregister/document/txt/test123/5da85b...ec" \
     -H "X-User-ID: user_01" \
     -H "X-Branch-ID: workspace_abc"
```

**Response Example:**
```json
{
  "asset_path": "document/txt/test123",
  "version_id": "5da85b77b4f7c8651f33989ba141bd4c7fa74ec9a5ca5c30a72812de24aea3ec",
  "primary_filename": "test123.txt",
  "associated_filenames": [],
  "upload_date": "2025-12-31T13:18:05.166625+08:00",
  "archive_date": null,
  "destroy_date": null,
  "status": "unregistered"
}
```

**Error Response (asset not active):**
```json
{"detail": "Asset document/txt/test123/5da85b...ec is archived, not active -- cannot unregister"}
```
`409 Conflict`

---

### 2.6 Destroy Asset
Permanently delete an asset. Asset must be `archived` first.

- **URL**: `POST /delfile/{asset_path:path}/{version_id}`

**cURL Example:**
```bash
curl -X POST "http://<host>:8000/delfile/document/txt/test123/5da85b...ec" \
     -H "X-User-ID: user_01" \
     -H "X-Branch-ID: workspace_abc"
```

**Response Example:**
```json
{
  "asset_path": "document/txt/test123",
  "version_id": "5da85b77b4f7c8651f33989ba141bd4c7fa74ec9a5ca5c30a72812de24aea3ec",
  "primary_filename": "test123.txt",
  "associated_filenames": [
    ["test.jpg", "5ae40aed0d2aa3038dcd7d22a2e052a2881ead115f387701a83473b77cf819fe"],
    ["98042.pdf", "2a9e33108b4047ffa7155aa88467c0355361691bc3e4fdbae0eb5ae751bb5c39"]
  ],
  "upload_date": "2025-12-31T13:18:05.166625+08:00",
  "archive_date": "2026-03-31T13:18:05.166625+08:00",
  "destroy_date": "2027-03-31T13:18:05.166625+08:00",
  "status": "destroyed"
}
```

---

### 2.7 List File Versions
Retrieve all historical versions for a specific file.

- **URL**: `GET /fileversions/{asset_path:path}/{filename}`

**cURL Example:**
```bash
curl -X GET "http://<host>:8000/fileversions/document/txt/test123/test123.txt" \
     -H "X-User-ID: user_01" \
     -H "X-Branch-ID: workspace_abc"
```

**Response Example:**
```json
[
  {
    "key": "document/txt/test123/test123.txt",
    "version_id": "5da85b77b4f7c8651f33989ba141bd4c7fa74ec9a5ca5c30a72812de24aea3ec",
    "upload_date": "2025-12-31T13:18:05.166625+08:00",
    "status": "archived",
    "url": "http://192.168.157.123:8333/lakefs/data/..."
  },
  ...
]
```

---

### 2.8 Get User Commit History
Get paginated history of assets uploaded/modified by the current user context.

- **URL**: `GET /users/commits`
- **Query Params**: `keyword`, `start_date`, `end_date`, `page`, `page_size`, `status` (repeatable).

**Status filtering:** `status` may be repeated to match any of the 6 values
(`uploaded`, `processing`, `active`, `failed`, `unregistered`, `archived`) via SQL `status = ANY(...)`.
**Omitting `status` entirely returns all 6 statuses** — this endpoint never defaults to "active only"
so a caller that only wants one status must filter explicitly.

**cURL Example:**
```bash
curl -X GET "http://<host>:8000/users/commits?keyword=test&page=1&page_size=5" \
     -H "X-User-ID: user_01" \
     -H "X-Branch-ID: workspace_abc"

# Only failed and processing uploads:
curl -X GET "http://<host>:8000/users/commits?status=failed&status=processing" \
     -H "X-User-ID: user_01" \
     -H "X-Branch-ID: workspace_abc"
```

**Response Example:**
```json
{
  "total_count": 1,
  "total_pages": 1,
  "page": 1,
  "page_size": 5,
  "commits": [
    {
      "asset_path": "document/txt/test123",
      "version_id": "5da85b77b4f7c8651f33989ba141bd4c7fa74ec9a5ca5c30a72812de24aea3ec",
      "primary_filename": "test123.txt",
      "status": "archived"
    },
    ...
  ]
}
```

---

### 2.9 Update Expiration Policies
Modify archiving or destruction dates. `400` if the asset is currently `archived` -- consistent with 2.4: archive is one-way, so an archived asset cannot be reactivated or have its dates changed by this endpoint either. The only way out of `archived` is Destroy (2.6).

- **URL**: `POST /file-expiration/{asset_path:path}/{version_id}`

**cURL Example:**
```bash
curl -X POST "http://<host>:8000/file-expiration/document/pdf/sample_report/8eeebfa..." \
     -H "X-User-ID: user_01" \
     -H "X-Branch-ID: workspace_abc" \
     -F "archive_date_or_ttl=2026-01-01" \
     -F "destroy_date_or_ttl=2027-09-30"
```

**Response Example:**
```json
{
  "asset_path": "document/pdf/雜糧及蔬菜_PCR_2.0版",
  "version_id": "8eeebfa2c3f412caa5f29c9c92c62664cec2c35aed6b581ae1aa578eee0405c3",
  "primary_filename": "雜糧及蔬菜_PCR_2.0版.pdf",
  "associated_filenames": [
    ["ISO14067_2018.pdf", "3816f48ca391d270e67bb98f5e691cfa8ea1a302c3c4dfd6b01d4e8adcb72f01"],
    ["Raptor_test_20251217.pdf", "3ede9962f2196b723239907c4a959f9818029376373b8567ad5360dba313c3e8"]
  ],
  "upload_date": "2025-12-31T10:37:12.981402+08:00",
  "archive_date": "2026-01-01T00:00:00+08:00",
  "destroy_date": "2027-09-30T00:00:00+08:00",
  "status": "active"
}
```

---

### 2.10 Video Thumbnail
Extracts a single JPEG frame from a video asset at a given timestamp, via GPU-accelerated decode (NVDEC). Only legal on an `active` asset -- `409` otherwise, `404` if the asset/version doesn't exist, `400` if `second` is negative or beyond the video's actual duration (probed via `ffprobe` before decode is attempted).

This endpoint never downloads the full video file. It calls the same `generate_presigned_url()` LakeFS uses for Download (2.3), then hands that URL straight to `ffmpeg`/`ffprobe` with `-ss <second>` placed *before* `-i <url>` -- ffmpeg's own HTTP protocol handler issues byte-range requests when seeking into an indexed container (mp4 moov atom, mkv cues, etc.), rather than reading from byte 0. No manual Range-header logic lives in this service.

**Response format decision**: raw JPEG bytes (`Content-Type: image/jpeg`), not a second presigned URL. A thumbnail is small (a single frame, `-q:v 2` JPEG), so the extra round trip a presigned-URL response would add (client gets a URL, then has to make a second request to actually fetch the image) outweighs any benefit -- unlike the source video file itself, which is large enough that forcing it through this service's own process would be wasteful. The 7-day cache stores the actual JPEG bytes in Redis under `thumbnail:{asset_path}:{version_id}:{second}`, so a cache hit skips `ffmpeg`/`ffprobe` entirely and returns straight from Redis.

GPU access (`-hwaccel cuda -hwaccel_output_format cuda`) lives only in the separate `thumbnail-worker` sidecar container (`docker-compose.yml`), not Module 04's main `app` container -- same split module 11 (video-processing) uses between its GPU-needing stages and its CPU-only ones. `thumbnail_service.py` calls the worker over HTTP (internal network only); the worker owns no database, cache, or asset state, only the ffprobe/ffmpeg call.

- **URL**: `GET /thumbnail/{asset_path:path}/{version_id}?second={seconds}`

**cURL Example:**
```bash
curl -X GET "http://<host>:8000/thumbnail/video/mp4/sample_clip/8eeebfa...?second=12.5" \
     -H "X-User-ID: user_01" \
     -H "X-Branch-ID: workspace_abc" \
     --output thumbnail.jpg
```

**Response**: raw JPEG bytes, `Content-Type: image/jpeg`.

**Error Responses:**
| Status | Condition |
|---|---|
| `404` | Asset/version does not exist |
| `409` | Asset exists but is not `active` |
| `400` | `second` is negative, or beyond the video's actual duration |
| `502` | `ffprobe`/`ffmpeg` failed or timed out (e.g. corrupt file, GPU unavailable) |

---

### 2.11 Batch Duplicate-Content Scan
Given a list of `(path, checksum)` pairs, reports whether each checksum already exists for the caller's branch -- no upload, read-only. Lets a client skip uploading files it already has stored under a different name.

Stored checksums are raw MD5 hex (`calculate_md5_bytes`, the same hash Upload computes) -- the ticket's own example used a `sha256:` prefix, which doesn't match. This endpoint accepts an optional `<algo>:` prefix on the request (ignored -- never checked against the actual algorithm, since this codebase doesn't use SHA-256 anywhere), strips it, and compares the remaining hex against the stored MD5. A checksum that isn't 32 hex characters (with or without a prefix) is malformed.

Each file's `status` is one of the **6-state vocabulary** (`uploaded`, `processing`, `active`, `failed`, `unregistered`, `archived`) if a match exists, or `not_found` if it doesn't. `destroyed` is never returned -- Destroy (2.6) deletes the row outright, so there's nothing left to match. Lookups run with a configurable concurrency cap (`SCAN_CONCURRENCY`, default 10), one DB connection per in-flight file, not a single batched query.

An empty `files` list returns an empty `results` list with `200`, not an error. A malformed checksum anywhere in the batch rejects the whole request with `400` -- not a per-file error mixed into the status list.

- **URL**: `POST /scan`

**cURL Example:**
```bash
curl -X POST "http://<host>:8000/scan" \
     -H "X-User-ID: user_01" \
     -H "X-Branch-ID: workspace_abc" \
     -H "Content-Type: application/json" \
     -d '{"files": [
           {"path": "report.pdf", "checksum": "8732e727e4c13c36cf18d31a4914652e"},
           {"path": "photo.jpg", "checksum": "sha256:ffffffffffffffffffffffffffffffff"}
         ]}'
```

**Response Example:**
```json
{
  "results": [
    {
      "path": "report.pdf",
      "checksum": "8732e727e4c13c36cf18d31a4914652e",
      "status": "active",
      "existing_asset_path": "document/pdf/report",
      "existing_version_id": "76cf1890...e8be"
    },
    {
      "path": "photo.jpg",
      "checksum": "ffffffffffffffffffffffffffffffff",
      "status": "not_found"
    }
  ]
}
```

**Error Responses:**
| Status | Condition |
|---|---|
| `400` | Any file's checksum isn't valid hex (with or without an `<algo>:` prefix) |

---

### 2.12 Register Asset -- internal
Called by Module 13's Register API (`POST /api/{v}/processing/process-file`), not by clients directly. Decides, from the asset's current status, whether it needs analysis. Never writes `processing` itself -- the worker that picks up the job does, via 2.13.

- **URL**: `POST /register/{asset_path:path}/{version_id}` (not in this service's OpenAPI schema)

| Current status | Result |
| :--- | :--- |
| not found | `404` |
| `processing`, `active`, `archived` | `409` `{"detail": "Asset is already <status>"}` |
| `unregistered` | `200` `outcome: "registered"` -- Module 25 confirms the content is still indexed, status flips to `active` synchronously, no re-analysis |
| `uploaded` / `failed` with identical content already indexed (`cloned_from` set at upload) | `200` `outcome: "registered"` -- straight to `active`, no re-analysis |
| `uploaded` / `failed` otherwise | `200` `outcome: "needs_processing"` -- status unchanged; Module 13 then publishes the Kafka analysis request |

**Response Example:**
```json
{
  "outcome": "registered",
  "metadata": {
    "asset_path": "video/mp4/clip",
    "version_id": "8eeebfa1...",
    "primary_filename": "clip.mp4",
    "associated_filenames": [],
    "upload_date": "2026-10-21T13:18:05+08:00",
    "archive_date": null,
    "destroy_date": null,
    "status": "active"
  }
}
```

---

### 2.13 Claim Processing -- internal
Called by the analysis orchestrator (Modules 09-12) when it picks up a Kafka job. Atomically moves `uploaded`/`failed` -> `processing` under the job's `correlation_id`. Retrying with the **same** `correlation_id` re-claims idempotently (`200`); a claim by a different `correlation_id`, or from any other status, is `409`.

- **URL**: `POST /claim-processing/{asset_path:path}/{version_id}?correlation_id={id}` (not in the OpenAPI schema)

**Response Example:**
```json
{"status": "processing", "correlation_id": "3f2b9c1e..."}
```

**Error Response:** `409` `{"detail": "Asset is not claimable (already claimed by another job, or in a non-claimable status)"}`

---

### 2.14 Update Status -- internal
Called by the orchestrator once a claimed job finishes. Writes `active` or `failed`, and best-effort syncs Module 25. The response carries the status **actually written**, not the one requested: a reported `active` is downgraded to `failed` if Module 25's own check finds the content wasn't indexed.

- **URL**: `POST /update-status/{asset_path:path}/{version_id}?status=active|failed` (not in the OpenAPI schema)

**Response Example:**
```json
{"asset_path": "video/mp4/clip", "version_id": "8eeebfa1...", "status": "active"}
```

---

### 2.15 Presigned URLs for Known Keys
For callers that already know a file's LakeFS key and version (or commit) -- e.g. Module 13 resolving face images for People Search and the Identity API (a face image is the video's associated file `{asset_path}/face_{trace_id}.jpg`, stored by Module 11). No metadata lookup and no ownership check: the caller is trusted to have verified access already.

**Single key** -- `GET /presign-url/{key:path}/{version_id}`
```json
{"url": "https://<s3-public>/...&X-Amz-Signature=..."}
```

**Many keys** -- `POST /presign-urls-batch`. Items under the same directory in the same commit are resolved with one LakeFS `list_objects` call instead of one call per key. Best-effort: an item that couldn't be resolved is simply missing from `resolved` (callers fall back to the single-key endpoint), never an error.

**cURL Example:**
```bash
curl -X POST "http://<host>:8000/presign-urls-batch" \
     -H "X-User-ID: user_01" \
     -H "X-Branch-ID: user_01" \
     -H "Content-Type: application/json" \
     -d '{"items": [
           {"key": "video/mp4/clip/face_t0003.jpg", "version_id": "c0ffee12..."},
           {"key": "identity/user_01/3b9d....jpg", "version_id": "user_01"}
         ]}'
```

**Response Example:**
```json
{
  "resolved": [
    {"key": "video/mp4/clip/face_t0003.jpg", "version_id": "c0ffee12...",
     "url": "https://<s3-public>/...", "content_type": "image/jpeg"}
  ]
}
```

---

### 2.16 Identity Seed Photo (dev plan v1.4 §4.5) -- internal
Called by Module 13's `POST /api/{v}/identity/{identity_id}/profile-image` after Module 11 has found a face in the photo. Stores the photo at the fixed key `identity/{branch}/{identity_id}.{ext}` as one LakeFS commit -- it is **not** an asset: no metadata row, no status, never indexed. The caller cannot choose the key. Re-uploading identical bytes makes no new commit (the branch is returned as the ref, serving the same content).

- **URL**: `POST /identity-photo/{identity_id}` (multipart field `photo`; not in the OpenAPI schema)

**Response Example:**
```json
{"image_path": "identity/user_01/3b9d2f0e-....jpg", "image_version_id": "c0ffee12..."}
```

**Error Responses:**
| Status | Condition |
|---|---|
| `400` | Invalid `identity_id`, or empty photo |
| `413` | Photo larger than 10 MB |
| `415` | Not JPEG, PNG or WebP (decided by the content, not the filename) |

---

## 3. Automated Lifecycle
The service includes a background scheduler:
1. **Auto-Archive**: Assets exceeding `archive_date` are set to `archived`.
2. **Auto-Destroy**: Assets exceeding `destroy_date` (while archived) are deleted.
3. **Log Cleanup**: Purges Audit Logs older than 120 days daily.
