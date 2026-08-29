# Raptor 0.4 API Reference

**Base URL:** `http://raptor_open_0_4_api.dhtsolution.com:8012`  
**API Version:** `0.4`  
**Authentication:** Bearer Token (Keycloak JWT)  

Swagger UI: `http://raptor_open_0_4_api.dhtsolution.com:8012/docs`

---

## Table of Contents

- [Authentication (Single Sign-On)](#sso)
- [Asset Management (Asset)](#asset)
- [Search](#search)
- [Video Search](#video-search)
- [RAG Query (A2A Protocol)](#rag)
- [Chat](#chat)
- [Memory Service](#memory)
- [Processing Status](#processing)
- [Benchmark](#benchmark)
- [Personal DB Service — Direct API (Module 25)](#personal-db)
- [Guardrail Service — Direct API (Module 23)](#guardrail)
- [Response Field Reference](#fields)

---

<a id="sso"></a>

## Authentication (Single Sign-On)

Every authenticated API call must carry a Bearer Token in the header:

```
Authorization: Bearer <access_token>
```

---

### `POST /api/0.4/sso/login`

User login — obtains a JWT access_token.

**Request:** `application/x-www-form-urlencoded`

| Parameter | Type | Required | Description |
|------|------|------|------|
| `username` | string | ✓ | Username |
| `password` | string | ✓ | Password |
| `realm_name` | string | | Keycloak realm, defaults to `dhtsolution` |
| `client_id` | string | | OAuth client, defaults to `raptor` (query param) |

**Response:**

```json
{
  "access_token": "eyJhbGci...",
  "token_type": "bearer",
  "refresh_token": "eyJhbGci..."
}
```

**curl:**

```bash
curl -X POST http://raptor_open_0_4_api.dhtsolution.com:8012/api/0.4/sso/login \
  -d "username=test_basicuser&password=12345"
```

**Python:**

```python
import requests

resp = requests.post(
    "http://raptor_open_0_4_api.dhtsolution.com:8012/api/0.4/sso/login",
    data={"username": "test_basicuser", "password": "12345"},
)
resp.raise_for_status()
token = resp.json()["access_token"]
print("Token:", token[:40], "...")
```

---

### `POST /api/0.4/sso/logout`

Logs out the current session via Keycloak.

> **Note:** This endpoint requires the `realm_name` cookie set at browser login — a plain API call (no cookie) returns `{"detail": "Missing realm_name cookie"}`.  
> A JWT access_token cannot be force-invalidated through the API; it simply expires naturally once its `exp` is reached (typically 5–30 minutes). If you need to revoke a session immediately, call this endpoint from the browser instead (the cookie is sent automatically).

**Headers:** `Authorization: Bearer <token>`  
**Cookie:** `realm_name=dhtsolution` (set automatically after browser login)

**curl (with cookie):**

```bash
curl -X POST http://raptor_open_0_4_api.dhtsolution.com:8012/api/0.4/sso/logout \
  -H "Authorization: Bearer <access_token>" \
  -b "realm_name=dhtsolution"
```

**Python (with cookie):**

```python
import requests

headers = {"Authorization": f"Bearer {token}"}
resp = requests.post(
    "http://raptor_open_0_4_api.dhtsolution.com:8012/api/0.4/sso/logout",
    headers=headers,
    cookies={"realm_name": "dhtsolution"},
)
print(resp.status_code, resp.json())
```

---

<a id="asset"></a>

## Asset Management (Asset)

### `POST /api/0.4/asset/fileupload_analysis`

Uploads a single file and triggers AI analysis processing (asynchronous, via Kafka).  
Supports video (mp4), audio, PDF, image, and other formats.

If the file already exists (same content hash), returns a `skipped` status immediately without reprocessing.

**Headers:** `Authorization: Bearer <token>`  
**Content-Type:** `multipart/form-data`

| Parameter | Type | Required | Description |
|------|------|------|------|
| `primary_file` | file | ✓ | The file to upload |
| `archive_date_or_ttl` | string | | Archive date (ISO 8601) or TTL in days (e.g. `"30"`) |
| `destroy_date_or_ttl` | string | | Deletion date or TTL in days |
| `processing_mode` | string | | Processing mode, defaults to `"default"` |

**Response (new file):**

```json
{
  "upload_result": {
    "asset_path": "video/mp4/習近平喊中美_應當夥伴__川普曝遇困難_互相致電__TVBS新聞_TVBSNEWS01_1080p",
    "version_id": "93d3389e480958dfcfdf7a89ba9c270126cb6a39042fae9ba55d1c503d895b63",
    "primary_filename": "習近平喊中美_應當夥伴__川普曝遇困難_互相致電__TVBS新聞_TVBSNEWS01_1080p.mp4",
    "associated_filenames": [],
    "upload_date": "2026-05-18T22:56:19.957014+08:00",
    "archive_date": null,
    "destroy_date": null,
    "user": "c9d6b4ab-ad57-488b-987a-141f59fda512",
    "branch": "c9d6b4ab-ad57-488b-987a-141f59fda512",
    "status": "active",
    "checksum": "5ebc8e9b53b294f58346c2c5de1e7ecd",
    "existence_info": {
      "exists": false,
      "message": "New asset content",
      "existing_asset_path": null,
      "existing_version_id": null
    }
  },
  "processing_result": {
    "message_id": "672a90a9-6029-4530-92cb-7f16fbce99af",
    "correlation_id": "41644469-7492-44b6-8ae5-4af3c343c035"
  }
}
```

**Response (duplicate file):**

```json
{
  "upload_result": {
    "asset_path": "video/mp4/習近平喊中美_應當夥伴__川普曝遇困難_互相致電__TVBS新聞_TVBSNEWS01_1080p",
    "version_id": "93d3389e480958dfcfdf7a89ba9c270126cb6a39042fae9ba55d1c503d895b63",
    "primary_filename": "習近平喊中美_應當夥伴__川普曝遇困難_互相致電__TVBS新聞_TVBSNEWS01_1080p.mp4",
    "checksum": "5ebc8e9b53b294f58346c2c5de1e7ecd",
    "existence_info": {
      "exists": true,
      "message": "Same content already exists at video/mp4/... with version: 93d3389e..., reuse it and skipping upload.",
      "existing_asset_path": "video/mp4/習近平喊中美_應當夥伴__川普曝遇困難_互相致電__TVBS新聞_TVBSNEWS01_1080p",
      "existing_version_id": "93d3389e480958dfcfdf7a89ba9c270126cb6a39042fae9ba55d1c503d895b63"
    }
  },
  "processing_result": {
    "status": "skipped",
    "reason": "duplicate_file",
    "message": "Same content already exists at ... reuse it and skipping upload."
  }
}
```

**curl:**

```bash
curl -X POST http://raptor_open_0_4_api.dhtsolution.com:8012/api/0.4/asset/fileupload_analysis \
  -H "Authorization: Bearer <access_token>" \
  -F "primary_file=@/path/to/video.mp4" \
  -F "processing_mode=default"
```

**Python:**

```python
import requests

headers = {"Authorization": f"Bearer {token}"}
with open("video.mp4", "rb") as f:
    resp = requests.post(
        "http://raptor_open_0_4_api.dhtsolution.com:8012/api/0.4/asset/fileupload_analysis",
        headers=headers,
        files={"primary_file": f},
        data={"processing_mode": "default"},
    )
resp.raise_for_status()
result = resp.json()
print("version_id:", result["upload_result"]["version_id"])
processing = result["processing_result"]
if "correlation_id" in processing:
    # New file: grab correlation_id to check processing status afterwards
    correlation_id = processing["correlation_id"]
    print("correlation_id:", correlation_id)
else:
    # Duplicate file
    print("status:", processing["status"])  # "skipped"
```

---

### `POST /api/0.4/asset/fileupload_analysis_batch`

Batch-uploads multiple files and triggers AI analysis for each.

**Headers:** `Authorization: Bearer <token>`  
**Content-Type:** `multipart/form-data`

| Parameter | Type | Required | Description |
|------|------|------|------|
| `primary_files` | file[] | ✓ | Multiple files |
| `archive_date_or_ttl` | string | | Archive setting |
| `destroy_date_or_ttl` | string | | Deletion setting |
| `processing_mode` | string | | Processing mode, defaults to `"default"` |
| `concurrency` | int | | Concurrent uploads, defaults to `4`, max `8` |

**Response:**

```json
{
  "upload_summary": {
    "total": 2,
    "success_count": 2,
    "failure_count": 0
  },
  "upload_successes": [
    {
      "filename": "video1.mp4",
      "result": {"version_id": "...", "asset_path": "..."}
    }
  ],
  "upload_failures": [],
  "processing_results": [
    {
      "filename": "video1.mp4",
      "status": "kafka_sent",
      "kafka_result": {}
    }
  ]
}
```

**curl:**

```bash
curl -X POST http://raptor_open_0_4_api.dhtsolution.com:8012/api/0.4/asset/fileupload_analysis_batch \
  -H "Authorization: Bearer <access_token>" \
  -F "primary_files=@/path/to/video1.mp4" \
  -F "primary_files=@/path/to/video2.mp4" \
  -F "processing_mode=default" \
  -F "concurrency=4"
```

**Python:**

```python
import requests

headers = {"Authorization": f"Bearer {token}"}
files = [
    ("primary_files", open("video1.mp4", "rb")),
    ("primary_files", open("video2.mp4", "rb")),
]
resp = requests.post(
    "http://raptor_open_0_4_api.dhtsolution.com:8012/api/0.4/asset/fileupload_analysis_batch",
    headers=headers,
    files=files,
    data={"processing_mode": "default", "concurrency": "4"},
)
resp.raise_for_status()
result = resp.json()
print("succeeded:", result["upload_summary"]["success_count"])
print("failed:", result["upload_summary"]["failure_count"])
```

---

### `GET /api/0.4/asset/filedownload/{asset_path}/{version_id}`

Retrieves download info for a specific asset version, including a presigned URL (valid 24 hours).

For videos, also returns presigned URLs for all associated frame images.

**Headers:** `Authorization: Bearer <token>`  
**Path params:**

| Parameter | Description |
|------|------|
| `asset_path` | Asset path, e.g. `video/mp4/some_video_title_1080p` |
| `version_id` | Version ID (64 hex chars) |

**Query params:**

| Parameter | Type | Description |
|------|------|------|
| `return_file_content` | bool | When `true`, returns raw binary content directly (default `false`) |

**Response:**

```json
{
  "metadata": {
    "asset_path": "video/mp4/川普參觀天壇_習近平陪同_記者連三問是否談台灣_已讀不回__shorts__鏡新聞_1080p",
    "version_id": "109fd79e263ae402334255f27660878b5ecc9eafe6a110e6a960c9f462d9cc37",
    "primary_filename": "川普參觀天壇_習近平陪同_記者連三問是否談台灣_已讀不回__shorts__鏡新聞_1080p.mp4",
    "associated_filenames": [["frame_0001.jpg", "<version_id>"], ["frame_0002.jpg", "<version_id>"]],
    "upload_date": "2026-05-18T21:52:45.843945+08:00",
    "archive_date": null,
    "destroy_date": null,
    "status": "active"
  },
  "primary_file": {
    "filename": "川普參觀天壇_習近平陪同_記者連三問是否談台灣_已讀不回__shorts__鏡新聞_1080p.mp4",
    "content_type": null,
    "version_id": "109fd79e263ae402334255f27660878b5ecc9eafe6a110e6a960c9f462d9cc37",
    "url": "http://raptor_open_0_4_api.dhtsolution.com:8333/lakefs/data/..."
  },
  "associated_file_0": {
    "filename": "frame_0001.jpg",
    "url": "http://raptor_open_0_4_api.dhtsolution.com:8333/lakefs/data/..."
  }
}
```

**curl:**

```bash
ASSET_PATH="video/mp4/川普參觀天壇_習近平陪同_記者連三問是否談台灣_已讀不回__shorts__鏡新聞_1080p"
VERSION_ID="109fd79e263ae402334255f27660878b5ecc9eafe6a110e6a960c9f462d9cc37"

curl -G "http://raptor_open_0_4_api.dhtsolution.com:8012/api/0.4/asset/filedownload/${ASSET_PATH}/${VERSION_ID}" \
  -H "Authorization: Bearer <access_token>"
```

**Python:**

```python
import requests
from urllib.parse import quote

asset_path = "video/mp4/川普參觀天壇_習近平陪同_記者連三問是否談台灣_已讀不回__shorts__鏡新聞_1080p"
version_id = "109fd79e263ae402334255f27660878b5ecc9eafe6a110e6a960c9f462d9cc37"

headers = {"Authorization": f"Bearer {token}"}
resp = requests.get(
    f"http://raptor_open_0_4_api.dhtsolution.com:8012/api/0.4/asset/filedownload/{asset_path}/{version_id}",
    headers=headers,
)
resp.raise_for_status()
data = resp.json()
print("primary file URL:", data["primary_file"]["url"])
# list all associated frames
for key, val in data.items():
    if key.startswith("associated_file_"):
        print(f"  {val['filename']}: {val['url'][:60]}...")
```

---

### `GET /api/0.4/asset/fileversions/{asset_path}/{filename}`

Looks up the full version history for an asset file, each version including a presigned URL.

**Headers:** `Authorization: Bearer <token>`  
**Path params:**

| Parameter | Description |
|------|------|
| `asset_path` | Asset path, e.g. `video/mp4/川普參觀天壇_習近平陪同_記者連三問是否談台灣_已讀不回__shorts__鏡新聞_1080p` |
| `filename` | Primary filename, e.g. `川普參觀天壇_習近平陪同_記者連三問是否談台灣_已讀不回__shorts__鏡新聞_1080p.mp4` |

**Response:**

```json
{
  "versions": [
    {
      "key": "video/mp4/川普參觀天壇_.../川普參觀天壇_...mp4",
      "version_id": "109fd79e263ae402334255f27660878b5ecc9eafe6a110e6a960c9f462d9cc37",
      "upload_date": "2026-05-18T21:52:45.843945+08:00",
      "status": "active",
      "url": "http://raptor_open_0_4_api.dhtsolution.com:8333/lakefs/data/..."
    }
  ]
}
```

**curl:**

```bash
ASSET_PATH="video/mp4/川普參觀天壇_習近平陪同_記者連三問是否談台灣_已讀不回__shorts__鏡新聞_1080p"
FILENAME="川普參觀天壇_習近平陪同_記者連三問是否談台灣_已讀不回__shorts__鏡新聞_1080p.mp4"

curl "http://raptor_open_0_4_api.dhtsolution.com:8012/api/0.4/asset/fileversions/${ASSET_PATH}/${FILENAME}" \
  -H "Authorization: Bearer <access_token>"
```

**Python:**

```python
import requests

asset_path = "video/mp4/川普參觀天壇_習近平陪同_記者連三問是否談台灣_已讀不回__shorts__鏡新聞_1080p"
filename = "川普參觀天壇_習近平陪同_記者連三問是否談台灣_已讀不回__shorts__鏡新聞_1080p.mp4"

headers = {"Authorization": f"Bearer {token}"}
resp = requests.get(
    f"http://raptor_open_0_4_api.dhtsolution.com:8012/api/0.4/asset/fileversions/{asset_path}/{filename}",
    headers=headers,
)
resp.raise_for_status()
versions = resp.json()["versions"]
for v in versions:
    print(f"version_id: {v['version_id'][:16]}... | status: {v['status']} | date: {v['upload_date']}")
```

---

### `GET /api/0.4/asset/users/commits`

Lists all assets uploaded by the current user, with keyword search and date filtering.

**Headers:** `Authorization: Bearer <token>`  
**Query params:**

| Parameter | Type | Description |
|------|------|------|
| `keyword` | string | Keyword search over filenames |
| `start_date` | string | Start date (ISO 8601) |
| `end_date` | string | End date (ISO 8601) |
| `page` | int | Page number, defaults to `1` |
| `page_size` | int | Items per page, defaults to `10` |

**Response:**

```json
{
  "total_count": 12,
  "total_pages": 2,
  "page": 1,
  "page_size": 10,
  "commits": [
    {
      "asset_path": "video/mp4/川普強調美國AI領先中國_曝與習近平會晤日期_民視新聞__1080p",
      "version_id": "d9243fa19b59c4daf793d77ebe31ba91433b2826d8185e4657be2cd8fd3a52d2",
      "primary_filename": "川普強調美國AI領先中國_曝與習近平會晤日期_民視新聞__1080p.mp4",
      "associated_filenames": [["frame_0001.jpg", "<vid>"], ["frame_0002.jpg", "<vid>"]],
      "upload_date": "2026-05-18T21:52:45.843945+08:00",
      "archive_date": null,
      "destroy_date": null,
      "status": "active"
    }
  ]
}
```

**curl:**

```bash
# search for assets whose filename contains a keyword
curl "http://raptor_open_0_4_api.dhtsolution.com:8012/api/0.4/asset/users/commits?keyword=trump&page=1&page_size=5" \
  -H "Authorization: Bearer <access_token>"
```

**Python:**

```python
import requests

headers = {"Authorization": f"Bearer {token}"}
resp = requests.get(
    "http://raptor_open_0_4_api.dhtsolution.com:8012/api/0.4/asset/users/commits",
    headers=headers,
    params={"keyword": "trump", "page": 1, "page_size": 5},
)
resp.raise_for_status()
data = resp.json()
print(f"{data['total_count']} total, page {data['page']}/{data['total_pages']}")
for c in data["commits"]:
    print(f"  {c['primary_filename']} | {c['status']}")
```

---

### `POST /api/0.4/asset/filearchive/{asset_path}/{version_id}`

Marks a specific version as archived. An archived asset is still accessible but no longer appears in the main listing.

**Headers:** `Authorization: Bearer <token>`

**curl:**

```bash
curl -X POST \
  "http://raptor_open_0_4_api.dhtsolution.com:8012/api/0.4/asset/filearchive/video%2Fmp4%2F川普啟動自由計劃/b02a0b42ba37f06cafeedd55e953762dc42988e9aa5163f877a6c170f08d0ebb" \
  -H "Authorization: Bearer <access_token>"
```

**Python:**

```python
import requests

asset_path = "video/mp4/川普啟動自由計劃"
version_id = "b02a0b42ba37f06cafeedd55e953762dc42988e9aa5163f877a6c170f08d0ebb"

headers = {"Authorization": f"Bearer {token}"}
resp = requests.post(
    f"http://raptor_open_0_4_api.dhtsolution.com:8012/api/0.4/asset/filearchive/{asset_path}/{version_id}",
    headers=headers,
)
resp.raise_for_status()
print(resp.json())
```

---

### `POST /api/0.4/asset/delfile/{asset_path}/{version_id}`

Deletes a specific asset version (removed from LakeFS). This operation is irreversible.

**Headers:** `Authorization: Bearer <token>`

**curl:**

```bash
curl -X POST \
  "http://raptor_open_0_4_api.dhtsolution.com:8012/api/0.4/asset/delfile/video%2Fmp4%2F川普啟動自由計劃/b02a0b42ba37f06cafeedd55e953762dc42988e9aa5163f877a6c170f08d0ebb" \
  -H "Authorization: Bearer <access_token>"
```

**Python:**

```python
import requests

asset_path = "video/mp4/川普啟動自由計劃"
version_id = "b02a0b42ba37f06cafeedd55e953762dc42988e9aa5163f877a6c170f08d0ebb"

headers = {"Authorization": f"Bearer {token}"}
resp = requests.post(
    f"http://raptor_open_0_4_api.dhtsolution.com:8012/api/0.4/asset/delfile/{asset_path}/{version_id}",
    headers=headers,
)
resp.raise_for_status()
print(resp.json())
```

---

### `POST /api/0.4/asset/file-expiration/{asset_path}/{version_id}`

Updates the archive date or deletion date for an asset. Accepts either an ISO 8601 date string or a number of days (TTL).

> **Note:** If the asset had no expiration set at upload time (a permanent asset), `archive_date_or_ttl` and `destroy_date_or_ttl` must **both be provided**, otherwise this returns `400`.

**Headers:** `Authorization: Bearer <token>`  
**Content-Type:** `application/x-www-form-urlencoded`

| Parameter | Type | Description |
|------|------|------|
| `archive_date_or_ttl` | int \| string | Archive date (`"2026-12-31"`) or days (`30`) |
| `destroy_date_or_ttl` | int \| string | Deletion date or days |

**curl:**

```bash
curl -X POST \
  "http://raptor_open_0_4_api.dhtsolution.com:8012/api/0.4/asset/file-expiration/video%2Fmp4%2F川普啟動自由計劃/b02a0b42ba37f06cafeedd55e953762dc42988e9aa5163f877a6c170f08d0ebb" \
  -H "Authorization: Bearer <access_token>" \
  -d "archive_date_or_ttl=90&destroy_date_or_ttl=365"
```

**Python:**

```python
import requests

asset_path = "video/mp4/川普啟動自由計劃"
version_id = "b02a0b42ba37f06cafeedd55e953762dc42988e9aa5163f877a6c170f08d0ebb"

headers = {"Authorization": f"Bearer {token}"}
resp = requests.post(
    f"http://raptor_open_0_4_api.dhtsolution.com:8012/api/0.4/asset/file-expiration/{asset_path}/{version_id}",
    headers=headers,
    data={"archive_date_or_ttl": "90", "destroy_date_or_ttl": "365"},
)
resp.raise_for_status()
print(resp.json())
```

---

<a id="search"></a>

## Search

Every search endpoint requires authentication. `hybrid`/`bm25`/`vector` return `{"results": [...], "timing": {...}}`; `tkg`/`graphrag` have a different response shape (see their own sections) with no `results` array. All of these endpoints query a per-user isolated index (Module 25, ArcadeDB-backed) — not a site-wide shared index.

`asset_url` inside each result's `payload` is a presigned download link for the asset (valid 24 hours; the actual signature string is omitted from the examples below).

### Shared request schema (`hybrid`/`bm25`/`vector`)

| Field | Type | Default | Description |
|------|------|------|------|
| `query` | string | required | Search text |
| `top_k` | int | `10` | Maximum number of results |
| `embedding_type` | string | null | `"text"` or `"summary"` |
| `type` | string \| string[] | null | Asset type filter, e.g. `"videos"`, `["documents","audios"]` |
| `filename` | string[] | null | Filter by filename |
| `speaker` | string[] | null | Filter by speaker (video/audio) |
| `source` | string | null | Filter by source format, e.g. `"pdf"`, `"mp4"` |

---

### `POST /api/0.4/search/hybrid`

Hybrid search (BM25 + vector fusion), results ranked with RRF then reranked.  
Suitable for most general search scenarios.

**curl:**

```bash
curl -X POST http://raptor_open_0_4_api.dhtsolution.com:8012/api/0.4/search/hybrid \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "川習會 貿易談判",
    "top_k": 5
  }'
```

**curl (with filters):**

```bash
curl -X POST http://raptor_open_0_4_api.dhtsolution.com:8012/api/0.4/search/hybrid \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "川普習近平 關稅",
    "top_k": 5,
    "type": "videos",
    "filename": ["川普強調美國AI領先中國_曝與習近平會晤日期_民視新聞__1080p.mp4"]
  }'
```

**Python:**

```python
import requests

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
}
resp = requests.post(
    "http://raptor_open_0_4_api.dhtsolution.com:8012/api/0.4/search/hybrid",
    headers=headers,
    json={"query": "川習會 貿易談判", "top_k": 5},
)
resp.raise_for_status()
results = resp.json()["results"]
for r in results:
    p = r["payload"]
    print(f"score={r['score']:.4f} | file={p.get('filename','?')} | type={p.get('type','?')}")
    if p.get("asset_url"):
        print(f"  URL: {p['asset_url'][:80]}...")
```

**Example response:**

```json
{
  "results": [
    {
      "id": "79626f7d-437e-4137-8a5d-3b0805172a18",
      "score": 0.500029,
      "payload": {
        "version_id": "79626f7d-437e-4137-8a5d-3b0805172a18",
        "summary": "(AI-generated overall summary)...",
        "filename": "直球對決__習近平_台獨和台海和平水火不容__TVBS新聞__TVBSNEWS01_1080p.mp4",
        "asset_path": "video/mp4/直球對決__習近平_台獨和台海和平水火不容__TVBS新聞__TVBSNEWS01_1080p",
        "type": "videos",
        "embedding_type": "summary",
        "asset_url": "http://<host>:8333/lakefs/data/...?X-Amz-Signature=...(presigned, 24h)"
      }
    },
    {
      "id": "2386aa1de7a951bcbd56950aef4efa289bbd4cad91b1870761558b10c374e536",
      "score": 0.500025,
      "payload": {
        "chunk_id": "2386aa1de7a951bcbd56950aef4efa289bbd4cad91b1870761558b10c374e536",
        "type": "videos",
        "embedding_type": "text",
        "text": "ocr:{...} / asr:{...} / lvlm:{...}",
        "summary": null,
        "filename": "直球對決__習近平_台獨和台海和平水火不容__TVBS新聞__TVBSNEWS01_1080p.mp4",
        "source": "mp4",
        "asset_path": "video/mp4/直球對決__習近平_台獨和台海和平水火不容__TVBS新聞__TVBSNEWS01_1080p",
        "version_id": "79626f7d-437e-4137-8a5d-3b0805172a18",
        "status": "active",
        "speaker": "SPEAKER_02",
        "chunk_index": 3,
        "start_time": "30.0",
        "end_time": "40.0",
        "asset_url": "http://<host>:8333/lakefs/data/...?X-Amz-Signature=...(presigned, 24h)"
      }
    }
  ],
  "timing": {
    "total_sec": 0.0761,
    "embed_sec": 0.0192,
    "fusion_sec": 0.0118,
    "rerank_sec": 0.0406
  }
}
```

**Note:** the fields inside `payload` vary by `embedding_type` — `"summary"`-type results don't have `chunk_index`/`start_time`/`end_time`/`speaker`/`status`; only `"text"`-type results carry that per-segment detail. Neither type has a `branch_id` field (an earlier version of this doc mistakenly listed one). `score` is the fused-then-reranked score, empirically clustering around 0.5 — it is not a 0–1 similarity.

---

### `POST /api/0.4/search/bm25`

Pure BM25 (keyword) search, over the per-user isolated ArcadeDB index (Module 25).  
Good for exact keyword matching, no semantic vectors involved. Score is the BM25 score (higher = more relevant).

**curl:**

```bash
curl -X POST http://raptor_open_0_4_api.dhtsolution.com:8012/api/0.4/search/bm25 \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "川普 貿易關稅",
    "top_k": 5
  }'
```

**Python:**

```python
import requests

headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
resp = requests.post(
    "http://raptor_open_0_4_api.dhtsolution.com:8012/api/0.4/search/bm25",
    headers=headers,
    json={"query": "川普 貿易關稅", "top_k": 5},
)
resp.raise_for_status()
for r in resp.json()["results"]:
    print(f"BM25 score={r['score']:.2f} | {r['payload'].get('filename','?')}")
```

**Example response:**

```json
{
  "results": [
    {
      "id": "79626f7d-437e-4137-8a5d-3b0805172a18",
      "score": 0.7911,
      "payload": {
        "version_id": "79626f7d-437e-4137-8a5d-3b0805172a18",
        "summary": "(AI-generated overall summary)...",
        "filename": "直球對決__習近平_台獨和台海和平水火不容__TVBS新聞__TVBSNEWS01_1080p.mp4",
        "asset_path": "video/mp4/直球對決__習近平_台獨和台海和平水火不容__TVBS新聞__TVBSNEWS01_1080p",
        "type": "videos",
        "embedding_type": "summary",
        "asset_url": "http://<host>:8333/lakefs/data/...?X-Amz-Signature=...(presigned, 24h)"
      }
    }
  ],
  "timing": {
    "total_sec": 0.0149,
    "bm25_sec": 0.0092
  }
}
```

When there are no matches, `results` is an empty array (`[]`), not a 404.

---

### `POST /api/0.4/search/vector`

Pure vector (semantic) search, over the per-user isolated ArcadeDB index (Module 25).  
Good for semantic similarity search; score is cosine similarity (0–1).

**curl:**

```bash
curl -X POST http://raptor_open_0_4_api.dhtsolution.com:8012/api/0.4/search/vector \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "川普與習近平 AI 科技競爭",
    "top_k": 5
  }'
```

**Python:**

```python
import requests

headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
resp = requests.post(
    "http://raptor_open_0_4_api.dhtsolution.com:8012/api/0.4/search/vector",
    headers=headers,
    json={"query": "川普與習近平 AI 科技競爭", "top_k": 5},
)
resp.raise_for_status()
for r in resp.json()["results"]:
    print(f"cosine={r['score']:.4f} | {r['payload'].get('filename','?')}")
```

**Example response:**

```json
{
  "results": [
    {
      "id": "2386aa1de7a951bcbd56950aef4efa289bbd4cad91b1870761558b10c374e536",
      "score": 0.4066,
      "payload": {
        "chunk_id": "2386aa1de7a951bcbd56950aef4efa289bbd4cad91b1870761558b10c374e536",
        "type": "videos",
        "embedding_type": "text",
        "text": "ocr:{...} / asr:{...} / lvlm:{...}",
        "summary": null,
        "filename": "直球對決__習近平_台獨和台海和平水火不容__TVBS新聞__TVBSNEWS01_1080p.mp4",
        "source": "mp4",
        "asset_path": "video/mp4/直球對決__習近平_台獨和台海和平水火不容__TVBS新聞__TVBSNEWS01_1080p",
        "version_id": "79626f7d-437e-4137-8a5d-3b0805172a18",
        "status": "active",
        "speaker": "SPEAKER_02",
        "chunk_index": 5,
        "start_time": "50.0",
        "end_time": "52.34",
        "asset_url": "http://<host>:8333/lakefs/data/...?X-Amz-Signature=...(presigned, 24h)"
      }
    }
  ],
  "timing": {
    "total_sec": 0.0421
  }
}
```

---

### `POST /api/0.4/search/graphrag`

GraphRAG knowledge-graph search — finds related relationships via entity nodes plus subgraph expansion.  
Good for queries like "who is X" or "what's the relationship between X and Y" — entity/relationship queries.

**Request schema:**

| Field | Type | Default | Description |
|------|------|------|------|
| `query` | string | required | Natural-language query |
| `max_depth` | int | `2` | Subgraph expansion depth (1–4) |
| `limit` | int | `50` | Max nodes returned (1–200) |
| `score_threshold` | float | `0.5` | Minimum score filter (0–10) |
| `strategy` | string | `"hybrid"` | Search strategy: `"hybrid"` / `"literal"` / `"semantic"` |

**curl:**

```bash
curl -X POST http://raptor_open_0_4_api.dhtsolution.com:8012/api/0.4/search/graphrag \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "川普習近平關係",
    "max_depth": 2,
    "limit": 20,
    "strategy": "hybrid"
  }'
```

**Python:**

```python
import requests

headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
resp = requests.post(
    "http://raptor_open_0_4_api.dhtsolution.com:8012/api/0.4/search/graphrag",
    headers=headers,
    json={
        "query": "川普習近平關係",
        "max_depth": 2,
        "limit": 20,
        "strategy": "hybrid",
    },
)
resp.raise_for_status()
data = resp.json()
print("Query:", data.get("query"))
for e in data.get("matched_entities", [])[:5]:
    print(f"  [{e['type']}] {e['name']} (score={e['score']:.2f}): {e.get('description','')[:60]}")
```

**Example response:**

```json
{
  "query": "川普習近平關係",
  "strategy": "hybrid",
  "matched_entities": [
    {
      "id": "xi_jinping",
      "name": "習近平",
      "type": "PERSON",
      "description": "中華人民共和國最高領導人",
      "node_kind": "entity",
      "score": 4.21
    },
    {
      "id": "trump_visit_china",
      "name": "川普訪華",
      "type": "EVENT",
      "description": "川普總統訪問中國的外交事件",
      "node_kind": "entity",
      "score": 3.85
    }
  ],
  "matched_moments": [...],
  "nodes": [...],
  "relationships": [...],
  "moment_ids": [...],
  "citations": [...]
}
```

---

### `POST /api/0.4/search/tkg`

Temporal Knowledge Graph search — queries event relationships with a time ordering.  
Good for queries like "did X happen before/after Y" or "timeline of events" — temporal queries.

**Request schema:**

| Field | Type | Default | Description |
|------|------|------|------|
| `query` | string | required | Natural-language query |
| `time_start` | string | null | Start of time range (ISO 8601), e.g. `"2025-01-01"` |
| `time_end` | string | null | End of time range |
| `max_depth` | int | `2` | Subgraph expansion depth (1–4) |
| `limit` | int | `50` | Max nodes returned (1–200) |
| `score_threshold` | float | `0.3` | Minimum score filter (0–10) |

**curl:**

```bash
curl -X POST http://raptor_open_0_4_api.dhtsolution.com:8012/api/0.4/search/tkg \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "川習峰會時間序列",
    "max_depth": 2,
    "limit": 20
  }'
```

**curl (with a time range):**

```bash
curl -X POST http://raptor_open_0_4_api.dhtsolution.com:8012/api/0.4/search/tkg \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "中美貿易談判",
    "time_start": "2025-01-01",
    "time_end": "2026-12-31",
    "max_depth": 2
  }'
```

**Python:**

```python
import requests

headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
resp = requests.post(
    "http://raptor_open_0_4_api.dhtsolution.com:8012/api/0.4/search/tkg",
    headers=headers,
    json={"query": "川習峰會時間序列", "max_depth": 2, "limit": 20},
)
resp.raise_for_status()
data = resp.json()
for e in data.get("matched_entities", [])[:5]:
    print(f"  [{e['type']}] {e['name']} | {e.get('description','')[:60]}")
print("temporal_facts:", len(data.get("temporal_facts", [])))
```

**Example response:**

```json
{
  "query": "川習峰會時間序列",
  "matched_entities": [
    {
      "id": "trump_visit_china",
      "name": "川普訪華",
      "type": "EVENT",
      "description": "外交事件",
      "score": 3.85
    }
  ],
  "subgraph_nodes": [...],
  "subgraph_edges": [...],
  "temporal_facts": [...],
  "moment_ids": [...]
}
```

---

<a id="video-search"></a>

## Video Search

### `POST /api/0.4/search/video_search`

A search endpoint purpose-built for video. First does multi-recall retrieval (BM25 + Vector + GraphRAG + TKG) to gather candidate chunks, fuses them via RRF, reranks with a cross-encoder, then aggregates by video, returning each video with its list of most-relevant time segments.

**Request schema:**

| Field | Type | Default | Description |
|------|------|------|------|
| `query` | string | required | Natural-language search |
| `top_k` | int | `10` | Max number of videos returned (1–50) |
| `candidate_multiplier` | int | `5` | Each recaller takes `top_k × N` candidate chunks (1–20) |
| `score_threshold` | float | `0.52` | Minimum score filter (0–1); chunks below this are dropped |

**curl:**

```bash
curl -X POST http://raptor_open_0_4_api.dhtsolution.com:8012/api/0.4/search/video_search \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "川普習近平",
    "top_k": 5
  }'
```

**Python:**

```python
import requests

headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
resp = requests.post(
    "http://raptor_open_0_4_api.dhtsolution.com:8012/api/0.4/search/video_search",
    headers=headers,
    json={"query": "川普習近平", "top_k": 5},
)
resp.raise_for_status()
data = resp.json()
print(f"found {data['total']} videos")
for video in data["results"]:
    print(f"\n[{video['score']:.4f}] {video['filename']}")
    print(f"  download: {video['asset_url'][:80]}...")
    for seg in video.get("segments", [])[:2]:
        print(f"  segment {seg['start_time']}s-{seg['end_time']}s (score={seg['score']:.4f})")
        print(f"    source: {seg['sources']}")
```

**Example response:**

```json
{
  "query": "川普習近平",
  "total": 3,
  "results": [
    {
      "video_id": "d9243fa19b59c4daf793d77ebe31ba91433b2826d8185e4657be2cd8fd3a52d2",
      "filename": "川普強調美國AI領先中國_曝與習近平會晤日期_民視新聞__1080p.mp4",
      "score": 0.7102,
      "asset_url": "http://raptor_open_0_4_api.dhtsolution.com:8333/lakefs/data/...",
      "upload_time": "2026-05-18T...",
      "segments": [
        {
          "start_time": 0.0,
          "end_time": 10.0,
          "score": 0.7102,
          "text": "contextual:{...} / ocr:{...} / asr:{...} / lvlm:{...}",
          "sources": ["bm25", "vector"]
        }
      ]
    }
  ]
}
```

---

<a id="rag"></a>

## RAG Query (A2A Protocol)

### `POST /api/0.4/a2a/query`

Full RAG pipeline: intent classification → multi-path search → rerank → LLM-generated answer.  
Returns the LLM-generated answer, source chunks, and graph context.

Supports three execution modes:
- `direct`: deterministic pipeline (Module 18 classification → vector/keyword/graph search → rerank → Ollama LLM)
- `agent`: smolagents CodeAgent mode (the LLM autonomously decides which tools to call)
- `tool`: smolagents ToolCallingAgent mode

> **Note:** this endpoint invokes LLM generation and typically responds in 15–60 seconds. Use a longer timeout.

**Request schema:**

| Field | Type | Default | Description |
|------|------|------|------|
| `question` | string | required | The question (at least 1 character) |
| `top_k` | int | `5` | Number of search results used (1–50) |
| `mode` | string | `"direct"` | `"direct"` / `"agent"` / `"tool"` |

**curl:**

```bash
curl -X POST http://raptor_open_0_4_api.dhtsolution.com:8012/api/0.4/a2a/query \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "川普和習近平在天壇見面時，記者問了什麼問題？",
    "top_k": 5,
    "mode": "direct"
  }'
```

**Python:**

```python
import requests

headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
resp = requests.post(
    "http://raptor_open_0_4_api.dhtsolution.com:8012/api/0.4/a2a/query",
    headers=headers,
    json={
        "question": "川普和習近平在天壇見面時，記者問了什麼問題？",
        "top_k": 5,
        "mode": "direct",
    },
    timeout=120,  # LLM inference needs a longer timeout
)
resp.raise_for_status()
data = resp.json()
print("Answer:", data["answer"])
print(f"cited {data['chunks_used']} sources:")
for src in data.get("sources", [])[:3]:
    filename = src.get("metadata", {}).get("filename", "?")
    print(f"  score={src['score']:.4f} | {filename}")
    if src.get("storage_uri"):
        print(f"  URL: {src['storage_uri'][:80]}...")
```

**Example response:**

```json
{
  "answer": "根據影片記錄，川普和習近平在天壇參觀時，記者連續三次詢問川普是否在會談中提及台灣問題。川普僅簡短回應「很棒」、「地方很讚」及「中國很美」，對台灣問題保持沉默未予正面回答，引發外界關注。",
  "sources": [
    {
      "id": "7b3e24ec-ea2e-4758-a528-9f7a25df187f",
      "doc_id": "7b3e24ec-ea2e-4758-a528-9f7a25df187f",
      "score": 0.7102,
      "content": "contextual:{記者首次詢問川普會談狀況，川普簡短回應並讚美天壇環境。} / ...",
      "metadata": {
        "filename": "川普參觀天壇_習近平陪同_記者連三問是否談台灣_已讀不回__shorts__鏡新聞_1080p.mp4",
        "type": "videos"
      },
      "start_time": 0.0,
      "end_time": 10.0,
      "asset_path": "video/mp4/川普參觀天壇_...",
      "version_id": "109fd79e...",
      "storage_uri": "http://raptor_open_0_4_api.dhtsolution.com:8333/lakefs/data/..."
    }
  ],
  "graph_context": "",
  "chunks_used": 3
}
```

---

<a id="chat"></a>

## Chat

Proxy for Module 15 (chat-service) — multi-turn conversation with automatic RAG search, memory kept in Redis.

### `POST /api/0.4/chat/chat`

Sends one message; the service automatically decides whether RAG search is needed and generates a reply via a LangGraph pipeline.

**Pipeline:** load_memory → prepare_context → call_model (may call tools) → execute_tools (if applicable) → save_memory.

**Keywords that auto-trigger search:** `search / find / pdf / document / image / video / audio` (Chinese equivalents too), or a message ending in `?`/`？`.

**Request schema:**

| Field | Type | Default | Description |
|------|------|------|------|
| `message` | string | required | This turn's user message |
| `history` | `[{role, content}]` | null | Conversation history prior to this turn; omit to rely solely on memory already stored in Redis |
| `session_id` | string | null | Isolates memory across different conversation contexts; omit to use the default session |

`user_id` is always taken automatically from the Bearer token — it cannot (and should not) be passed manually.

**curl:**

```bash
curl -X POST http://raptor_open_0_4_api.dhtsolution.com:8012/api/0.4/chat/chat \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"message": "你好，簡單自我介紹一下你能做什麼"}'
```

**Example response (live-tested):**

```json
{
  "response": "您好！我是有幫助的 AI 助手！我主要可以協助您以下事情：\n\n💡 搜尋相關資料 - 跨多個資料庫、檔案、影片、音訊和圖片進行統一搜尋...",
  "user_id": "c9d6b4ab-ad57-488b-987a-141f59fda512",
  "session_id": null,
  "search_triggered": false,
  "search_results": null,
  "tool_calls": []
}
```

When `search_triggered=true`, `search_results` is an array of matched chunks (same shape as the [Search](#search) section), and `tool_calls` lists the tools invoked this turn (`unified_search_tool`, `calculate`, `get_current_time`, etc.).

---

### `GET /api/0.4/chat/memory`

Reads the current user's conversation memory (Module 15's own short-term Redis cache, `chat_memory:*`, which disappears once `MEMORY_TTL` expires).

**This is not** Module 26's (Memory Service) long-term facts/preferences, searchable session history, or multimedia index — for those, use `/memory/retrieve` or `/memory/timeline` in the [Memory Service](#memory) section below instead.

**Query params:** `session_id` (optional; omit to read the default session)

**Example response (live-tested):**

```json
{
  "user_id": "c9d6b4ab-ad57-488b-987a-141f59fda512",
  "session_id": null,
  "memory_count": 1,
  "memories": [
    {
      "timestamp": 1787747671.63,
      "user_message": "你好，簡單自我介紹一下你能做什麼",
      "assistant_response": "您好！我是有幫助的 AI 助手！...",
      "search_results": null
    }
  ]
}
```

---

### `DELETE /api/0.4/chat/memory`

Clears the current user's short-term conversation context (the same Redis cache as above), used to "start a new conversation." `?session_id=xxx` clears only that session; omit to clear the default session. Returns `204 No Content` on success.

**This does not** delete anything stored by Module 26 (long-term facts/preferences, archived session history, multimedia memory) — those don't expire; a true GDPR erasure requires `DELETE /api/0.4/memory` (see below).

---

### `POST /api/0.4/chat/completions`

A Gateway-layer OpenAI-compatible `chat/completions` proxy, forwarding to Module 07's `/v1/chat/completions`. Pure passthrough, no streaming support (`stream` is always forwarded as `false` — when `GuardrailMiddleware` is enabled, it needs the full response content to run its post-generation policy check, which an SSE partial stream can't provide early enough).

**Request:** standard OpenAI `chat/completions` format, requiring at minimum `model` and `messages`; `model` must be a model name already registered in Module 07's MLflow registry (`GET /api/0.4/aiml/models/local` lists what's currently registered).

```bash
curl -X POST http://raptor_open_0_4_api.dhtsolution.com:8012/api/0.4/chat/completions \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"model": "<registered_model_name>", "messages": [{"role": "user", "content": "hello"}]}'
```

**Common error (live-tested, unregistered model, HTTP 404):**

```json
{
  "detail": {"error": {"message": "model '<name>' not in MLflow registry", "type": "invalid_request_error", "code": "model_not_found"}}
}
```

**Calling an unregistered model:** passing `engine` (e.g. `"engine": "ollama"`) bypasses the MLflow registry lookup entirely and calls that runtime's native tag name directly, without needing to register first:

```bash
curl -X POST http://raptor_open_0_4_api.dhtsolution.com:8012/api/0.4/chat/completions \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5:7b", "engine": "ollama", "messages": [{"role": "user", "content": "hello"}]}'
```

**Example response (live-tested, `engine` successfully bypassing an unregistered model):**

```json
{
  "id": "chatcmpl-3a2d40f1ec1d464c8426d5a2",
  "object": "chat.completion",
  "created": 1787797340,
  "model": "qwen2.5:7b",
  "choices": [
    {"index": 0, "message": {"role": "assistant", "content": "我是Qwen，來自阿里雲的人工智慧助手。"}, "finish_reason": "stop"}
  ],
  "usage": {"prompt_tokens": 34, "completion_tokens": 13, "total_tokens": 47}
}
```

---

<a id="memory"></a>

## Memory Service

Proxy for Module 26 (memory-service) — long-term facts/preferences, conversation history searchable across sessions, and multimedia memory. This is a **different store** from the short-term Redis cache in the [Chat](#chat) section above and does not expire along with that conversation cache.

The gateway only exposes 6 flattened routes externally (`store`/`retrieve`/`timeline`/`compact`/`archive` plus the `DELETE` below), plus 2 endpoints reserved for Module 21 (agent-protocol) to call (`compact/evaluate`, `sessions/{id}/summaries`). Most of the originally richer session-scoped interface has been disabled (kept in code, commented out — can be re-enabled later if needed).

### `POST /api/0.4/memory/store`

Writes a single memory node not tied to any session (fact/preference/entity). To record a full conversation turn, use `/memory/archive` instead.

**Request schema:** `text` (required, the content to remember), `frame_type` (`conversation`/`preference`/`entity`/`fact`, default `fact`), `session_id` (optional, for traceability only).

**Example response (live-tested, writing the same content twice):**

```json
{"frame_id": "14", "frame_type": "preference", "timestamp": 1787485472.93, "status": "duplicate", "duplicate_of": "14"}
```

`status` is `"created"` on first write (an idempotent dedup mechanism — writing the same content again for the same user reports `"duplicate"`).

---

### `GET /api/0.4/memory/retrieve`

Hybrid search (semantic + BM25) across all sessions, long-term memory, and multimedia memory.

**Query params:** `query` (required), `top_k` (default `5`, 1–50)

**Example response (live-tested):**

```json
{
  "sessions": [
    {
      "text": "User: hello\nAssistant: 您好！歡迎您～有什麼我可以幫助您的嗎？",
      "score": 0.4803,
      "timestamp": "2026-08-25T05:42:51.81Z",
      "turn_index": 17,
      "user_message": "hello",
      "assistant_response": "您好！歡迎您～有什麼我可以幫助您的嗎？",
      "session_id": "default"
    }
  ]
}
```

---

### `GET /api/0.4/memory/timeline`

Returns the user's conversation turns across all sessions, paginated, in ascending time order (interleaved across sessions). To view a single session only, call Module 26's own `/sessions/{session_id}/timeline` directly instead (not proxied through this gateway).

**Query params:** `page` (default `1`), `page_size` (default `20`, max `100`)

**Example response (live-tested, excerpt):**

```json
{
  "entries": [
    {
      "turn_index": 1,
      "timestamp": "2026-08-21T02:47:57.87Z",
      "frame_type": "session_turn",
      "user_message": "請問有柴犬相關的圖片嗎?",
      "assistant_response": "根據提供的上下文片段和圖譜關係，沒有關於柴犬相關的圖片的資訊。",
      "media_refs": [],
      "tool_calls": [],
      "provider_message_id": null,
      "session_id": "c9d6b4ab-ad57-488b-987a-141f59fda512"
    }
  ]
}
```

---

### `POST /api/0.4/memory/compact`

Triggers compaction for a given session (once the context-window budget is exceeded, older conversation turns are summarized into a single summary frame). `?session_id=xxx` omitted compacts the `default` session.

**Request schema:** `trigger` (`auto`/`manual`/`reactive`, for logging purposes only, default `auto`), `context_window` (default `128000`), `max_tokens` (optional), `last_summarized_frame_id` (optional; omit to auto-locate the session's latest summary frame).

**Example response (live-tested, below the compaction threshold):**

```json
{
  "compacted": false,
  "source": "under_budget",
  "pre_compact_tokens": 4576,
  "post_compact_tokens": 4576,
  "turns_compacted": 0,
  "turns_kept": 21,
  "summary_frame_written": false,
  "threshold_exceeded": false,
  "status": "fallback"
}
```

---

### `POST /api/0.4/memory/archive`

Archives one conversation turn (`user_message` + `assistant_response`) to a given session. `?session_id=xxx` omitted writes to the `default` session.

**Request schema:** `user_message`, `assistant_response` (both required), `search_results`/`tool_calls` (optional arrays), `timestamp` (optional), `provider_message_id` (optional).

**Example response (live-tested):**

```json
{"frame_id": "20", "turn_index": 21, "session_id": "default"}
```

---

### `DELETE /api/0.4/memory`

GDPR erasure: permanently deletes **all** of the current user's memory in Module 26 (sessions, long-term memory, multimedia). Returns `404` if no memory is found. **This operation is irreversible**, and also clears the Redis short-term cache mentioned in the Chat section.

---

### `POST /api/0.4/memory/multimedia/search`

Hybrid search over the current user's video/audio/image index; `media_type` can restrict to a single media type. (Module 21 agent-protocol calls this endpoint through the gateway, which is why it remains part of the external interface.)

**Request schema:** `query` (required), `top_k` (default `5`), `media_type` (optional, `video`/`audio`/`image`)

**Example response (live-tested, no results):** `[]`

---

### `POST /api/0.4/memory/compact/evaluate`

Estimates the compaction budget without actually running compaction. Omitting `session_id` estimates only the size of the `messages` passed in; passing `session_id` aggregates and estimates that session's archived conversation + long-term facts + multimedia chunks together.

**Example response (live-tested, empty session):**

```json
{
  "token_count": 0,
  "context_window": 128000,
  "auto_compact_threshold": 95000,
  "should_compact": false,
  "tokens_over_threshold": 0
}
```

---

### `GET /api/0.4/memory/sessions/{session_id}/summaries`

Lists the summary frames generated so far for a given session (written when `/memory/compact` runs). Returns `[]` when there are none.

---

<a id="processing"></a>

## Processing Status

Once a file is uploaded, AI analysis runs asynchronously in the background. Use these APIs to check processing status.

`correlation_id` comes from the `upload_result.correlation_id` field in `fileupload_analysis`'s response.

---

### `GET /api/0.4/processing/cache/all`

Looks up processing status for all of the current user's upload tasks.  
Only returns data belonging to the current user (same `branch_id`).

**Headers:** `Authorization: Bearer <token>`

**curl:**

```bash
curl http://raptor_open_0_4_api.dhtsolution.com:8012/api/0.4/processing/cache/all \
  -H "Authorization: Bearer <access_token>"
```

**Python:**

```python
import requests

headers = {"Authorization": f"Bearer {token}"}
resp = requests.get(
    "http://raptor_open_0_4_api.dhtsolution.com:8012/api/0.4/processing/cache/all",
    headers=headers,
)
resp.raise_for_status()
data = resp.json()
print(f"{data['count']} processing tasks")
for cache_key, info in data.get("data", {}).items():
    print(f"  {cache_key}: step={info.get('step')}")
```

**Example response:**

```json
{
  "count": 3,
  "data": {
    "video_orchestrator:8d44cfc4-5a59-4be7-bc4d-f0992af5db75": {
      "step": "complete",
      "summary": "影片記錄了川普訪華期間參觀北京天壇的畫面...",
      "text": [...]
    },
    "video_orchestrator:0d297ad5-a636-419e-b6d3-e11f03988b06": {
      "step": "indexing",
      "summary": null
    }
  }
}
```

---

### `GET /api/0.4/processing/cache/{m_type}/{key}`

Looks up detailed processing status for a single upload task.

**Headers:** `Authorization: Bearer <token>`  
**Path params:**

| Parameter | Values | Description |
|------|------|------|
| `m_type` | `document` / `video` / `image` / `audio` | Media type |
| `key` | UUID | The `correlation_id` returned at upload time |

**Processing steps (`step` field):**

| step | Description |
|------|------|
| `queued` | Queued, awaiting processing |
| `transcribing` | Speech-to-text in progress (video/audio) |
| `extracting` | Feature extraction in progress |
| `indexing` | Vectorizing and building the index |
| `complete` | Processing complete |
| `failed` | Processing failed |

**curl:**

```bash
# obtain correlation_id at upload time, then check progress with it
CORRELATION_ID="8d44cfc4-5a59-4be7-bc4d-f0992af5db75"

curl "http://raptor_open_0_4_api.dhtsolution.com:8012/api/0.4/processing/cache/video/${CORRELATION_ID}" \
  -H "Authorization: Bearer <access_token>"
```

**Python (poll until complete):**

```python
import requests
import time

def wait_for_completion(token: str, m_type: str, correlation_id: str, timeout: int = 600) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    url = f"http://raptor_open_0_4_api.dhtsolution.com:8012/api/0.4/processing/cache/{m_type}/{correlation_id}"
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.get(url, headers=headers)
        if resp.status_code == 404:
            print("task not started yet, or has expired")
            time.sleep(5)
            continue
        resp.raise_for_status()
        data = resp.json()
        step = data.get("value", {}).get("step") or data.get("step")
        print(f"  current step: {step}")
        if step in ("complete", "failed"):
            return data
        time.sleep(10)
    raise TimeoutError("timed out waiting for completion")

# usage example
result = wait_for_completion(token, "video", "8d44cfc4-5a59-4be7-bc4d-f0992af5db75")
print("summary:", result.get("value", {}).get("summary", "")[:100])
```

**Example response:**

```json
{
  "key": "video_orchestrator:8d44cfc4-5a59-4be7-bc4d-f0992af5db75",
  "original_key": "8d44cfc4-5a59-4be7-bc4d-f0992af5db75",
  "m_type": "video",
  "value": {
    "step": "complete",
    "summary": "影片記錄了川普訪華期間參觀北京天壇的畫面，期間有記者上前提問。畫面顯示川普與習近平等人站在天壇石階旁，背景為天壇古建築。記者連續三次詢問川普總統在會談中是否提及台灣，川普僅簡短回應「很棒」、「地方很讚」及「中國很美」，對台灣問題則保持沉默未予置答。",
    "text": [
      {
        "id": "7b3e24ec-ea2e-4758-a528-9f7a25df187f",
        "payload": {
          "video_id": "川普參觀天壇_...mp4_moment_0",
          "filename": "川普參觀天壇_習近平陪同_記者連三問是否談台灣_已讀不回__shorts__鏡新聞_1080p.mp4",
          "chunk_index": 0,
          "start_time": "0.0",
          "end_time": "10.0",
          "text": "contextual:{...} / ocr:{...} / asr:{...} / lvlm:{...}",
          "type": "videos",
          "asset_path": "video/mp4/川普參觀天壇_...",
          "version_id": "109fd79e..."
        }
      }
    ]
  }
}
```

---

<a id="benchmark"></a>

## Benchmark

A proxy for Module 22, with endpoints under `/api/0.4/benchmark/*`. Upload a "schema" (test cases + scoring
rules), run it once against a real pipeline to get an objective score; the same schema can be run
repeatedly and compared, to track whether a change made output better or worse.

`POST /runs` is asynchronous: it returns `202` + `run_id`, the actual run executes in the background, and
you poll `GET /runs/{run_id}` until `status` becomes `completed`/`failed`.

**This service itself is only a scoring mechanism — its value depends entirely on how well the schema is written.** The API responding and running successfully does not mean the score is meaningful —
`keyword_match` (checking whether a word appears somewhere in a sentence) is easy to write as a test case, but what actually reflects output quality is a test case paired with `llm_judge` (given a clear rubric) or `expected_answer` + `cosine_similarity`, covering the scenarios actually affected by whatever change you're measuring. Writing a handful of test cases just to confirm the API works is not the same as having a schema that can tell you whether a change made things better or worse.

### `target_pipeline` values (all live-tested)

| Value | Hits | Needs `branch_id`? |
|---|---|---|
| `chat` | Module 15 chat-service | No |
| `rag` | Module 21 agent-protocol (`mode=direct`) | **Yes** |
| `search` | Module 25 personal-db-service (`/personal/search/hybrid`) | **Yes** |
| `classify` | Module 18 query-orchestrator | No |
| `lifecycle_infer` | Module 07 AI Lifecycle API (the model under test must already be registered in 07) | No |
| `local_infer` | Module 16 training-service | No, but **Module 16 isn't deployed in this environment, so it can't be used** |

**`branch_id` for `search`/`rag`:** Module 25 is one physically isolated database per person, with no shared corpus.
Each test case's `input` can set its own `branch_id` (or `user_id`, equivalent) to specify whose data to test
against; if omitted, it automatically uses the identity of **whoever submitted this run** (Module 13 extracts it
from your login JWT and fills it in automatically — no manual handling needed). If neither is present, it errors
outright rather than silently searching someone else's data or quietly returning empty results.

`/optimize` (AutoTune: the train → score → suggest-next-params loop) also needs Module 16, so it's likewise unusable in this deployment.

### `POST /api/0.4/benchmark/schemas`

Uploads a schema.

**curl:**

```bash
curl -X POST http://raptor_open_0_4_api.dhtsolution.com:8012/api/0.4/benchmark/schemas \
  -H "Authorization: Bearer <access_token>" -H "Content-Type: application/json" \
  -d '{
    "name": "chat_url retest 3",
    "version": "1.0",
    "target_pipeline": "chat",
    "test_cases": [
      {"id": "tc001", "input": {"message": "1+1等於多少？", "user_id": "benchmark-user"}, "expected_keywords": ["2"]}
    ],
    "scoring_schema": {
      "dimensions": [{"name": "keywords", "weight": 1.0, "method": "keyword_match"}],
      "aggregate": "weighted_sum",
      "score_range": [0, 1]
    }
  }'
```

**Response (real response):**

```json
{"id":"d0873929-060d-4b14-8775-538b8336a206","name":"chat_url retest 3","version":"1.0","created_at":"2026-08-28T04:04:01.865316Z"}
```

Scoring methods (`scoring_schema.dimensions[].method`): `keyword_match`, `contains_all`, `contains_any`,
`exact_match`, `numeric_tolerance`, `regex_match`, `cosine_similarity` (via Module 07 embedding),
`llm_judge` (natural-language rubric, via Module 07 LLM), `latency_threshold`.

### `POST /api/0.4/benchmark/runs`

**curl:**

```bash
curl -X POST http://raptor_open_0_4_api.dhtsolution.com:8012/api/0.4/benchmark/runs \
  -H "Authorization: Bearer <access_token>" -H "Content-Type: application/json" \
  -d '{"schema_id": "d0873929-060d-4b14-8775-538b8336a206"}'
```

**Response:**

```json
{"run_id":"2b24dbec-a90b-4a68-89d6-0235f4340458","status":"queued"}
```

### `GET /api/0.4/benchmark/runs/{run_id}`

Poll until `status` becomes `completed`/`failed`.

**Response (real response):**

```json
{
  "run_id": "2b24dbec-a90b-4a68-89d6-0235f4340458",
  "schema_id": "d0873929-060d-4b14-8775-538b8336a206",
  "status": "completed",
  "aggregate_score": 1.0,
  "scores_per_dimension": {"keywords": 1.0},
  "scores_per_case": [
    {"case_id": "tc001", "output": "**1 + 1 = 2**。\n\n這是基本的數學運算規則，將兩個數字相加即可得到結果為 **2**。",
     "aggregate": 1.0, "latency_ms": 9573.970083147287, "error": null}
  ],
  "mlflow_run_id": "de53895f40364b38b49bf102d419c18f"
}
```

A non-`null` `error` means that particular test case itself failed (e.g. the wrong pipeline's service, or
`search`/`rag` had no usable `branch_id`) — `status` will still be `completed`; you have to check each
case's `error` field individually, not just the outer `status`.

### Other endpoints

| Method | Path | Description |
|--------|------|------|
| `GET` | `/benchmark/schemas` | Lists all schemas |
| `GET` / `DELETE` | `/benchmark/schemas/{id}` | Get / delete a schema (deleting also deletes its runs) |
| `GET` | `/benchmark/schemas/{id}/runs` | Run history for that schema |
| `GET` | `/benchmark/schemas/{id}/leaderboard` | Runs ranked by score |
| `GET` | `/benchmark/runs/{a}/compare/{b}` | Compares the score delta between two runs; `?pairwise=true` uses LLM pairwise comparison instead |

---

<a id="personal-db"></a>

## Personal DB Service — Direct API (Module 25)

**This section is different from everything above: these endpoints do not go through the API Gateway
(Module 13), are not mounted under `/api/0.4/*`, and don't use Bearer JWT auth.** Module 25 is one
physically isolated ArcadeDB database per user — it's what actually gets queried behind Search / RAG / A2A
(`hybrid`/`bm25`/`vector`/`graphrag`/`tkg`) — but it is itself an internal service meant to be called directly
by Modules 09–13/15/21 and the Kafka consumer, not hit directly by a frontend/end user. Auth here is via the
**`X-Branch-ID`** header (not a Bearer Token) — the Gateway normally derives this from the caller's JWT and
passes it down; calling directly means supplying it yourself.

**Base URL:** `http://<host>:${PORT_PERSONAL_DB:-8025}` (container always listens on `8000` internally)
**Interactive docs:** `GET /docs`

| Prefix | Description |
|---|---|
| `/internal/db/*` | Database lifecycle (create/drop/check-exists for a user's database) |
| `/personal/index/*` | Document/moment chunk indexing (PA-4) + entity/relationship/temporal-fact indexing (PA-5, same prefix, separate router) |
| `/personal/search/*` | hybrid/vector/bm25 search (PA-6) + graph/TKG/GraphRAG search (PA-7, same prefix, separate router) |
| `/personal/graph/*` | Entity list/detail + raw graph query (PA-7) |
| `/personal/publish/*` | Publishes an index-request straight to Kafka (test/demo use — the only endpoint here that carries `branch_id` in the body rather than the header, since it's simulating an upstream worker) |

Full request/response schemas: see `/docs` (Swagger) or
[`deployment/modules/25-personal-db-service/doc/personal-db-service.md`](deployment/modules/25-personal-db-service/doc/personal-db-service.md).

---

<a id="guardrail"></a>

## Guardrail Service — Direct API (Module 23)

Also **bypasses the API Gateway**, calling Module 23's own port directly. The service has three independent
safety mechanisms (all reading the same Postgres policy table, but interpreting it differently), plus one
older mechanism that doesn't look at policy at all:

| Mechanism | Prefix | How it works |
|---|---|---|
| Guard-model classification | `/guard/check/*` | Raw multi-model classification (Llama-Guard3 / Granite Guardian / GPT-OSS-Safeguard), using each model's fixed guard prompt — no policy involved |
| LLM-judged policy check | `/policy/check/llm/*` | Same multi-model dispatch as above, but each model receives the currently-active policy's content as its own prompt |
| GB-4 detector checker | `/guardrail/check/*` | Checks against the currently-active policy's regex + Llama-Guard detector rules; also handles violation audit logging and the global switch (`/guardrail/system/*`) and policy CRUD (`/guardrail/policies*`) |
| Confidence-based proxy (legacy mechanism) | `POST /api/generate` | A single classification model returns `{category, confidence, reason}`, allowed/blocked based on a configurable confidence threshold; unrelated to the policy format of the three mechanisms above |

**Actual deployment status:** Module 07/13's checks currently all go through `/guard/check/*` (no policy
involved) — Module 23 has never had an active policy configured, so live decisions rely on the guard
models' own built-in judgment. The whole service is **disabled by default** (see the enable switches in
`07-ai-ml-services`/`13-api-services`'s `.env.example`) — it only actually blocks content once turned on.

**Base URL:** `http://<host>:${PORT_GUARDRAIL_SERVICE:-8023}` (container always listens on `8026` internally)
**Health check:** `GET /health`  **Interactive docs:** `GET /docs`

Full endpoint and environment-variable documentation:
[`deployment/modules/23-guardrail-service/README.md`](deployment/modules/23-guardrail-service/README.md).

---

<a id="fields"></a>

## Response Field Reference

### Payload fields (`payload` inside search results)

| Field | Description |
|------|------|
| `type` | Media type: `videos` / `documents` / `audios` / `images` |
| `source` | Original format: `mp4` / `pdf` / `csv` / `jpg`, etc. |
| `filename` | Original filename |
| `asset_path` | Asset path in LakeFS |
| `version_id` | Version ID (64 hex chars) |
| `chunk_index` | Text-chunk index (a single file split by time/page) |
| `start_time` / `end_time` | Video segment timestamps (seconds, as a string) |
| `speaker` | Speaker ID (tagged by audio analysis) |
| `text` | Four text channels: `contextual` (contextual summary), `ocr` (on-screen text), `asr` (speech-to-text), `lvlm` (vision-language-model description) |
| `asset_url` | Presigned download URL (valid 24 hours) |

### Common HTTP status codes

| Code | Meaning |
|--------|------|
| `200` | Success |
| `401` | Unauthenticated (token invalid or expired) |
| `404` | Resource not found |
| `422` | Malformed request parameters |
| `429` | Rate limit exceeded |
| `502` | Downstream service unreachable |
| `504` | Downstream service response timed out |

### Rate Limits

| Endpoint | Limit |
|------|------|
| `fileupload_analysis` | 50 requests/minute |
| `fileupload_analysis_batch` | 10 requests/minute |

---

## Full Python example: upload a video, wait for processing, then search

```python
import requests
import time

BASE_URL = "http://raptor_open_0_4_api.dhtsolution.com:8012"

# 1. Log in
resp = requests.post(f"{BASE_URL}/api/0.4/sso/login",
                     data={"username": "test_basicuser", "password": "12345"})
token = resp.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# 2. Upload a video
with open("video.mp4", "rb") as f:
    resp = requests.post(
        f"{BASE_URL}/api/0.4/asset/fileupload_analysis",
        headers=headers,
        files={"primary_file": f},
        data={"processing_mode": "default"},
    )
upload = resp.json()
version_id = upload["upload_result"]["version_id"]
processing = upload["processing_result"]
print(f"upload complete, version_id={version_id[:16]}...")

if "status" in processing and processing["status"] == "skipped":
    print("file already exists, skipping processing")
else:
    correlation_id = processing["correlation_id"]  # new file: present in processing_result
    # 3. Poll until processing completes
    while True:
        resp = requests.get(
            f"{BASE_URL}/api/0.4/processing/cache/video/{correlation_id}",
            headers=headers,
        )
        if resp.status_code == 404:
            time.sleep(5)
            continue
        step = resp.json().get("value", {}).get("step")
        print(f"processing step: {step}")
        if step in ("complete", "failed"):
            break
        time.sleep(10)

# 4. Search
resp = requests.post(
    f"{BASE_URL}/api/0.4/search/hybrid",
    headers={**headers, "Content-Type": "application/json"},
    json={"query": "川普習近平 會面內容", "top_k": 5},
)
for r in resp.json()["results"]:
    print(f"score={r['score']:.4f} | {r['payload'].get('filename','?')}")

# 5. RAG question-answering
resp = requests.post(
    f"{BASE_URL}/api/0.4/a2a/query",
    headers={**headers, "Content-Type": "application/json"},
    json={"question": "川普和習近平會面討論了哪些議題？", "top_k": 5, "mode": "direct"},
    timeout=120,
)
print("\nRAG answer:", resp.json()["answer"])
```
