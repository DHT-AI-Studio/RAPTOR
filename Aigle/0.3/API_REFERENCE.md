# Raptor 0.3 API Reference

**Base URL:** `http://raptor_open_0_3_api.dhtsolution.com:8012`  
**API Version:** `0.3`  
**Authentication:** Bearer Token (Keycloak JWT)  

Swagger UI: `http://raptor_open_0_3_api.dhtsolution.com:8012/docs`

---

## 目錄

- [認證 (Single Sign-On)](#sso)
- [資產管理 (Asset)](#asset)
- [搜尋 (Search)](#search)
- [影片搜尋 (Video Search)](#video-search)
- [RAG 查詢 (A2A Protocol)](#rag)
- [處理進度 (Processing)](#processing)
- [回應欄位說明](#fields)

---

<a id="sso"></a>

## 認證 (Single Sign-On)

所有需要認證的 API 都必須在 Header 帶入 Bearer Token：

```
Authorization: Bearer <access_token>
```

---

### `POST /api/0.3/sso/login`

使用者登入，取得 JWT access_token。

**Request：** `application/x-www-form-urlencoded`

| 參數 | 型別 | 必填 | 說明 |
|------|------|------|------|
| `username` | string | ✓ | 使用者帳號 |
| `password` | string | ✓ | 密碼 |
| `realm_name` | string | | Keycloak realm，預設 `dhtsolution` |
| `client_id` | string | | OAuth client，預設 `raptor`（Query param） |

**Response：**

```json
{
  "access_token": "eyJhbGci...",
  "token_type": "bearer",
  "refresh_token": "eyJhbGci..."
}
```

**curl：**

```bash
curl -X POST http://raptor_open_0_3_api.dhtsolution.com:8012/api/0.3/sso/login \
  -d "username=test_basicuser&password=12345"
```

**Python：**

```python
import requests

resp = requests.post(
    "http://raptor_open_0_3_api.dhtsolution.com:8012/api/0.3/sso/login",
    data={"username": "test_basicuser", "password": "12345"},
)
resp.raise_for_status()
token = resp.json()["access_token"]
print("Token:", token[:40], "...")
```

---

### `POST /api/0.3/sso/logout`

透過 Keycloak 登出目前 session。

> **注意：** 此端點需要瀏覽器登入時設定的 `realm_name` cookie，純 API 程式呼叫（不帶 cookie）會回傳 `{"detail": "Missing realm_name cookie"}`。  
> JWT access_token 無法透過 API 強制失效，token 會在其 `exp` 到期後自然失效（通常為 5–30 分鐘）。若需要立即撤銷 session，請從瀏覽器呼叫此端點（cookie 會自動帶入）。

**Headers：** `Authorization: Bearer <token>`  
**Cookie：** `realm_name=dhtsolution`（瀏覽器登入後自動設定）

**curl（帶 cookie）：**

```bash
curl -X POST http://raptor_open_0_3_api.dhtsolution.com:8012/api/0.3/sso/logout \
  -H "Authorization: Bearer <access_token>" \
  -b "realm_name=dhtsolution"
```

**Python（帶 cookie）：**

```python
import requests

headers = {"Authorization": f"Bearer {token}"}
resp = requests.post(
    "http://raptor_open_0_3_api.dhtsolution.com:8012/api/0.3/sso/logout",
    headers=headers,
    cookies={"realm_name": "dhtsolution"},
)
print(resp.status_code, resp.json())
```

---

<a id="asset"></a>

## 資產管理 (Asset)

### `POST /api/0.3/asset/fileupload_analysis`

上傳單一檔案並觸發 AI 分析處理（Kafka 非同步）。  
支援影片（mp4）、音訊、PDF、圖片等格式。

若檔案已存在（相同內容 hash），直接回傳 `skipped` 狀態，不重複處理。

**Headers：** `Authorization: Bearer <token>`  
**Content-Type：** `multipart/form-data`

| 參數 | 型別 | 必填 | 說明 |
|------|------|------|------|
| `primary_file` | file | ✓ | 要上傳的檔案 |
| `archive_date_or_ttl` | string | | 封存日期（ISO 8601）或 TTL 天數（如 `"30"`） |
| `destroy_date_or_ttl` | string | | 刪除日期或 TTL 天數 |
| `processing_mode` | string | | 處理模式，預設 `"default"` |

**Response（新檔案）：**

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

**Response（重複檔案）：**

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

**curl：**

```bash
curl -X POST http://raptor_open_0_3_api.dhtsolution.com:8012/api/0.3/asset/fileupload_analysis \
  -H "Authorization: Bearer <access_token>" \
  -F "primary_file=@/path/to/川普訪北京.mp4" \
  -F "processing_mode=default"
```

**Python：**

```python
import requests

headers = {"Authorization": f"Bearer {token}"}
with open("川普訪北京.mp4", "rb") as f:
    resp = requests.post(
        "http://raptor_open_0_3_api.dhtsolution.com:8012/api/0.3/asset/fileupload_analysis",
        headers=headers,
        files={"primary_file": f},
        data={"processing_mode": "default"},
    )
resp.raise_for_status()
result = resp.json()
print("version_id:", result["upload_result"]["version_id"])
processing = result["processing_result"]
if "correlation_id" in processing:
    # 新檔案：取得 correlation_id 後續查詢處理進度
    correlation_id = processing["correlation_id"]
    print("correlation_id:", correlation_id)
else:
    # 重複檔案
    print("status:", processing["status"])  # "skipped"
```

---

### `POST /api/0.3/asset/fileupload_analysis_batch`

批次上傳多個檔案並觸發 AI 分析。

**Headers：** `Authorization: Bearer <token>`  
**Content-Type：** `multipart/form-data`

| 參數 | 型別 | 必填 | 說明 |
|------|------|------|------|
| `primary_files` | file[] | ✓ | 多個檔案 |
| `archive_date_or_ttl` | string | | 封存設定 |
| `destroy_date_or_ttl` | string | | 刪除設定 |
| `processing_mode` | string | | 處理模式，預設 `"default"` |
| `concurrency` | int | | 並行上傳數，預設 `4`，最大 `8` |

**Response：**

```json
{
  "upload_summary": {
    "total": 2,
    "success_count": 2,
    "failure_count": 0
  },
  "upload_successes": [
    {
      "filename": "川普訪北京.mp4",
      "result": {"version_id": "...", "asset_path": "..."}
    }
  ],
  "upload_failures": [],
  "processing_results": [
    {
      "filename": "川普訪北京.mp4",
      "status": "kafka_sent",
      "kafka_result": {}
    }
  ]
}
```

**curl：**

```bash
curl -X POST http://raptor_open_0_3_api.dhtsolution.com:8012/api/0.3/asset/fileupload_analysis_batch \
  -H "Authorization: Bearer <access_token>" \
  -F "primary_files=@/path/to/video1.mp4" \
  -F "primary_files=@/path/to/video2.mp4" \
  -F "processing_mode=default" \
  -F "concurrency=4"
```

**Python：**

```python
import requests

headers = {"Authorization": f"Bearer {token}"}
files = [
    ("primary_files", open("video1.mp4", "rb")),
    ("primary_files", open("video2.mp4", "rb")),
]
resp = requests.post(
    "http://raptor_open_0_3_api.dhtsolution.com:8012/api/0.3/asset/fileupload_analysis_batch",
    headers=headers,
    files=files,
    data={"processing_mode": "default", "concurrency": "4"},
)
resp.raise_for_status()
result = resp.json()
print("成功:", result["upload_summary"]["success_count"])
print("失敗:", result["upload_summary"]["failure_count"])
```

---

### `GET /api/0.3/asset/filedownload/{asset_path}/{version_id}`

取得指定版本的資產下載資訊，包含 presigned URL（有效期 24 小時）。

若為影片，同時回傳所有相關 frame 圖片的 presigned URL。

**Headers：** `Authorization: Bearer <token>`  
**Path Params：**

| 參數 | 說明 |
|------|------|
| `asset_path` | 資產路徑，如 `video/mp4/川普參觀天壇_習近平陪同_記者連三問是否談台灣_已讀不回__shorts__鏡新聞_1080p` |
| `version_id` | 版本 ID（64 hex chars） |

**Query Params：**

| 參數 | 型別 | 說明 |
|------|------|------|
| `return_file_content` | bool | `true` 時直接回傳二進位內容（預設 `false`） |

**Response：**

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
    "url": "http://raptor_open_0_3_api.dhtsolution.com:8333/lakefs/data/..."
  },
  "associated_file_0": {
    "filename": "frame_0001.jpg",
    "url": "http://raptor_open_0_3_api.dhtsolution.com:8333/lakefs/data/..."
  }
}
```

**curl：**

```bash
ASSET_PATH="video/mp4/川普參觀天壇_習近平陪同_記者連三問是否談台灣_已讀不回__shorts__鏡新聞_1080p"
VERSION_ID="109fd79e263ae402334255f27660878b5ecc9eafe6a110e6a960c9f462d9cc37"

curl -G "http://raptor_open_0_3_api.dhtsolution.com:8012/api/0.3/asset/filedownload/${ASSET_PATH}/${VERSION_ID}" \
  -H "Authorization: Bearer <access_token>"
```

**Python：**

```python
import requests
from urllib.parse import quote

asset_path = "video/mp4/川普參觀天壇_習近平陪同_記者連三問是否談台灣_已讀不回__shorts__鏡新聞_1080p"
version_id = "109fd79e263ae402334255f27660878b5ecc9eafe6a110e6a960c9f462d9cc37"

headers = {"Authorization": f"Bearer {token}"}
resp = requests.get(
    f"http://raptor_open_0_3_api.dhtsolution.com:8012/api/0.3/asset/filedownload/{asset_path}/{version_id}",
    headers=headers,
)
resp.raise_for_status()
data = resp.json()
print("主檔案 URL:", data["primary_file"]["url"])
# 列出所有關聯 frame
for key, val in data.items():
    if key.startswith("associated_file_"):
        print(f"  {val['filename']}: {val['url'][:60]}...")
```

---

### `GET /api/0.3/asset/fileversions/{asset_path}/{filename}`

查詢指定資產檔案的所有版本歷史，每個版本含 presigned URL。

**Headers：** `Authorization: Bearer <token>`  
**Path Params：**

| 參數 | 說明 |
|------|------|
| `asset_path` | 資產路徑，如 `video/mp4/川普參觀天壇_習近平陪同_記者連三問是否談台灣_已讀不回__shorts__鏡新聞_1080p` |
| `filename` | 主檔名，如 `川普參觀天壇_習近平陪同_記者連三問是否談台灣_已讀不回__shorts__鏡新聞_1080p.mp4` |

**Response：**

```json
{
  "versions": [
    {
      "key": "video/mp4/川普參觀天壇_.../川普參觀天壇_...mp4",
      "version_id": "109fd79e263ae402334255f27660878b5ecc9eafe6a110e6a960c9f462d9cc37",
      "upload_date": "2026-05-18T21:52:45.843945+08:00",
      "status": "active",
      "url": "http://raptor_open_0_3_api.dhtsolution.com:8333/lakefs/data/..."
    }
  ]
}
```

**curl：**

```bash
ASSET_PATH="video/mp4/川普參觀天壇_習近平陪同_記者連三問是否談台灣_已讀不回__shorts__鏡新聞_1080p"
FILENAME="川普參觀天壇_習近平陪同_記者連三問是否談台灣_已讀不回__shorts__鏡新聞_1080p.mp4"

curl "http://raptor_open_0_3_api.dhtsolution.com:8012/api/0.3/asset/fileversions/${ASSET_PATH}/${FILENAME}" \
  -H "Authorization: Bearer <access_token>"
```

**Python：**

```python
import requests

asset_path = "video/mp4/川普參觀天壇_習近平陪同_記者連三問是否談台灣_已讀不回__shorts__鏡新聞_1080p"
filename = "川普參觀天壇_習近平陪同_記者連三問是否談台灣_已讀不回__shorts__鏡新聞_1080p.mp4"

headers = {"Authorization": f"Bearer {token}"}
resp = requests.get(
    f"http://raptor_open_0_3_api.dhtsolution.com:8012/api/0.3/asset/fileversions/{asset_path}/{filename}",
    headers=headers,
)
resp.raise_for_status()
versions = resp.json()["versions"]
for v in versions:
    print(f"version_id: {v['version_id'][:16]}... | status: {v['status']} | date: {v['upload_date']}")
```

---

### `GET /api/0.3/asset/users/commits`

查詢目前登入使用者所有已上傳資產的清單，支援關鍵字搜尋與日期過濾。

**Headers：** `Authorization: Bearer <token>`  
**Query Params：**

| 參數 | 型別 | 說明 |
|------|------|------|
| `keyword` | string | 檔名關鍵字搜尋 |
| `start_date` | string | 起始日期（ISO 8601） |
| `end_date` | string | 結束日期（ISO 8601） |
| `page` | int | 頁碼，預設 `1` |
| `page_size` | int | 每頁筆數，預設 `10` |

**Response：**

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

**curl：**

```bash
# 搜尋包含「川普」的資產
curl "http://raptor_open_0_3_api.dhtsolution.com:8012/api/0.3/asset/users/commits?keyword=川普&page=1&page_size=5" \
  -H "Authorization: Bearer <access_token>"
```

**Python：**

```python
import requests

headers = {"Authorization": f"Bearer {token}"}
resp = requests.get(
    "http://raptor_open_0_3_api.dhtsolution.com:8012/api/0.3/asset/users/commits",
    headers=headers,
    params={"keyword": "川普", "page": 1, "page_size": 5},
)
resp.raise_for_status()
data = resp.json()
print(f"共 {data['total_count']} 筆，第 {data['page']}/{data['total_pages']} 頁")
for c in data["commits"]:
    print(f"  {c['primary_filename']} | {c['status']}")
```

---

### `POST /api/0.3/asset/filearchive/{asset_path}/{version_id}`

將指定版本標記為封存（archived）狀態。封存後仍可存取但不會出現在主要清單。

**Headers：** `Authorization: Bearer <token>`

**curl：**

```bash
curl -X POST \
  "http://raptor_open_0_3_api.dhtsolution.com:8012/api/0.3/asset/filearchive/video%2Fmp4%2F川普啟動自由計劃/b02a0b42ba37f06cafeedd55e953762dc42988e9aa5163f877a6c170f08d0ebb" \
  -H "Authorization: Bearer <access_token>"
```

**Python：**

```python
import requests

asset_path = "video/mp4/川普啟動自由計劃"
version_id = "b02a0b42ba37f06cafeedd55e953762dc42988e9aa5163f877a6c170f08d0ebb"

headers = {"Authorization": f"Bearer {token}"}
resp = requests.post(
    f"http://raptor_open_0_3_api.dhtsolution.com:8012/api/0.3/asset/filearchive/{asset_path}/{version_id}",
    headers=headers,
)
resp.raise_for_status()
print(resp.json())
```

---

### `POST /api/0.3/asset/delfile/{asset_path}/{version_id}`

刪除指定版本的資產（從 LakeFS 移除）。此操作不可逆。

**Headers：** `Authorization: Bearer <token>`

**curl：**

```bash
curl -X POST \
  "http://raptor_open_0_3_api.dhtsolution.com:8012/api/0.3/asset/delfile/video%2Fmp4%2F川普啟動自由計劃/b02a0b42ba37f06cafeedd55e953762dc42988e9aa5163f877a6c170f08d0ebb" \
  -H "Authorization: Bearer <access_token>"
```

**Python：**

```python
import requests

asset_path = "video/mp4/川普啟動自由計劃"
version_id = "b02a0b42ba37f06cafeedd55e953762dc42988e9aa5163f877a6c170f08d0ebb"

headers = {"Authorization": f"Bearer {token}"}
resp = requests.post(
    f"http://raptor_open_0_3_api.dhtsolution.com:8012/api/0.3/asset/delfile/{asset_path}/{version_id}",
    headers=headers,
)
resp.raise_for_status()
print(resp.json())
```

---

### `POST /api/0.3/asset/file-expiration/{asset_path}/{version_id}`

更新資產的封存日期或刪除日期。可傳入 ISO 8601 日期字串或天數（TTL）。

> **注意：** 若資產上傳時未設定任何過期日（permanent 資產），`archive_date_or_ttl` 與 `destroy_date_or_ttl` 必須**同時提供**，否則回傳 `400`。

**Headers：** `Authorization: Bearer <token>`  
**Content-Type：** `application/x-www-form-urlencoded`

| 參數 | 型別 | 說明 |
|------|------|------|
| `archive_date_or_ttl` | int \| string | 封存日期（`"2026-12-31"`）或天數（`30`） |
| `destroy_date_or_ttl` | int \| string | 刪除日期或天數 |

**curl：**

```bash
curl -X POST \
  "http://raptor_open_0_3_api.dhtsolution.com:8012/api/0.3/asset/file-expiration/video%2Fmp4%2F川普啟動自由計劃/b02a0b42ba37f06cafeedd55e953762dc42988e9aa5163f877a6c170f08d0ebb" \
  -H "Authorization: Bearer <access_token>" \
  -d "archive_date_or_ttl=90&destroy_date_or_ttl=365"
```

**Python：**

```python
import requests

asset_path = "video/mp4/川普啟動自由計劃"
version_id = "b02a0b42ba37f06cafeedd55e953762dc42988e9aa5163f877a6c170f08d0ebb"

headers = {"Authorization": f"Bearer {token}"}
resp = requests.post(
    f"http://raptor_open_0_3_api.dhtsolution.com:8012/api/0.3/asset/file-expiration/{asset_path}/{version_id}",
    headers=headers,
    data={"archive_date_or_ttl": "90", "destroy_date_or_ttl": "365"},
)
resp.raise_for_status()
print(resp.json())
```

---

<a id="search"></a>

## 搜尋 (Search)

所有搜尋端點皆需認證，回傳格式統一為 `{"results": [...]}` 陣列。

每筆結果 `payload` 中的 `asset_url` 為資產的 presigned 下載連結（24 小時有效）。

### 共用 Request Schema（HybridSearchRequest）

| 欄位 | 型別 | 預設 | 說明 |
|------|------|------|------|
| `query` | string | 必填 | 搜尋文字 |
| `top_k` | int | `10` | 最多回傳幾筆 |
| `payload_schema` | string | `"contextual"` | `"contextual"` 或 `"temporal"` |
| `embedding_type` | string | null | `"text"` 或 `"summary"` |
| `type` | string \| string[] | null | 資產類型過濾，如 `"videos"`、`["documents","audios"]` |
| `filename` | string[] | null | 依檔名過濾 |
| `speaker` | string[] | null | 依說話者過濾（影片/音訊） |
| `source` | string | null | 依來源格式過濾，如 `"pdf"`、`"mp4"` |

---

### `POST /api/0.3/search/hybrid`

混合搜尋（BM25 + Vector 融合），結果經 RRF 排序後再重排序（rerank）。  
適合大多數一般搜尋情境。

**curl：**

```bash
curl -X POST http://raptor_open_0_3_api.dhtsolution.com:8012/api/0.3/search/hybrid \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "川習會 貿易談判",
    "top_k": 5
  }'
```

**curl（加過濾條件）：**

```bash
curl -X POST http://raptor_open_0_3_api.dhtsolution.com:8012/api/0.3/search/hybrid \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "川普習近平 關稅",
    "top_k": 5,
    "type": "videos",
    "filename": ["川普強調美國AI領先中國_曝與習近平會晤日期_民視新聞__1080p.mp4"]
  }'
```

**Python：**

```python
import requests

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
}
resp = requests.post(
    "http://raptor_open_0_3_api.dhtsolution.com:8012/api/0.3/search/hybrid",
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

**Response 範例：**

```json
{
  "results": [
    {
      "id": "79626f7d-437e-4137-8a5d-3b0805172a18",
      "score": 0.1867,
      "payload": {
        "filename": "直球對決__習近平_台獨和台海和平水火不容__TVBS新聞__TVBSNEWS01_1080p.mp4",
        "type": "videos",
        "source": "mp4",
        "chunk_index": 3,
        "start_time": "30.0",
        "end_time": "40.0",
        "text": "contextual:{...} / ocr:{...} / asr:{...} / lvlm:{...}",
        "asset_path": "video/mp4/直球對決__習近平_台獨和台海和平水火不容__TVBS新聞__TVBSNEWS01_1080p",
        "version_id": "...",
        "branch_id": "c9d6b4ab-ad57-488b-987a-141f59fda512",
        "asset_url": "http://raptor_open_0_3_api.dhtsolution.com:8333/lakefs/data/..."
      }
    }
  ]
}
```

---

### `POST /api/0.3/search/bm25`

純 BM25（關鍵字）搜尋，使用 OpenSearch 引擎。  
適合精確關鍵字比對，不使用語意向量。Score 為 BM25 分數（越高越相關）。

**curl：**

```bash
curl -X POST http://raptor_open_0_3_api.dhtsolution.com:8012/api/0.3/search/bm25 \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "川普 貿易關稅",
    "top_k": 5
  }'
```

**Python：**

```python
import requests

headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
resp = requests.post(
    "http://raptor_open_0_3_api.dhtsolution.com:8012/api/0.3/search/bm25",
    headers=headers,
    json={"query": "川普 貿易關稅", "top_k": 5},
)
resp.raise_for_status()
for r in resp.json()["results"]:
    print(f"BM25 score={r['score']:.2f} | {r['payload'].get('filename','?')}")
```

---

### `POST /api/0.3/search/vector`

純向量（語意）搜尋，使用 Qdrant。  
適合語意相似搜尋，score 為餘弦相似度（0–1）。

**curl：**

```bash
curl -X POST http://raptor_open_0_3_api.dhtsolution.com:8012/api/0.3/search/vector \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "川普與習近平 AI 科技競爭",
    "top_k": 5
  }'
```

**Python：**

```python
import requests

headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
resp = requests.post(
    "http://raptor_open_0_3_api.dhtsolution.com:8012/api/0.3/search/vector",
    headers=headers,
    json={"query": "川普與習近平 AI 科技競爭", "top_k": 5},
)
resp.raise_for_status()
for r in resp.json()["results"]:
    print(f"cosine={r['score']:.4f} | {r['payload'].get('filename','?')}")
```

---

### `POST /api/0.3/search/graphrag`

GraphRAG 知識圖譜搜尋，透過實體節點 + 子圖擴展找出相關關係。  
適合「X 是誰」、「X 和 Y 有什麼關係」等實體/關係查詢。

**Request Schema：**

| 欄位 | 型別 | 預設 | 說明 |
|------|------|------|------|
| `query` | string | 必填 | 自然語言查詢 |
| `max_depth` | int | `2` | 子圖擴展深度（1–4） |
| `limit` | int | `50` | 最多回傳節點數（1–200） |
| `score_threshold` | float | `0.5` | 最低分數過濾（0–10） |
| `strategy` | string | `"hybrid"` | 搜尋策略：`"hybrid"` / `"literal"` / `"semantic"` |

**curl：**

```bash
curl -X POST http://raptor_open_0_3_api.dhtsolution.com:8012/api/0.3/search/graphrag \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "川普習近平關係",
    "max_depth": 2,
    "limit": 20,
    "strategy": "hybrid"
  }'
```

**Python：**

```python
import requests

headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
resp = requests.post(
    "http://raptor_open_0_3_api.dhtsolution.com:8012/api/0.3/search/graphrag",
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

**Response 範例：**

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
  "nodes": [...],
  "relationships": [...],
  "citations": [...]
}
```

---

### `POST /api/0.3/search/tkg`

時序知識圖譜（Temporal Knowledge Graph）搜尋，查詢有時間順序的事件關係。  
適合「X 發生在 Y 之前/之後」、「事件時間線」等時序查詢。

**Request Schema：**

| 欄位 | 型別 | 預設 | 說明 |
|------|------|------|------|
| `query` | string | 必填 | 自然語言查詢 |
| `time_start` | string | null | 時間區間起點（ISO 8601），如 `"2025-01-01"` |
| `time_end` | string | null | 時間區間終點 |
| `max_depth` | int | `2` | 子圖擴展深度（1–4） |
| `limit` | int | `50` | 最多回傳節點數（1–200） |
| `score_threshold` | float | `0.3` | 最低分數過濾（0–10） |

**curl：**

```bash
curl -X POST http://raptor_open_0_3_api.dhtsolution.com:8012/api/0.3/search/tkg \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "川習峰會時間序列",
    "max_depth": 2,
    "limit": 20
  }'
```

**curl（指定時間區間）：**

```bash
curl -X POST http://raptor_open_0_3_api.dhtsolution.com:8012/api/0.3/search/tkg \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "中美貿易談判",
    "time_start": "2025-01-01",
    "time_end": "2026-12-31",
    "max_depth": 2
  }'
```

**Python：**

```python
import requests

headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
resp = requests.post(
    "http://raptor_open_0_3_api.dhtsolution.com:8012/api/0.3/search/tkg",
    headers=headers,
    json={"query": "川習峰會時間序列", "max_depth": 2, "limit": 20},
)
resp.raise_for_status()
data = resp.json()
for e in data.get("matched_entities", [])[:5]:
    print(f"  [{e['type']}] {e['name']} | {e.get('description','')[:60]}")
print("temporal_facts:", len(data.get("temporal_facts", [])))
```

**Response 範例：**

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

## 影片搜尋 (Video Search)

### `POST /api/0.3/search/video_search`

專為影片設計的搜尋，以「影片為單位」聚合回傳，而非片段（chunk）為單位。  
使用多路召回（BM25 + Vector + GraphRAG）後經 cross-encoder rerank，每個影片回傳最相關的時間片段列表。

**Request Schema：**

| 欄位 | 型別 | 預設 | 說明 |
|------|------|------|------|
| `query` | string | 必填 | 自然語言搜尋 |
| `top_k` | int | `10` | 最多回傳幾部影片（1–50） |
| `candidate_multiplier` | int | `5` | 每個召回器取 `top_k × N` 候選片段（1–20） |
| `score_threshold` | float | `0.52` | 最低分數過濾（0–1），低於此值的片段不顯示 |

**curl：**

```bash
curl -X POST http://raptor_open_0_3_api.dhtsolution.com:8012/api/0.3/search/video_search \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "川普習近平",
    "top_k": 5
  }'
```

**Python：**

```python
import requests

headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
resp = requests.post(
    "http://raptor_open_0_3_api.dhtsolution.com:8012/api/0.3/search/video_search",
    headers=headers,
    json={"query": "川普習近平", "top_k": 5},
)
resp.raise_for_status()
data = resp.json()
print(f"找到 {data['total']} 部影片")
for video in data["results"]:
    print(f"\n[{video['score']:.4f}] {video['filename']}")
    print(f"  下載: {video['asset_url'][:80]}...")
    for seg in video.get("segments", [])[:2]:
        print(f"  片段 {seg['start_time']}s–{seg['end_time']}s (score={seg['score']:.4f})")
        print(f"    來源: {seg['sources']}")
```

**Response 範例：**

```json
{
  "query": "川普習近平",
  "total": 3,
  "results": [
    {
      "video_id": "d9243fa19b59c4daf793d77ebe31ba91433b2826d8185e4657be2cd8fd3a52d2",
      "filename": "川普強調美國AI領先中國_曝與習近平會晤日期_民視新聞__1080p.mp4",
      "score": 0.7102,
      "asset_url": "http://raptor_open_0_3_api.dhtsolution.com:8333/lakefs/data/...",
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

## RAG 查詢 (A2A Protocol)

### `POST /api/0.3/a2a/query`

完整 RAG pipeline：意圖分類 → 多路搜尋 → 重排序 → LLM 生成答案。  
回傳 LLM 生成的答案、來源片段、圖譜背景資訊。

支援三種執行模式：
- `direct`：確定性 pipeline（Module 18 分類 → 向量/關鍵字/圖譜搜尋 → Rerank → Ollama LLM）
- `agent`：smolagents CodeAgent 模式（LLM 自主決策呼叫哪些工具）
- `tool`：smolagents ToolCallingAgent 模式

> **注意：** 此端點需要 LLM 生成，回應時間通常 15–60 秒。建議搭配較長 timeout。

**Request Schema：**

| 欄位 | 型別 | 預設 | 說明 |
|------|------|------|------|
| `question` | string | 必填 | 問題（至少 1 個字元） |
| `top_k` | int | `5` | 搜尋使用的結果數（1–50） |
| `mode` | string | `"direct"` | `"direct"` / `"agent"` / `"tool"` |

**curl：**

```bash
curl -X POST http://raptor_open_0_3_api.dhtsolution.com:8012/api/0.3/a2a/query \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "川普和習近平在天壇見面時，記者問了什麼問題？",
    "top_k": 5,
    "mode": "direct"
  }'
```

**Python：**

```python
import requests

headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
resp = requests.post(
    "http://raptor_open_0_3_api.dhtsolution.com:8012/api/0.3/a2a/query",
    headers=headers,
    json={
        "question": "川普和習近平在天壇見面時，記者問了什麼問題？",
        "top_k": 5,
        "mode": "direct",
    },
    timeout=120,  # LLM 推理需要較長時間
)
resp.raise_for_status()
data = resp.json()
print("Answer:", data["answer"])
print(f"引用來源 {data['chunks_used']} 筆：")
for src in data.get("sources", [])[:3]:
    filename = src.get("metadata", {}).get("filename", "?")
    print(f"  score={src['score']:.4f} | {filename}")
    if src.get("storage_uri"):
        print(f"  URL: {src['storage_uri'][:80]}...")
```

**Response 範例：**

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
      "storage_uri": "http://raptor_open_0_3_api.dhtsolution.com:8333/lakefs/data/..."
    }
  ],
  "graph_context": "",
  "chunks_used": 3
}
```

---

<a id="processing"></a>

## 處理進度 (Processing)

上傳檔案後，AI 分析會非同步在背景執行。使用以下 API 查詢處理狀態。

`correlation_id` 從 `fileupload_analysis` 的回應 `upload_result.correlation_id` 取得。

---

### `GET /api/0.3/processing/cache/all`

查詢目前登入使用者所有上傳任務的處理進度。  
只回傳屬於目前使用者（相同 `branch_id`）的資料。

**Headers：** `Authorization: Bearer <token>`

**curl：**

```bash
curl http://raptor_open_0_3_api.dhtsolution.com:8012/api/0.3/processing/cache/all \
  -H "Authorization: Bearer <access_token>"
```

**Python：**

```python
import requests

headers = {"Authorization": f"Bearer {token}"}
resp = requests.get(
    "http://raptor_open_0_3_api.dhtsolution.com:8012/api/0.3/processing/cache/all",
    headers=headers,
)
resp.raise_for_status()
data = resp.json()
print(f"共 {data['count']} 個處理任務")
for cache_key, info in data.get("data", {}).items():
    print(f"  {cache_key}: step={info.get('step')}")
```

**Response 範例：**

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

### `GET /api/0.3/processing/cache/{m_type}/{key}`

查詢單一上傳任務的詳細處理進度。

**Headers：** `Authorization: Bearer <token>`  
**Path Params：**

| 參數 | 值 | 說明 |
|------|------|------|
| `m_type` | `document` / `video` / `image` / `audio` | 媒體類型 |
| `key` | UUID | 上傳時回傳的 `correlation_id` |

**處理步驟（`step` 欄位）：**

| step | 說明 |
|------|------|
| `queued` | 已加入佇列，等待處理 |
| `transcribing` | 語音轉文字中（影片/音訊） |
| `extracting` | 特徵提取中 |
| `indexing` | 向量化並建立索引中 |
| `complete` | 處理完成 |
| `failed` | 處理失敗 |

**curl：**

```bash
# 上傳後取得 correlation_id，再查詢進度
CORRELATION_ID="8d44cfc4-5a59-4be7-bc4d-f0992af5db75"

curl "http://raptor_open_0_3_api.dhtsolution.com:8012/api/0.3/processing/cache/video/${CORRELATION_ID}" \
  -H "Authorization: Bearer <access_token>"
```

**Python（輪詢等待完成）：**

```python
import requests
import time

def wait_for_completion(token: str, m_type: str, correlation_id: str, timeout: int = 600) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    url = f"http://raptor_open_0_3_api.dhtsolution.com:8012/api/0.3/processing/cache/{m_type}/{correlation_id}"
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.get(url, headers=headers)
        if resp.status_code == 404:
            print("任務尚未開始或已過期")
            time.sleep(5)
            continue
        resp.raise_for_status()
        data = resp.json()
        step = data.get("value", {}).get("step") or data.get("step")
        print(f"  目前進度: {step}")
        if step in ("complete", "failed"):
            return data
        time.sleep(10)
    raise TimeoutError("等待逾時")

# 使用範例
result = wait_for_completion(token, "video", "8d44cfc4-5a59-4be7-bc4d-f0992af5db75")
print("摘要:", result.get("value", {}).get("summary", "")[:100])
```

**Response 範例：**

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

<a id="fields"></a>

## 回應欄位說明

### Payload 欄位（搜尋結果中的 `payload`）

| 欄位 | 說明 |
|------|------|
| `type` | 媒體類型：`videos` / `documents` / `audios` / `images` |
| `source` | 原始格式：`mp4` / `pdf` / `csv` / `jpg` 等 |
| `filename` | 原始檔名 |
| `asset_path` | 在 LakeFS 的資產路徑 |
| `version_id` | 版本 ID（64 hex chars） |
| `chunk_index` | 文字片段索引（同一檔案按時間/頁面切分） |
| `start_time` / `end_time` | 影片片段時間點（秒，字串型別） |
| `speaker` | 說話者 ID（音訊分析後標記） |
| `text` | 四路文字：`contextual`（上下文摘要）、`ocr`（畫面文字）、`asr`（語音轉文字）、`lvlm`（視覺語言模型描述） |
| `asset_url` | Presigned 下載 URL（24 小時有效） |

### 常見 HTTP 狀態碼

| 狀態碼 | 說明 |
|--------|------|
| `200` | 成功 |
| `401` | 未認證（Token 無效或已過期） |
| `404` | 資源不存在 |
| `422` | 請求參數格式錯誤 |
| `429` | 請求頻率超限（Rate Limit） |
| `502` | 下游服務無法連線 |
| `504` | 下游服務回應逾時 |

### Rate Limits

| 端點 | 限制 |
|------|------|
| `fileupload_analysis` | 50 次/分鐘 |
| `fileupload_analysis_batch` | 10 次/分鐘 |

---

## 完整 Python 範例：上傳影片並等待處理完成後搜尋

```python
import requests
import time

BASE_URL = "http://raptor_open_0_3_api.dhtsolution.com:8012"

# 1. 登入
resp = requests.post(f"{BASE_URL}/api/0.3/sso/login",
                     data={"username": "test_basicuser", "password": "12345"})
token = resp.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# 2. 上傳影片
with open("川普習近平會面.mp4", "rb") as f:
    resp = requests.post(
        f"{BASE_URL}/api/0.3/asset/fileupload_analysis",
        headers=headers,
        files={"primary_file": f},
        data={"processing_mode": "default"},
    )
upload = resp.json()
version_id = upload["upload_result"]["version_id"]
processing = upload["processing_result"]
print(f"上傳完成 version_id={version_id[:16]}...")

if "status" in processing and processing["status"] == "skipped":
    print("檔案已存在，跳過處理")
else:
    correlation_id = processing["correlation_id"]  # 新檔案：在 processing_result 中
    # 3. 輪詢等待處理完成
    while True:
        resp = requests.get(
            f"{BASE_URL}/api/0.3/processing/cache/video/{correlation_id}",
            headers=headers,
        )
        if resp.status_code == 404:
            time.sleep(5)
            continue
        step = resp.json().get("value", {}).get("step")
        print(f"處理進度: {step}")
        if step in ("complete", "failed"):
            break
        time.sleep(10)

# 4. 搜尋
resp = requests.post(
    f"{BASE_URL}/api/0.3/search/hybrid",
    headers={**headers, "Content-Type": "application/json"},
    json={"query": "川普習近平 會面內容", "top_k": 5},
)
for r in resp.json()["results"]:
    print(f"score={r['score']:.4f} | {r['payload'].get('filename','?')}")

# 5. RAG 問答
resp = requests.post(
    f"{BASE_URL}/api/0.3/a2a/query",
    headers={**headers, "Content-Type": "application/json"},
    json={"question": "川普和習近平會面討論了哪些議題？", "top_k": 5, "mode": "direct"},
    timeout=120,
)
print("\nRAG 答案:", resp.json()["answer"])
```
