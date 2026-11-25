# Raptor API Reference

**Base URL:** `http://raptor_open_0_5_api.dhtsolution.com:8012`  
**API Version:** `0.5`  
**Authentication:** Bearer Token (Keycloak JWT)  

Swagger UI: `http://raptor_open_0_5_api.dhtsolution.com:8012/docs`

---

## 目錄

- [認證 (Single Sign-On)](#sso)
- [資產管理 (Asset)](#asset)
- [搜尋 (Search)](#search)
- [影片搜尋 (Video Search)](#video-search)
- [RAG 查詢 (A2A Protocol)](#rag)
- [推理服務 (Inference)](#inference)
- [對話 (Chat)](#chat)
- [記憶服務 (Memory)](#memory)
- [處理進度 (Processing)](#processing)
- [基準測試 (Benchmark)](#benchmark)
- [內容防護 (Guardrail)](#guardrail)
- [合規報表 (Compliance)](#compliance)
- [GDPR / CCPA](#gdpr)
- [文件代理 (Doc Agent)](#docagent)
- [內容審核 (Moderation)](#moderation)
- [AI/ML 模型與資料集 (AIML)](#aiml)
- [模型訓練 (Training)](#training)
- [人物身分 (Identity)](#identity)
- [時序推理 (Temporal)](#temporal)
- [回應欄位說明](#fields)

---

<a id="sso"></a>

## 認證 (Single Sign-On)

所有需要認證的 API 都必須在 Header 帶入 Bearer Token：

```
Authorization: Bearer <access_token>
```

---

### `POST /api/0.5/sso/login`

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
  "token_type": "bearer"
}
```

（這個端點不回傳 `refresh_token`——JWT 到期後要重新 `/sso/login`，不是 refresh。）

**curl：**

```bash
curl -X POST http://raptor_open_0_5_api.dhtsolution.com:8012/api/0.5/sso/login \
  -d "username=<your_username>&password=<your_password>"
```

**Python：**

```python
import requests

resp = requests.post(
    "http://raptor_open_0_5_api.dhtsolution.com:8012/api/0.5/sso/login",
    data={"username": "<your_username>", "password": "<your_password>"},
)
resp.raise_for_status()
token = resp.json()["access_token"]
print("Token:", token[:40], "...")
```

---

### `POST /api/0.5/sso/logout`

透過 Keycloak 登出目前 session。

> **注意：** 此端點需要瀏覽器登入時設定的 `realm_name` cookie，純 API 程式呼叫（不帶 cookie）會回傳 `400 {"detail": "Missing realm_name cookie"}`。  
> 帶正確 cookie 呼叫後（`200 {"message": "Logout successful, all cookies cleared"}`），同一顆 JWT access_token **立即**失效——後續任何帶這顆 token 的請求都會收到 `401 {"detail": "Session expired due to inactivity, please log in again (Invalid bearer token)"}`，不是等 `exp` 到期才失效。

**Headers：** `Authorization: Bearer <token>`  
**Cookie：** `realm_name=dhtsolution`（瀏覽器登入後自動設定）

**curl（帶 cookie）：**

```bash
curl -X POST http://raptor_open_0_5_api.dhtsolution.com:8012/api/0.5/sso/logout \
  -H "Authorization: Bearer <access_token>" \
  -b "realm_name=dhtsolution"
```

**Python（帶 cookie）：**

```python
import requests

headers = {"Authorization": f"Bearer {token}"}
resp = requests.post(
    "http://raptor_open_0_5_api.dhtsolution.com:8012/api/0.5/sso/logout",
    headers=headers,
    cookies={"realm_name": "dhtsolution"},
)
print(resp.status_code, resp.json())
```

---

<a id="asset"></a>

## 資產管理 (Asset)

### `POST /api/0.5/asset/fileupload_analysis`

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

`upload_result.status` 剛上傳完是 `"uploaded"`；AI 分析完成（見下方 `processing_result.correlation_id` 查詢）後才會變成 `"active"`（`fileversions`/`filedownload` 回傳的 `status` 欄位是這個最終值）。

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
    "user": "c9d6b4ab-...",
    "branch": "c9d6b4ab-...",
    "status": "uploaded",
    "checksum": "5ebc8e9b53b294f58346c2c5de1e7ecd",
    "moderation_flags": null,
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
curl -X POST http://raptor_open_0_5_api.dhtsolution.com:8012/api/0.5/asset/fileupload_analysis \
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
        "http://raptor_open_0_5_api.dhtsolution.com:8012/api/0.5/asset/fileupload_analysis",
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

### `POST /api/0.5/asset/fileupload_analysis_batch`

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
      "index": 0,
      "filename": "川普訪北京.mp4",
      "status": "success",
      "result": {"version_id": "...", "asset_path": "...", "status": "uploaded", "moderation_flags": null, "...": "其餘欄位同單檔上傳 upload_result"}
    }
  ],
  "upload_failures": [],
  "processing_results": [
    {
      "filename": "川普訪北京.mp4",
      "status": "kafka_sent",
      "kafka_result": {"message_id": "...", "correlation_id": "..."}
    }
  ]
}
```

（`kafka_result` 實際帶有 `message_id`/`correlation_id`，不是空物件——用這個 `correlation_id` 查詢處理進度，見下方「處理進度」章節。）

**curl：**

```bash
curl -X POST http://raptor_open_0_5_api.dhtsolution.com:8012/api/0.5/asset/fileupload_analysis_batch \
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
    "http://raptor_open_0_5_api.dhtsolution.com:8012/api/0.5/asset/fileupload_analysis_batch",
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

### `GET /api/0.5/asset/filedownload/{asset_path}/{version_id}`

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
    "status": "active",
    "moderation_flags": null
  },
  "primary_file": {
    "filename": "川普參觀天壇_習近平陪同_記者連三問是否談台灣_已讀不回__shorts__鏡新聞_1080p.mp4",
    "content_type": null,
    "version_id": "109fd79e263ae402334255f27660878b5ecc9eafe6a110e6a960c9f462d9cc37",
    "url": "http://raptor_open_0_5_api.dhtsolution.com:8333/lakefs/data/..."
  },
  "associated_file_0": {
    "filename": "frame_0001.jpg",
    "url": "http://raptor_open_0_5_api.dhtsolution.com:8333/lakefs/data/..."
  }
}
```

**curl：**

```bash
ASSET_PATH="video/mp4/川普參觀天壇_習近平陪同_記者連三問是否談台灣_已讀不回__shorts__鏡新聞_1080p"
VERSION_ID="109fd79e263ae402334255f27660878b5ecc9eafe6a110e6a960c9f462d9cc37"

curl -G "http://raptor_open_0_5_api.dhtsolution.com:8012/api/0.5/asset/filedownload/${ASSET_PATH}/${VERSION_ID}" \
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
    f"http://raptor_open_0_5_api.dhtsolution.com:8012/api/0.5/asset/filedownload/{asset_path}/{version_id}",
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

### `GET /api/0.5/asset/fileversions/{asset_path}/{filename}`

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
      "url": "http://raptor_open_0_5_api.dhtsolution.com:8333/lakefs/data/..."
    }
  ]
}
```

**curl：**

```bash
ASSET_PATH="video/mp4/川普參觀天壇_習近平陪同_記者連三問是否談台灣_已讀不回__shorts__鏡新聞_1080p"
FILENAME="川普參觀天壇_習近平陪同_記者連三問是否談台灣_已讀不回__shorts__鏡新聞_1080p.mp4"

curl "http://raptor_open_0_5_api.dhtsolution.com:8012/api/0.5/asset/fileversions/${ASSET_PATH}/${FILENAME}" \
  -H "Authorization: Bearer <access_token>"
```

**Python：**

```python
import requests

asset_path = "video/mp4/川普參觀天壇_習近平陪同_記者連三問是否談台灣_已讀不回__shorts__鏡新聞_1080p"
filename = "川普參觀天壇_習近平陪同_記者連三問是否談台灣_已讀不回__shorts__鏡新聞_1080p.mp4"

headers = {"Authorization": f"Bearer {token}"}
resp = requests.get(
    f"http://raptor_open_0_5_api.dhtsolution.com:8012/api/0.5/asset/fileversions/{asset_path}/{filename}",
    headers=headers,
)
resp.raise_for_status()
versions = resp.json()["versions"]
for v in versions:
    print(f"version_id: {v['version_id'][:16]}... | status: {v['status']} | date: {v['upload_date']}")
```

---

### `GET /api/0.5/asset/users/commits`

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
      "status": "active",
      "moderation_flags": null
    }
  ]
}
```

**curl：**

```bash
# 搜尋包含「川普」的資產
curl "http://raptor_open_0_5_api.dhtsolution.com:8012/api/0.5/asset/users/commits?keyword=川普&page=1&page_size=5" \
  -H "Authorization: Bearer <access_token>"
```

**Python：**

```python
import requests

headers = {"Authorization": f"Bearer {token}"}
resp = requests.get(
    "http://raptor_open_0_5_api.dhtsolution.com:8012/api/0.5/asset/users/commits",
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

### `POST /api/0.5/asset/filearchive/{asset_path}/{version_id}`

將指定版本標記為封存（archived）狀態。封存後仍可存取但不會出現在主要清單。

**Headers：** `Authorization: Bearer <token>`

**Response（回傳更新後的 metadata，`status` 變為 `"archived"`）：**

```json
{
  "asset_path": "video/mp4/川普啟動自由計劃",
  "version_id": "b02a0b42ba37f06cafeedd55e953762dc42988e9aa5163f877a6c170f08d0ebb",
  "primary_filename": "川普啟動自由計劃.mp4",
  "associated_filenames": [],
  "upload_date": "2026-05-18T21:52:45.843945+08:00",
  "archive_date": "2027-02-18T20:21:32.411915+08:00",
  "destroy_date": "2028-02-18T20:21:32.411915+08:00",
  "status": "archived",
  "moderation_flags": null
}
```

**curl：**

```bash
curl -X POST \
  "http://raptor_open_0_5_api.dhtsolution.com:8012/api/0.5/asset/filearchive/video%2Fmp4%2F川普啟動自由計劃/b02a0b42ba37f06cafeedd55e953762dc42988e9aa5163f877a6c170f08d0ebb" \
  -H "Authorization: Bearer <access_token>"
```

**Python：**

```python
import requests

asset_path = "video/mp4/川普啟動自由計劃"
version_id = "b02a0b42ba37f06cafeedd55e953762dc42988e9aa5163f877a6c170f08d0ebb"

headers = {"Authorization": f"Bearer {token}"}
resp = requests.post(
    f"http://raptor_open_0_5_api.dhtsolution.com:8012/api/0.5/asset/filearchive/{asset_path}/{version_id}",
    headers=headers,
)
resp.raise_for_status()
print(resp.json())
```

---

### `POST /api/0.5/asset/delfile/{asset_path}/{version_id}`

刪除指定版本的資產（從 LakeFS 移除）。此操作不可逆。

> **前提：** 資產必須先呼叫 `filearchive` 變成 `"archived"` 狀態才能刪除。對一個還是 `"active"`／`"uploaded"` 狀態的資產直接呼叫會回傳 `400 {"detail": "Asset service error: {\"detail\":\"Asset <path>/<version_id> is not archived\"}"}`。

**Headers：** `Authorization: Bearer <token>`

**Response（回傳更新後的 metadata，`status` 變為 `"destroyed"`——不是刪除整筆記錄）：**

```json
{
  "asset_path": "video/mp4/川普啟動自由計劃",
  "version_id": "b02a0b42ba37f06cafeedd55e953762dc42988e9aa5163f877a6c170f08d0ebb",
  "primary_filename": "川普啟動自由計劃.mp4",
  "associated_filenames": [],
  "upload_date": "2026-05-18T21:52:45.843945+08:00",
  "archive_date": "2027-02-18T20:21:32.411915+08:00",
  "destroy_date": "2028-02-18T20:21:32.411915+08:00",
  "status": "destroyed",
  "moderation_flags": null
}
```

**curl：**

```bash
curl -X POST \
  "http://raptor_open_0_5_api.dhtsolution.com:8012/api/0.5/asset/delfile/video%2Fmp4%2F川普啟動自由計劃/b02a0b42ba37f06cafeedd55e953762dc42988e9aa5163f877a6c170f08d0ebb" \
  -H "Authorization: Bearer <access_token>"
```

**Python：**

```python
import requests

asset_path = "video/mp4/川普啟動自由計劃"
version_id = "b02a0b42ba37f06cafeedd55e953762dc42988e9aa5163f877a6c170f08d0ebb"

headers = {"Authorization": f"Bearer {token}"}
resp = requests.post(
    f"http://raptor_open_0_5_api.dhtsolution.com:8012/api/0.5/asset/delfile/{asset_path}/{version_id}",
    headers=headers,
)
resp.raise_for_status()
print(resp.json())
```

---

### `POST /api/0.5/asset/file-expiration/{asset_path}/{version_id}`

更新資產的封存日期或刪除日期。可傳入 ISO 8601 日期字串或天數（TTL）。

> **注意：** 若資產上傳時未設定任何過期日（permanent 資產），`archive_date_or_ttl` 與 `destroy_date_or_ttl` 必須**同時提供**，否則回傳 `400`。

**Headers：** `Authorization: Bearer <token>`  
**Content-Type：** `application/x-www-form-urlencoded`

| 參數 | 型別 | 說明 |
|------|------|------|
| `archive_date_or_ttl` | int \| string | 封存日期（`"2026-12-31"`）或天數（`30`） |
| `destroy_date_or_ttl` | int \| string | 刪除日期或天數 |

**Response（回傳更新後的 metadata）：**

```json
{
  "asset_path": "video/mp4/川普啟動自由計劃",
  "version_id": "b02a0b42ba37f06cafeedd55e953762dc42988e9aa5163f877a6c170f08d0ebb",
  "primary_filename": "川普啟動自由計劃.mp4",
  "associated_filenames": [],
  "upload_date": "2026-05-18T21:52:45.843945+08:00",
  "archive_date": "2027-02-18T20:21:32.411915+08:00",
  "destroy_date": "2028-02-18T20:21:32.411915+08:00",
  "status": "active",
  "moderation_flags": null
}
```

**curl：**

```bash
curl -X POST \
  "http://raptor_open_0_5_api.dhtsolution.com:8012/api/0.5/asset/file-expiration/video%2Fmp4%2F川普啟動自由計劃/b02a0b42ba37f06cafeedd55e953762dc42988e9aa5163f877a6c170f08d0ebb" \
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
    f"http://raptor_open_0_5_api.dhtsolution.com:8012/api/0.5/asset/file-expiration/{asset_path}/{version_id}",
    headers=headers,
    data={"archive_date_or_ttl": "90", "destroy_date_or_ttl": "365"},
)
resp.raise_for_status()
print(resp.json())
```

---

### `POST /api/0.5/asset/fileupload`

只上傳檔案到儲存，**不觸發**任何 AI 分析。想批次上傳、之後再決定哪些要送分析，用這支搭配下面的
`POST /processing/process-file`；想上傳後立刻分析，用上面的 `POST /asset/fileupload_analysis`。

**Headers：** `Authorization: Bearer <token>`  
**Content-Type：** `multipart/form-data`  
**Rate limit：** 100 requests/minute

| 參數 | 型別 | 必填 | 說明 |
|------|------|------|------|
| `primary_file` | file | ✓ | 要上傳的檔案 |
| `archive_date_or_ttl` | int \| string | | 封存日期或 TTL 天數 |
| `destroy_date_or_ttl` | int \| string | | 刪除日期或 TTL 天數 |

**Response：** 跟上面 `fileupload_analysis` 的 `upload_result` 物件同一個形狀，但**沒有外層包裝、沒有
`processing_result`**（因為沒有觸發分析）。`status` 會停在 `"uploaded"`，直到用下面的
`POST /processing/process-file` 顯式觸發分析。

需要附加檔案？先用這支傳主檔案，再呼叫 `POST /add-associated-files/{asset_path}`（表單裡放一個
`List[UploadFile]` 欄位在這個 gateway 的 multipart 代理下只會留下最後一個檔案，所以這裡不接受內嵌多檔，
專用端點沒有這個問題）。

---

### `POST /api/0.5/asset/scan`

批次檢查一批 `(path, checksum)` 是否已經存在，**不上傳**任何東西。用來在真正上傳前先知道哪些檔案是重複的。

**Request Schema：** `files`（陣列，每項 `{"path": string, "checksum": string}`）。空陣列回傳空結果，不是錯誤；
checksum 格式不對回 `400`。

**Response 範例：**

```json
{"path/to/file.mp4": {"exists": true, "existing_asset_path": "...", "existing_version_id": "..."}}
```

---

### `POST /api/0.5/asset/unregister/{asset_path}/{version_id}`

把一個 `active` 版本的狀態改成 `unregistered`，從搜尋/索引結果中移除，但**不刪除**任何 embedding、
vector 或 graph 資料（filter-only）。只能從 `active` 狀態呼叫，否則 `409`。

**可逆**：呼叫下面的 `POST /processing/process-file`（帶同一個 `asset_path`/`version_id`）即可快速切回
`active`，不會重新分析、不會發 Kafka 訊息。**沒有獨立的 `POST /register` 路由**（module 04 內部確實有一個
同名端點，但那是 `process-file` 內部呼叫用的，刻意不對外公開，直接打會 404）。

---

### `GET /api/0.5/asset/thumbnail/{asset_path}/{version_id}`

用 NVDEC 硬體解碼，直接從影片的 presigned URL 用 HTTP Range 抓單一畫面（不下載整個檔案），回傳一張 JPEG。
同一個 `asset_path`/`version_id`/`second` 組合快取 7 天。

**Query 參數：** `second`（必填，要擷取的時間點，秒）

**錯誤情況：** 資產/版本不存在 → `404`；資產不是 `active` → `409`；`second` 超過影片長度 → `400`

**Response：** `image/jpeg`，不是 JSON。

---

<a id="search"></a>

## 搜尋 (Search)

所有搜尋端點皆需認證。`hybrid`/`bm25`/`vector` 回傳 `{"results": [...], "timing": {...}}`；`tkg`/`graphrag` 是不同的回應形狀（見各自章節），沒有 `results` 陣列。這些端點都是每位使用者各自獨立的索引（Module 25，ArcadeDB 後端），不是全站共用索引。

每筆結果 `payload` 中的 `asset_url` 為資產的 presigned 下載連結（24 小時有效；下方範例已省略實際簽章字串）。

### 共用 Request Schema（`hybrid`/`bm25`/`vector` 共用）

| 欄位 | 型別 | 預設 | 說明 |
|------|------|------|------|
| `query` | string | 必填 | 搜尋文字 |
| `top_k` | int | `10` | 最多回傳幾筆 |
| `embedding_type` | string | null | `"text"` 或 `"summary"` |
| `type` | string \| string[] | null | 資產類型過濾，如 `"videos"`、`["documents","audios"]` |
| `filename` | string[] | null | 依檔名過濾 |
| `speaker` | string[] | null | 依說話者過濾（影片/音訊） |
| `source` | string | null | 依來源格式過濾，如 `"pdf"`、`"mp4"` |

---

### `POST /api/0.5/search/hybrid`

混合搜尋（BM25 + Vector 融合），結果經 RRF 排序後再重排序（rerank）。  
適合大多數一般搜尋情境。

**curl：**

```bash
curl -X POST http://raptor_open_0_5_api.dhtsolution.com:8012/api/0.5/search/hybrid \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "川習會 貿易談判",
    "top_k": 5
  }'
```

**curl（加過濾條件）：**

```bash
curl -X POST http://raptor_open_0_5_api.dhtsolution.com:8012/api/0.5/search/hybrid \
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
    "http://raptor_open_0_5_api.dhtsolution.com:8012/api/0.5/search/hybrid",
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
      "score": 0.500029,
      "payload": {
        "version_id": "79626f7d-437e-4137-8a5d-3b0805172a18",
        "summary": "（AI 生成的整體摘要）...",
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

**注意：** `payload` 內的欄位依 `embedding_type` 而異——`"summary"` 型結果沒有 `chunk_index`/`start_time`/`end_time`/`speaker`/`status`，`"text"` 型結果才有這些逐段資訊。兩種都**沒有** `branch_id` 欄位（舊版文件曾誤植此欄位）。`score` 是 RRF 融合後再經 rerank 的分數，落在 0.5 附近，不是 0–1 的相似度。

---

### `POST /api/0.5/search/bm25`

純 BM25（關鍵字）搜尋，每位使用者各自獨立的 ArcadeDB 索引（Module 25）。  
適合精確關鍵字比對，不使用語意向量。Score 為 BM25 分數（越高越相關）。

**curl：**

```bash
curl -X POST http://raptor_open_0_5_api.dhtsolution.com:8012/api/0.5/search/bm25 \
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
    "http://raptor_open_0_5_api.dhtsolution.com:8012/api/0.5/search/bm25",
    headers=headers,
    json={"query": "川普 貿易關稅", "top_k": 5},
)
resp.raise_for_status()
for r in resp.json()["results"]:
    print(f"BM25 score={r['score']:.2f} | {r['payload'].get('filename','?')}")
```

**Response 範例：**

```json
{
  "results": [
    {
      "id": "79626f7d-437e-4137-8a5d-3b0805172a18",
      "score": 0.7911,
      "payload": {
        "version_id": "79626f7d-437e-4137-8a5d-3b0805172a18",
        "summary": "（AI 生成的整體摘要）...",
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

無符合結果時 `results` 為空陣列（`[]`），非 404。

---

### `POST /api/0.5/search/vector`

純向量（語意）搜尋，每位使用者各自獨立的 ArcadeDB 索引（Module 25）。  
適合語意相似搜尋，score 為餘弦相似度（0–1）。

**curl：**

```bash
curl -X POST http://raptor_open_0_5_api.dhtsolution.com:8012/api/0.5/search/vector \
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
    "http://raptor_open_0_5_api.dhtsolution.com:8012/api/0.5/search/vector",
    headers=headers,
    json={"query": "川普與習近平 AI 科技競爭", "top_k": 5},
)
resp.raise_for_status()
for r in resp.json()["results"]:
    print(f"cosine={r['score']:.4f} | {r['payload'].get('filename','?')}")
```

**Response 範例：**

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

### `POST /api/0.5/search/graphrag`

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
curl -X POST http://raptor_open_0_5_api.dhtsolution.com:8012/api/0.5/search/graphrag \
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
    "http://raptor_open_0_5_api.dhtsolution.com:8012/api/0.5/search/graphrag",
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
      "entity_id": "xi_jinping",
      "name": "習近平",
      "type": "PERSON",
      "description": "中華人民共和國最高領導人",
      "node_kind": "entity",
      "mention_count": 8,
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

### `POST /api/0.5/search/tkg`

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
curl -X POST http://raptor_open_0_5_api.dhtsolution.com:8012/api/0.5/search/tkg \
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
curl -X POST http://raptor_open_0_5_api.dhtsolution.com:8012/api/0.5/search/tkg \
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
    "http://raptor_open_0_5_api.dhtsolution.com:8012/api/0.5/search/tkg",
    headers=headers,
    json={"query": "川習峰會時間序列", "max_depth": 2, "limit": 20},
)
resp.raise_for_status()
data = resp.json()
for e in data.get("matched_entities", [])[:5]:
    print(f"  [{e['type']}] {e['name']} | {e.get('description','')[:60]}")
print("temporal_facts:", len(data.get("temporal_facts", [])))
```

**Response 範例（含 `retrieval`/`complete`/`version`/`sort`/`cursor`/`total`/`reference` 欄位，以及展開的 `temporal_facts[]` 實際欄位）：**

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
  "temporal_facts": [
    {
      "fact_id": "tf-9ce1ec9b88b2620b1dcbf16e9218c346",
      "entity": "黃仁勳",
      "entity_id": "黃仁勳",
      "relation": "出生於",
      "value": "1963年",
      "time_start": "1963-01-01",
      "time_end": "1963-12-31",
      "confidence": 0.7,
      "source_version_id": "dc248af9e8ccec29fef746ac4bc0c642a35db6a66ba7102ac63c1601f757e34c",
      "precision": "year",
      "date_text": "1963年",
      "source_sentence": "...（抽取來源句）",
      "anchor": "explicit",
      "derived": null,
      "support_ids": null,
      "when": "1963"
    }
  ],
  "moment_ids": [...],
  "retrieval": "temporal",
  "complete": true,
  "version": 1,
  "sort": null,
  "cursor": null,
  "total": null,
  "reference": null
}
```

`sort`/`cursor`/`total`/`reference` 目前皆為 `null`——推測是保留給未來分頁功能的欄位，尚未啟用。

---

<a id="video-search"></a>

## 影片搜尋 (Video Search)

### `POST /api/0.5/search/video_search`

專為影片設計的搜尋端點。先以多路召回（BM25 + Vector + GraphRAG + TKG）取得候選片段（chunk），經 RRF 融合後再以 cross-encoder rerank，最後以影片為單位聚合回傳，每部影片附帶最相關的時間片段列表。

**Request Schema：**

| 欄位 | 型別 | 預設 | 說明 |
|------|------|------|------|
| `query` | string | 必填 | 自然語言搜尋 |
| `top_k` | int | `10` | 最多回傳幾部影片（1–50） |
| `candidate_multiplier` | int | `5` | 每個召回器取 `top_k × N` 候選片段（1–20） |
| `score_threshold` | float | `0.52` | 最低分數過濾（0–1），低於此值的片段不顯示 |

**curl：**

```bash
curl -X POST http://raptor_open_0_5_api.dhtsolution.com:8012/api/0.5/search/video_search \
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
    "http://raptor_open_0_5_api.dhtsolution.com:8012/api/0.5/search/video_search",
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
      "version_id": "d9243fa19b59c4daf793d77ebe31ba91433b2826d8185e4657be2cd8fd3a52d2",
      "asset_path": "video/mp4/川普強調美國AI領先中國_曝與習近平會晤日期_民視新聞__1080p",
      "filename": "川普強調美國AI領先中國_曝與習近平會晤日期_民視新聞__1080p.mp4",
      "score": 0.7102,
      "asset_url": "http://raptor_open_0_5_api.dhtsolution.com:8333/lakefs/data/...",
      "upload_time": "2026-05-18T...",
      "segments": [
        {
          "start_time": 0.0,
          "end_time": 10.0,
          "score": 0.7102,
          "text": "contextual:{...} / ocr:{...} / asr:{...} / lvlm:{...}",
          "sources": ["bm25", "vector"]
        }
      ],
      "people": []
    }
  ]
}
```

（含 `version_id`/`asset_path`/`people` 三個欄位——`video_id` 與 `version_id` 值相同。`people` 是這部影片裡、跟 `query` 同時比對到名字的已知身分（People Search）出現片段，範圍是「RRF 候選影片」∪「單純用姓名比對到的影片」的聯集，不受 `candidate_multiplier`/`top_k` 限制（這個聯集內的比對沒有上限，不會因為某個身分在平台上出現次數多就被截斷）；沒有比對到任何已知身分時為空陣列。）

---

### `POST /api/0.5/search/people`

依姓名／別名搜尋人物，回傳這個人出現過的每一段連續片段（不是整部影片）。已 unregister 或封存的影片不會出現在結果裡。

**需要 `face_detection` 的 GDPR 同意**（見 [GDPR / CCPA](#gdpr) 章節）——沒有同意會回 `403 {"error": "PERMISSION_DENIED", "message": "face_detection consent not granted"}`；同意狀態查不到時是 `503 CONSENT_CHECK_UNAVAILABLE`。

**Request Schema：** `query`（必填，人名或別名，整段或部分都可）、`limit`（預設 `50`，1–500，片段數上限）。

**Response 範例：**

```json
{
  "query": "黃仁勳",
  "identities": [{"identity_id": "f1a0a1ff...", "name": "黃仁勳", "matched": "黃仁勳", "exact": true}],
  "total": 48,
  "results": [
    {
      "identity_id": "f1a0a1ff...",
      "identity_name": "黃仁勳",
      "version_id": "30e328c2...",
      "trace_id": "0dc1da5d...",
      "first_sec": 0.0,
      "last_sec": 3.24,
      "confidence": 0.977,
      "keyframe": {"face_id": "rep_0dc1da5d...", "image_url": "<presigned>"},
      "asset_path": "video/mp4/...",
      "filename": "....mp4",
      "asset_url": "<presigned>"
    }
  ]
}
```

`identities` 是比對到的人物（`exact` 代表 `query` 整段剛好等於某個名字/別名，不只是部分匹配）；`results` 每筆是一段連續出現（`first_sec`–`last_sec` 是這段的播放範圍，不是整部影片），`total` 是 `limit` 篩選前的片段總數，不是 `results` 陣列長度。

---

<a id="rag"></a>

## RAG 查詢 (A2A Protocol)

### `POST /api/0.5/a2a/query`

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
curl -X POST http://raptor_open_0_5_api.dhtsolution.com:8012/api/0.5/a2a/query \
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
    "http://raptor_open_0_5_api.dhtsolution.com:8012/api/0.5/a2a/query",
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
      "storage_uri": "http://raptor_open_0_5_api.dhtsolution.com:8333/lakefs/data/..."
    }
  ],
  "graph_context": "",
  "chunks_used": 3,
  "temporal_claims": [],
  "people": [],
  "temporal_status": "NO_TEMPORAL",
  "temporal_reason": null,
  "temporal_constraint_applied": false
}
```

（含後面 5 個欄位——`temporal_claims`/`people` 沒比對到時為空陣列，`temporal_status` 問題沒有時間限定詞時是 `"NO_TEMPORAL"`，`temporal_constraint_applied` 對應是否真的套用了時間過濾。`graph_context` 只有在 GraphRAG 比對到相關實體時才會是非空字串，不是每次都是空字串。）

---

<a id="inference"></a>

## 推理服務 (Inference)

代理 Module 35（inference-service）。跟[文件代理 (Doc Agent)](#docagent)那支通用反向代理不同——這裡每支路由各自宣告在這個閘道的 Swagger 上（方便直接看到真實的 request/response 形狀），但實際驗證/處理邏輯全部由 Module 35 自己決定，這層只做標準 Bearer 認證跟原樣轉發。

### 文字推理

`translate`/`summarize`/`complete`/`transform` 四支共用同一個回應形狀（除了各自多帶的欄位）：

| 欄位 | 型別 | 說明 |
|------|------|------|
| `task` | string | 對應的任務名稱 |
| `model` | string | 實際使用的模型名稱 |
| `model_version` | string \| null | MLflow 版本（非 MLflow 來源的模型是 `null`） |
| `output` | string | 生成結果 |
| `usage` | object | `prompt_tokens`/`completion_tokens`/`total_tokens` |
| `latency_ms` | int | 這次推理花費的時間（毫秒） |

### `POST /api/0.5/inference/text/translate`

**Request：** `input`（必填）、`options.target_language`（必填）、`options.source_language`（選填，省略則自動偵測，回應會多一個 `detected_language` 欄位）。

**Response 範例：**

```json
{"task":"translate","model":"gemma4:12b","model_version":null,"output":"Good morning","usage":{"prompt_tokens":83,"completion_tokens":3,"total_tokens":86},"latency_ms":13076,"target_language":"en","source_language":"zh-TW"}
```

### `POST /api/0.5/inference/text/summarize`

**Request：** `input`（必填）、`options.style`（如 `"bullet"`）、`options.max_sentences`。

**Response 範例（多帶 `style`/`chunks` 兩個欄位）：**

```json
{"task":"summarize","model":"gemma4:12b","output":"- 愛因斯坦是20世紀最重要的物理學家之一，於1879年出生於德國。\n- 他提出了相對論，對現代物理學產生了深遠的影響。","usage":{"prompt_tokens":137,"completion_tokens":43,"total_tokens":180},"latency_ms":13148,"style":"bullet","chunks":1}
```

### `POST /api/0.5/inference/text/complete`

**Request：** `input`（必填）、`options.max_tokens`、`options.temperature`。純文字生成，不是對話，不會帶入任何先前歷史。

### `POST /api/0.5/inference/text/transform`

**Request：** `input`（必填）、`options.instruction`（必填，如 `"改寫成正式語氣"`）。

**Response 範例：**

```json
{"task":"transform","model":"gemma4:12b","output":"請問您用餐了嗎？","usage":{"prompt_tokens":65,"completion_tokens":6,"total_tokens":71},"latency_ms":12806}
```

### `POST /api/0.5/inference/text`

通用 dispatcher：body 多帶一個 `task`（`translate`/`summarize`/`complete`/`transform` 之一），其餘欄位跟對應的獨立端點完全相同，回應形狀也相同（走 `translate` 時一樣會帶 `detected_language`）。

### `POST /api/0.5/inference/text/complete/stream`

SSE 版的 `complete`，同樣的 `input`/`options`。**Content-Type：** `text/event-stream`。

**frame 格式：** 每個 token 一個 `data: {"token": "...", "finish_reason": null}`；最後一個 frame 的 `finish_reason` 會是非 null 值，接著一個 `data: [DONE]` 的 sentinel frame。這個閘道只負責原樣轉發串流，不重新解析內容。

### 語音

### `POST /api/0.5/inference/voice/diarize`

Speaker diarization（WhisperX）。**Content-Type：** `multipart/form-data`。

| 參數 | 型別 | 必填 | 說明 |
|------|------|------|------|
| `audio` | file | ✓ | WAV / MP3 / FLAC |
| `model` | string | | 預設 `"pyannote-3.1"` |
| `num_speakers` | int | | 1–64 |
| `language` | string | | BCP-47，省略自動偵測 |

**Response 範例：**

```json
{"speakers":["SPEAKER_00"],"segments":[{"speaker":"SPEAKER_00","text":"請問明天早上台北會不會下雨？","start":0.715,"end":3.414}],"latency_ms":22627.0}
```

### `POST /api/0.5/inference/voice/stt`

語音轉文字（轉發到 Module 34 已載入的 Whisper 模型）。**Request：** multipart，`audio`（必填）、`language`（選填，BCP-47）。

**Response 範例：**

```json
{"text":"請問明天早上臺北會不會下雨","language":"zh","language_probability":null,"confidence":1.0}
```

### `POST /api/0.5/inference/voice/tts`

文字轉語音（轉發到 Module 34 已載入的 TTS 模型）。**Request：** multipart，`text`（必填）、`language`（選填，BCP-47）。**Response：** `audio/wav` 二進位內容。

### `POST /api/0.5/inference/voice/clone`

聲音克隆合成——目前有兩個可選後端：`xtts-v2`、`chatterbox-multilingual-v3`（沒指定 `model` 時用部署的預設後端）。**Content-Type：** `multipart/form-data`。

| 參數 | 型別 | 必填 | 說明 |
|------|------|------|------|
| `reference_audio` | file | ✓ | WAV / MP3 / FLAC，長度 1–60 秒 |
| `text` | string | ✓ | 1–3000 字 |
| `language` | string | ✓ | BCP-47 |
| `speaker_name` | string | | 1–128 字，重複使用同名可跳過重新編碼 |
| `model` | string | | `xtts-v2` 或 `chatterbox-multilingual-v3`，省略用部署預設 |

**需要 `voice_clone` 的 GDPR 同意**（見 [GDPR / CCPA](#gdpr) 章節）——沒有同意會回 `403 consent_required`，在讀取音檔之前就會擋下來。

**Response：** `audio/wav` 二進位內容，外加兩個 Header：`X-Speaker-Id`（這次克隆用的 speaker 快取 id，多後端部署會帶後端名稱前綴，如 `chatterbox-multilingual-v3.<speaker_name>`）、`X-Watermarked`（`true`/`false`——Chatterbox 的輸出永遠會加水印，XTTS 不會）。

### `GET /api/0.5/inference/voice/clone/speakers`

列出目前使用者快取的 speaker embedding。**Response：** `{"speakers": [{"id": "...", "ttl_remaining_s": 86390}]}`。

### `DELETE /api/0.5/inference/voice/clone/speakers/{speaker_id}`

移除一個 speaker embedding（204，無 body）。

### 模型管理

### `GET /api/0.5/inference/models`

列出 Module 07 MLflow registry + Ollama 的模型（含 Module 35 自己內建的聲音克隆後端），結果會快取一段時間。

**Response 範例：**

```json
{
  "models": [
    {"name": "chatterbox-multilingual-v3", "type": "clone", "size_gb": 1.0, "capabilities": ["clone"], "source": "builtin", "status": "available", "watermarked": true},
    {"name": "xtts-v2", "type": "clone", "size_gb": 1.8, "capabilities": ["clone"], "source": "builtin", "status": "available", "watermarked": false},
    {"name": "gemma4:12b", "type": "llm", "size_gb": 7.47, "capabilities": ["translate", "summarize", "complete", "transform"], "source": "ollama", "status": "loaded", "runtime": "ollama", "runtime_model": "gemma4:12b", "watermarked": false}
  ],
  "count": 8,
  "sources": {"mlflow": "ok", "ollama": "ok"}
}
```

`version`/`stage`（MLflow 專屬）、`runtime`/`runtime_model`（非 MLflow 來源時用來標示實際呼叫的 runtime tag）對不適用的來源是 `null`；`watermarked` 只對 clone 類模型有意義，其餘一律 `false`。

### `GET /api/0.5/inference/models/default`

目前每個任務類型的預設模型。**Response 範例：**

```json
{"text": "gemma4:12b", "stt": null, "tts": null, "clone": "chatterbox-multilingual-v3", "diarize": null}
```

`null` 代表那個任務類型目前沒有設定明確的預設（走各自的 factory-default 邏輯）。

### `POST /api/0.5/inference/models/default`

設定某個任務類型的預設模型（存進 Redis，立即生效）。**Request：** `{"task": "text", "model": "gemma-prod"}`。成功 `204 No Content`。

### `GET /api/0.5/inference/models/{name}`

取得單一模型的 registry 項目，形狀同 `GET /models` 陣列裡的一筆。

### `GET /api/0.5/inference/health`

Module 35 自己的健康檢查，透傳不做任何改寫。**Response 範例：**

```json
{"status":"ok","module":35,"gpu_available":true,"gpu_name":"NVIDIA GB10","module07_reachable":true,"module34_reachable":true,"loaded_voice_models":["pyannote-whisperx"],"voice":{"diarize_loaded":true,"clone_loaded":true,"clone_configured_backends":["chatterbox-multilingual-v3","xtts-v2"]},"warnings":["Available GPU VRAM is below 12 GiB"]}
```

`voice.clone_loaded`/`loaded_voice_models` 反映的是「目前已經載入到 GPU 記憶體」的狀態，不是「有沒有設定」——clone 後端是懶載入的，部署剛啟動、還沒人呼叫過 `/voice/clone` 時可能是空的，即使 `clone_configured_backends` 列著後端名稱。

---

<a id="chat"></a>

## 對話 (Chat)

Module 15（chat-service）代理，多輪對話搭配自動 RAG 搜尋，記憶存在 Redis。

### `POST /api/0.5/chat/chat`

發送一則訊息，自動判斷是否需要 RAG 搜尋，並用 LangGraph pipeline 產生回覆。

**Pipeline：** load_memory → prepare_context → call_model（可呼叫工具）→ execute_tools（視情況）→ save_memory。

**自動觸發搜尋的關鍵字：** `搜尋 / 找 / pdf / 文件 / 圖片 / 影片 / 音訊`，或訊息以 `？`/`?` 結尾。

**Request Schema：**

| 欄位 | 型別 | 預設 | 說明 |
|------|------|------|------|
| `message` | string | 必填 | 這一輪的使用者訊息 |
| `history` | `[{role, content}]` | null | 這一輪之前的對話歷史；省略則只用 Redis 裡已存的記憶 |
| `session_id` | string | null | 用來隔離不同對話情境的記憶；省略則用預設 session |

`user_id` 一律從 Bearer token 自動取得，不需要（也不能）手動傳。

**curl：**

```bash
curl -X POST http://raptor_open_0_5_api.dhtsolution.com:8012/api/0.5/chat/chat \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"message": "你好，簡單自我介紹一下你能做什麼"}'
```

**Response 範例：**

```json
{
  "response": "您好！我是有幫助的 AI 助手！我主要可以協助您以下事情：\n\n💡 搜尋相關資料 - 跨多個資料庫、檔案、影片、音訊和圖片進行統一搜尋...",
  "user_id": "c9d6b4ab-...",
  "session_id": null,
  "search_triggered": false,
  "search_results": null,
  "tool_calls": []
}
```

`search_triggered=true` 時，`search_results` 會是命中片段陣列（形狀同 [搜尋 (Search)](#search) 章節），`tool_calls` 會列出這輪呼叫過的工具（`unified_search_tool`、`calculate`、`get_current_time` 等）。

---

### `GET /api/0.5/chat/memory`

讀取目前使用者的對話記憶（Module 15 自己的短期 Redis 快取，`chat_memory:*`，`MEMORY_TTL` 到期即消失）。

**這不是** Module 26（記憶服務）的長期事實/偏好、可搜尋的 session 歷史或多媒體索引——那些請改用下面[記憶服務 (Memory)](#memory) 章節的 `/memory/retrieve` 或 `/memory/timeline`。

**Query 參數：** `session_id`（選填，省略則讀預設 session）

**Response 範例：**

```json
{
  "user_id": "c9d6b4ab-...",
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

### `DELETE /api/0.5/chat/memory`

清空目前使用者的短期對話上下文（同上 Redis 快取），用於「開始新對話」。`?session_id=xxx` 只清指定 session，省略則清預設 session。成功回傳 `204 No Content`。

**這不會**刪除 Module 26 儲存的任何東西（長期事實/偏好、已歸檔的 session 歷史、多媒體記憶）——那些不會過期，需要真正的 GDPR 抹除請用 `DELETE /api/0.5/memory`（見下方）。

---

### `POST /api/0.5/chat/completions`

Gateway 層的 OpenAI 相容 `chat/completions` 代理，轉發到 Module 07 的 `/v1/chat/completions`。純代理，不支援 streaming（`stream` 一律轉發為 `false`——`GuardrailMiddleware` 若啟用，需要完整回應內容才能做生成後的政策檢查，SSE 部分串流無法提前放行）。

**Request：** 標準 OpenAI `chat/completions` 格式，至少需要 `model` 與 `messages`；`model` 必須是已在 Module 07 MLflow registry 註冊的模型名稱（`GET /api/0.5/aiml/models/local` 可查目前有哪些）。

```bash
curl -X POST http://raptor_open_0_5_api.dhtsolution.com:8012/api/0.5/chat/completions \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"model": "<已註冊的模型名稱>", "messages": [{"role": "user", "content": "你好"}]}'
```

**常見錯誤（模型未註冊時，HTTP 404）：**

```json
{
  "detail": {"error": {"message": "model '<name>' not in MLflow registry", "type": "invalid_request_error", "code": "model_not_found"}}
}
```

**呼叫未註冊的模型：** 帶上 `engine`（例如 `"engine": "ollama"`）可以繞過 MLflow registry 查詢，直接用該 runtime 上的原生 tag 名稱呼叫，不用先註冊：

```bash
curl -X POST http://raptor_open_0_5_api.dhtsolution.com:8012/api/0.5/chat/completions \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5:7b", "engine": "ollama", "messages": [{"role": "user", "content": "你好"}]}'
```

**Response 範例（帶 `engine` 成功繞過未註冊的模型）：**

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

## 記憶服務 (Memory)

Module 26（memory-service）代理，長期事實/偏好、可跨 session 搜尋的對話歷史，以及多媒體記憶。與上方 [對話 (Chat)](#chat) 章節的短期 Redis 快取是**不同的儲存**，不會隨對話快取過期而消失。

閘道實際對外只保留 6 支扁平化路由（`store`/`retrieve`/`timeline`/`compact`/`archive` + 下面的 `DELETE`），外加 2 支保留給 Module 21（agent-protocol）呼叫的端點（`compact/evaluate`、`sessions/{id}/summaries`）。原本更完整的 session-scoped 介面大多已停用（程式碼保留、註解掉，之後有需要可以再開）。

### `POST /api/0.5/memory/store`

寫入一則不綁定任何 session 的獨立記憶節點（事實／偏好／實體）。要記錄完整一輪對話請用 `/memory/archive`。

**Request Schema：** `text`（必填，要記住的內容）、`frame_type`（`conversation`/`preference`/`entity`/`fact`，預設 `fact`）、`session_id`（選填，僅作追溯用途）。

**Response 範例（重複寫入同一內容）：**

```json
{"frame_id": "14", "frame_type": "preference", "timestamp": 1787485472.93, "status": "duplicate", "duplicate_of": "14"}
```

`status` 首次寫入會是 `"created"`（幂等去重機制，同一使用者對同樣內容重複寫入會回報 `"duplicate"`）。

---

### `GET /api/0.5/memory/retrieve`

跨所有 session、長期記憶與多媒體記憶做混合搜尋（語意 + BM25）。

**Query 參數：** `query`（必填）、`top_k`（預設 `5`，1–50）

**Response 範例：**

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
  ],
  "longterm": [
    {
      "text": "使用者偏好繁體中文回覆",
      "score": 0.72,
      "timestamp": "2026-08-25T05:40:00.00Z",
      "frame_type": "preference",
      "session_id": ""
    }
  ],
  "multimedia": [],
  "total_frames_searched": 5
}
```

`sessions`/`longterm`/`multimedia` 三個都是獨立的結果列表（同一次混合搜尋拆開回傳，不是巢狀在一起），`total_frames_searched` 是這次搜尋實際掃過的 frame 總數（三個來源加總）。

---

### `GET /api/0.5/memory/timeline`

依時間升序、分頁回傳使用者所有 session 的對話輪次（彼此穿插排列）。只想看單一 session 請改打 Module 26 自己的 `/sessions/{session_id}/timeline`（未經此閘道代理）。

**Query 參數：** `page`（預設 `1`）、`page_size`（預設 `20`，最大 `100`）

**Response 範例（節錄）：**

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
      "session_id": "c9d6b4ab-..."
    }
  ],
  "total": 1,
  "page": 1,
  "page_size": 20,
  "has_next": false
}
```

---

### `POST /api/0.5/memory/compact`

觸發指定 session 的壓縮（超過 context window 預算時，把較舊的對話輪次摘要成一則 summary frame）。`?session_id=xxx` 省略時壓縮 `default` session。

**Request Schema：** `trigger`（`auto`/`manual`/`reactive`，僅作記錄用途，預設 `auto`）、`context_window`（預設 `128000`）、`max_tokens`（選填）、`last_summarized_frame_id`（選填，省略則自動找該 session 最新的 summary frame）。

**Response 範例（未達壓縮門檻）：**

```json
{
  "compacted": false,
  "source": "under_budget",
  "summary_id": null,
  "boundary_id": null,
  "pre_compact_tokens": 142,
  "post_compact_tokens": 142,
  "turns_compacted": 0,
  "turns_kept": 1,
  "summary_frame_written": false,
  "fallback_used": false,
  "threshold_exceeded": false,
  "summary_tokens": 0,
  "turns_before": 1,
  "turns_after": 1,
  "status": "fallback"
}
```

`summary_id`/`boundary_id` 只有在真的壓縮發生時才會非 null；`fallback_used` 表示是否因為某種原因退回未壓縮的
fallback 路徑（跟 `status: "fallback"` 對應）；`turns_before`/`turns_after` 是壓縮前後的輪次數（未壓縮時兩者相同）。

---

### `POST /api/0.5/memory/archive`

歸檔一輪對話（`user_message` + `assistant_response`）到指定 session。`?session_id=xxx` 省略時寫入 `default` session。

**Request Schema：** `user_message`、`assistant_response`（皆必填）、`search_results`/`tool_calls`（選填陣列）、`timestamp`（選填）、`provider_message_id`（選填）。

**Response 範例：**

```json
{"frame_id": "20", "turn_index": 21, "session_id": "default"}
```

---

### `DELETE /api/0.5/memory`

GDPR 抹除：永久刪除目前使用者在 Module 26 的**所有**記憶（sessions、長期記憶、多媒體）。找不到任何記憶回 `404`。**此操作不可復原**，同時也會清掉 Chat 章節提到的 Redis 短期快取。

---

### `POST /api/0.5/memory/multimedia/search`

在目前使用者的 video/audio/image 索引中做混合搜尋，`media_type` 可限定單一媒體類型。（Module 21 agent-protocol 透過閘道呼叫此端點，因此保留在對外介面上。）

**Request Schema：** `query`（必填）、`top_k`（預設 `5`）、`media_type`（選填，`video`/`audio`/`image`）

**Response 範例（無結果時）：** `[]`

---

### `POST /api/0.5/memory/compact/evaluate`

估算壓縮預算，不實際執行壓縮。省略 `session_id` 只估算傳入的 `messages` 大小；帶 `session_id` 則彙總該 session 的歸檔對話 + 長期事實 + 多媒體片段一併估算。

> **注意：** `session_id` 要放在 **JSON body** 裡，不是 query 參數（跟上面 `/memory/compact` 不一樣，那支是
> query 參數）。放成 query 參數不會報錯，但會悄悄走「無 session」的空估算路徑，回傳看起來正常但其實是錯的
> `token_count: 0`。

**Response 範例（空 session）：**

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

### `GET /api/0.5/memory/sessions/{session_id}/summaries`

列出指定 session 目前已產生的 summary frames（`/memory/compact` 壓縮後寫入的摘要）。無摘要時回傳 `[]`。

---

<a id="processing"></a>

## 處理進度 (Processing)

上傳檔案後，AI 分析會非同步在背景執行。使用以下 API 查詢處理狀態。

`correlation_id` 從 `fileupload_analysis` 的回應 `upload_result.correlation_id` 取得。

---

### `GET /api/0.5/processing/cache/all`

查詢目前登入使用者所有上傳任務的處理進度。  
只回傳屬於目前使用者（相同 `branch_id`）的資料。

**Headers：** `Authorization: Bearer <token>`

**curl：**

```bash
curl http://raptor_open_0_5_api.dhtsolution.com:8012/api/0.5/processing/cache/all \
  -H "Authorization: Bearer <access_token>"
```

**Python：**

```python
import requests

headers = {"Authorization": f"Bearer {token}"}
resp = requests.get(
    "http://raptor_open_0_5_api.dhtsolution.com:8012/api/0.5/processing/cache/all",
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
    "video_orchestrator:8d44cfc4-...": {
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

### `GET /api/0.5/processing/cache/{m_type}/{key}`

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
CORRELATION_ID="8d44cfc4-..."

curl "http://raptor_open_0_5_api.dhtsolution.com:8012/api/0.5/processing/cache/video/${CORRELATION_ID}" \
  -H "Authorization: Bearer <access_token>"
```

**Python（輪詢等待完成）：**

```python
import requests
import time

def wait_for_completion(token: str, m_type: str, correlation_id: str, timeout: int = 600) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    url = f"http://raptor_open_0_5_api.dhtsolution.com:8012/api/0.5/processing/cache/{m_type}/{correlation_id}"
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
result = wait_for_completion(token, "video", "8d44cfc4-...")
print("摘要:", result.get("value", {}).get("summary", "")[:100])
```

**Response 範例：**

```json
{
  "key": "video_orchestrator:8d44cfc4-...",
  "original_key": "8d44cfc4-...",
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

### `POST /api/0.5/processing/process-file`

**Register API。** 對一個先前用 `POST /asset/fileupload` 上傳、狀態是 `uploaded`
的資產，顯式觸發 AI 分析（把上傳跟分析解耦）；也是 `POST /asset/unregister/...` 的還原方式。

**Request Schema：** `asset_path`、`version_id`（皆必填，來自 `POST /fileupload` 的回應）、
`processing_mode`（選填，例如 PDF 用 `"ocr"`）。

**行為依資產目前狀態分支：**

| 目前狀態 | 行為 |
|------|------|
| 不存在 | `404` |
| `unregistered`（內容已透過去重索引過） | 同步切回 `active`，**不發 Kafka 訊息** |
| `processing` 或 `active` | `409`（已在分析中，或已經是最新） |
| `uploaded` 或 `failed` | 發一則 Kafka 分析請求；狀態維持不變，直到真正處理的 orchestrator 內部呼叫 claim-processing 才會變成 `processing` |

對 `failed` 資產重新呼叫會重發 Kafka，但狀態在真正被 claim 之前仍然是 `failed`。

---

<a id="benchmark"></a>

## 基準測試 (Benchmark)

Module 22 的 proxy，端點掛在 `/api/0.5/benchmark/*`。上傳一份「schema」（測試案例 + 評分規則），
對真實 pipeline 跑一次，拿到客觀分數；同一份 schema 可以跑多次、互相 compare，追蹤某次改動是讓
輸出變好還是變差。

`POST /runs` 是非同步的：回 `202` + `run_id`，實際執行在背景跑，要輪詢 `GET /runs/{run_id}`
直到 `status` 變成 `completed`/`failed`。

**這個服務本身只是評分機制，價值取決於 schema 寫得好不好。** API 通了、跑得動，不代表分數有意義——
`keyword_match` 對一句話裡有沒有某個字這種簡單測試容易寫，但真的能反映輸出品質的，是搭配
`llm_judge`（給清楚的 rubric）或 `expected_answer` + `cosine_similarity` 的 test case，而且要涵蓋
會被改動實際影響到的情境。隨便寫幾個 test case 測 API 通不通，不等於這份 schema 能告訴你「這次改動
讓東西變好還是變差」。

### `target_pipeline` 可選值

| 值 | 打誰 | 需要 `branch_id`？ |
|---|---|---|
| `chat` | Module 15 chat-service | 否 |
| `rag` | Module 21 agent-protocol（`mode=direct`） | **是** |
| `search` | Module 25 personal-db-service（`/personal/search/hybrid`） | **是** |
| `classify` | Module 18 query-orchestrator | 否 |
| `lifecycle_infer` | Module 07 AI Lifecycle API（受測模型要先在 07 註冊） | 否 |
| `local_infer` | Module 16 訓練服務 | 否，但**這個部署還沒開 Module 16，用不了** |

**`search`/`rag` 的 `branch_id`：** Module 25 是每人一份物理隔離的資料庫，沒有共用語料。
每個 test case 的 `input` 可以自己填 `branch_id`（或 `user_id`，效果一樣）指定要測誰的資料；
沒填的話，會自動用**送出這次 run 的人**的身分（Module 13 從你登入的 JWT 解出來，自動塞進去，
不用手動處理）。兩者都沒有的話會直接報錯，不會偷偷搜到別人的資料或安靜地回空結果。

`/optimize`（AutoTune：訓練 → 評分 → 建議下一組參數的迴圈）也需要 Module 16，這個部署一樣用不了。

### `POST /api/0.5/benchmark/schemas`

上傳一份 schema。

**curl：**

```bash
curl -X POST http://raptor_open_0_5_api.dhtsolution.com:8012/api/0.5/benchmark/schemas \
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

**Response：**

```json
{"id":"d0873929-060d-4b14-8775-538b8336a206","name":"chat_url retest 3","version":"1.0","created_at":"2026-08-28T04:04:01.865316Z"}
```

評分方法（`scoring_schema.dimensions[].method`）：`keyword_match`、`contains_all`、`contains_any`、
`exact_match`、`numeric_tolerance`、`regex_match`、`cosine_similarity`（走 Module 07 embedding）、
`llm_judge`（自然語言 rubric，走 Module 07 LLM）、`latency_threshold`。

### `POST /api/0.5/benchmark/runs`

**curl：**

```bash
curl -X POST http://raptor_open_0_5_api.dhtsolution.com:8012/api/0.5/benchmark/runs \
  -H "Authorization: Bearer <access_token>" -H "Content-Type: application/json" \
  -d '{"schema_id": "d0873929-060d-4b14-8775-538b8336a206"}'
```

**Response：**

```json
{"run_id":"2b24dbec-a90b-4a68-89d6-0235f4340458","status":"queued"}
```

### `GET /api/0.5/benchmark/runs/{run_id}`

輪詢直到 `status` 變 `completed`/`failed`。

**Response：**

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

`error` 非 `null` 代表這個 test case 本身失敗（例如打錯 pipeline 的服務、或 `search`/`rag`
沒有可用的 `branch_id`）——`status` 還是會是 `completed`，要另外檢查每個 case 的 `error` 欄位，
不能只看外層 `status`。

> **注意：** 上面是精簡版範例，回應的最外層還有 `config_override`、`started_at`、`completed_at`、
> `created_at`、`progress_pct` 這 5 個欄位（省略具體值，因為是每次 run 都不同的時間戳），而每個
> `scores_per_case[]` 項目還多一個 `per_dimension` 物件（該 case 在每個評分維度各自的分數，不是只有彙總後的
> `aggregate`）。

### 其他端點

| Method | Path | 說明 |
|--------|------|------|
| `GET` | `/benchmark/schemas` | 列出所有 schema |
| `POST` | `/benchmark/schemas/upload` | 跟 `POST /benchmark/schemas` 一樣，但用 YAML/JSON 檔案上傳（multipart），不是 JSON body |
| `GET` / `DELETE` | `/benchmark/schemas/{id}` | 取得 / 刪除 schema（刪除會連帶刪掉底下的 runs） |
| `GET` | `/benchmark/schemas/{id}/runs` | 該 schema 的執行歷史 |
| `GET` | `/benchmark/schemas/{id}/leaderboard` | 依分數排名的 run 清單 |
| `GET` | `/benchmark/runs/{a}/compare/{b}` | 比較兩次 run 的分數差，`?pairwise=true` 改用 LLM 兩兩對比 |

---

<a id="guardrail"></a>

## 內容防護 (Guardrail)

代理 Module 23（guardrail-service）的 policy CRUD 跟系統開關兩組端點。注意：這組跟 chat/a2a 底層一直在跑的 `GuardrailMiddleware`（即時檢查邏輯）是分開的——這裡純粹是管理介面，沒有即時檢查邏輯。

### `GET /api/0.5/guardrail/policies`

列出所有已存的 policy（輕量版，不含內容）。

**Response：**

```json
[
  {"id": "<uuid>", "name": "default", "version": "v1", "is_active": true, "created_at": "2026-11-20T10:00:00Z"}
]
```

### `POST /api/0.5/guardrail/policies`

新增一個 policy 的內容（純文字，不限格式，YAML/JSON/純文字都可）。

**Request：** Query 參數 `name`（必填）、`version`（必填）、`target`（enum: `original`/`llama-guard`/`granite-guardian`/`gpt-oss-safeguard`，預設 `original`——非 original 的是針對特定 guard model 的客製 prompt，沒設就 fallback 回 original）。Body 是純文字內容。

**Response（201）：**

```json
{"id": "<uuid>", "name": "default", "version": "v1", "created_at": "2026-11-20T10:00:00Z"}
```

### `POST /api/0.5/guardrail/policies/upload`

跟上面一樣，但用檔案上傳（multipart）。Form 欄位：`file`（必填）、`name`（必填）、`version`（必填）、`target`（同上）。Response 跟上面一樣。

### `GET /api/0.5/guardrail/policies/template`

取得範例 policy 內容，複製修改後上傳。Query `format`（enum: `standard`/`checker_rules`/`moderation`，預設 `checker_rules`）。Response：`{"description": "...", "example_json": "...", "example_yaml": "..."}`。

### `GET /api/0.5/guardrail/policies/active`

取得目前啟用中的 policy 完整內容。沒有任一 policy 啟用時回 404。

### `GET /api/0.5/guardrail/policies/{policy_id}`

依 id 取得單一 policy 完整內容。

**Response：**

```json
{
  "id": "<uuid>", "name": "default", "version": "v1",
  "raw_content": "...", "content_llama_guard": null,
  "content_granite_guardian": null, "content_gpt_oss_safeguard": null,
  "is_active": true, "created_at": "2026-11-20T10:00:00Z"
}
```

### `PUT /api/0.5/guardrail/{policy_id}`

直接修改一個 policy 的內容欄位（注意這條路徑是直接掛在 `/guardrail` 下，不是 `/guardrail/policies/{id}`）。Body（全部選填，只有給的欄位會寫入）：`raw_content`、`content_llama_guard`、`content_granite_guardian`、`content_gpt_oss_safeguard`。Body 完全沒給任何欄位時回 400。若此 policy 目前是啟用中，寫入後 module 23 的快取會立即刷新生效。Response 同上。

### `PUT /api/0.5/guardrail/policies/{policy_id}/activate` / `PUT /api/0.5/guardrail/policies/{policy_id}/deactivate`

啟用／停用指定 policy（啟用會先停用目前啟用中的那個）。停用時，除非這個就是內建的 `default-standard-policy`，系統會自動 fallback 啟用該內建 policy（沒有的話會先建立）——永遠有一個 policy 在啟用中。`deactivate` 錯誤：404（不存在）、409（存在但目前不是啟用狀態）。

### `DELETE /api/0.5/guardrail/policies/{policy_id}`

刪除一個 policy（204）。錯誤：409——若是目前啟用中的 policy，或是內建的 `default-standard-policy`（任何版本，永遠不能刪，它是停用 fallback 的對象）。

### 系統開關

| Method | Path | 說明 |
|--------|------|------|
| `POST` | `/guardrail/system/enable` / `/disable` | 全域啟用／停用 guardrail 檢查，回 `{"enabled": bool}` |
| `POST` | `/guardrail/system/policy/enable` / `/disable` | 啟用／停用 policy-based LLM 檢查（只影響 `/policy/check/llm/*`，不影響 `/guard/check/*`；停用時那兩個端點一律回 `safe=true`），回 `{"policy_enabled": bool}` |
| `GET` | `/guardrail/system/status` | 目前所有開關狀態 + 啟用中 policy 摘要 + `conflict_strategy`（多個 guard model 判定不一致時的解決策略：`strict`/`majority`/`permissive`）+ `proxy_mounted`（OpenAI 相容代理是否已掛載） |

---

<a id="compliance"></a>

## 合規報表 (Compliance)

代理 Module 33 內部的 `/reports/compliance`，這個 router 自己額外檢查 Keycloak realm role `compliance-auditor`（JWT 的 `realm_access.roles`），沒有這個角色一律 403。

### `GET /api/0.5/compliance/report`

產生一段時間內的合規稽核報表，涵蓋 guardrail 違規、審核(moderation)標記、GDPR 工作（抹除/匯出/DSAR/同意撤回）、GDPR 權利請求（更正/限制/異議/同意/法律保留）、以及資料存取紀錄。

**Request：** 參數 `from`（必填，ISO 8601，含起點）、`to`（必填）、`format`（`json`/`csv`，預設 `json`）、`max_rows`（每個區塊最多回幾筆事件，預設上限 50000）。

**Response：** `format=json` 時每個區塊有 `total`（該期間精確計數）、`summary`（依分類/動作/狀態/日期的統計）、以及最多 `max_rows` 筆事件（最舊到最新，超過時 `truncated: true`）。`format=csv` 直接回檔案下載，欄位為 `section, occurred_at, record_id, actor_hash, subject_hash, kind, category, outcome, status, module, details`——**所有人員 id 都是 SHA-256 雜湊，絕不包含原始內容/檔名/自由文字**。Header `X-Report-Truncated-Sections` / `X-Report-Unavailable-Sections` 列出被截斷或沒有對應資料表的區塊。

**錯誤：** 403（缺角色）、422（參數錯誤）、502（module 33 連不上）。

---

<a id="gdpr"></a>

## GDPR / CCPA

代理 Module 33（privacy-compliance-service）。整組規律：「自助」端點永遠作用在呼叫者自己的 JWT `sub`；對應的「管理」端點（通常是同路徑加 `/{user_id}` 或在 `/admin/` 下）需要 Keycloak role `gdpr-admin`，沒有一律 403。除了下面特別標註的一條，全部都要 Bearer token。

### 抹除（Erasure，GDPR Art.17 / CCPA §1798.105）

- `POST /api/0.5/gdpr/erasure` — 自助。無 body。30 天內刪除你在 RAPTOR 所有儲存系統裡的個人資料。對已抹除過的帳號再送一次會回原本那個 job，不會新建——**Response（repeat call）：** `{"job_id": "...", "status": "already_erased", "estimated_completion_at": null, "reason": null, "hold_until": null, "exception_fields": [...]}`，`exception_fields` 這時列的是依法必須永久保留的紀錄本身（job 記錄/同意歷史/存取稽核等），不是使用者資料。若帳號有法律保留(legal hold)，回應是 `{"status": "DEFERRED", "reason": ..., "hold_until": ...}`，回應也會列出 `exception_fields`（依法必須保留的資料跟保留依據）。
- `POST /api/0.5/gdpr/erasure/{user_id}` — 管理員專用（`gdpr-admin`）。對已抹除過的使用者呼叫，行為跟上面自助版一樣回原本那個 job。
- `GET /api/0.5/gdpr/erasure/jobs/{job_id}` — **完全不需要登入 token**（不帶 `Authorization` header 一樣 200），刻意設計：自助抹除流程的第二步就是把申請者的 session 失效，如果這條還要求有效 JWT，就永遠無法確認自己的抹除結果。`job_id`（送出時給一次的隨機 UUID）本身就是存取控制，跟包裹追蹤連結是同一個邏輯。
- `POST /api/0.5/gdpr/erasure/jobs/{job_id}/retry` — 管理員專用，重試 `PARTIAL_FAILURE` 的 job。對一個卡在 `partial_failure`（某個 store 之前失敗）的 job 重試，失敗的 store 轉成 `deleted`、整個 job 的 `status` 變成 `completed`，`store_statuses` 準確反映每個 store 的最終結果。錯誤：409（使用者目前有法律保留——重試會刪到被保留的資料）。

### 匯出 / 可攜權（GDPR Art.20 / CCPA §1798.100）

- `GET` / `POST /api/0.5/gdpr/export` — 自助，每次呼叫都開一個新 job（`GET` 是票面規定的方法，`POST` 是為了向後相容保留）。打包你在每個儲存系統裡的資料成 ZIP。Response：`{"job_id": "...", "status": "pending"}`。完成後 `GET .../jobs/{job_id}` 的 Response：`{"job_id": "...", "user_id": "...", "requested_at": "...", "requested_by": "...", "status": "completed", "notes": null, "completed_at": "...", "download_url": "https://...", "download_urls": ["https://..."]}`——`download_url` 是單一連結，`download_urls` 是陣列版本，兩個都會有。
- `POST /api/0.5/gdpr/export/{user_id}` — 管理員專用。回應跟自助版同一個 `{"job_id": ..., "status": "pending"}` 形狀。
- `GET /api/0.5/gdpr/export/jobs/{job_id}` — 自己或管理員；明確檢查擁有權（job 的 `user_id` 必須等於呼叫者，除非是 `gdpr-admin`），否則 403「屬於別的使用者」。匯出不會讓 session 失效，所以這條仍然要求 JWT。完成後回下載連結。

### 保留政策引擎（Retention Policy，全部管理員專用，非個人資源）

| Method | Path | 說明 |
|--------|------|------|
| `GET` | `/gdpr/retention/policy` | 目前的保留政策表。**Response：** `{"policies": [{"data_category": "benchmark_runs", "retention_days": 90, "auto_delete": true, "legal_basis": "...", "notes": "...", "updated_at": "...", "updated_by": "..."}, ...]}`，每個 `data_category` 一列（共 10 列：`benchmark_runs`/`gdpr_access_log`/`gdpr_consent_history`/`gdpr_erasure_jobs`/`guardrail_violations`/`loki_logs`/`memvid_sessions`/`moderation_flags`/`moderation_flags_actionable`/`voice_recordings`） |
| `PUT` | `/gdpr/retention/policy` | Body 是 `list[dict]`（任意保留政策列的陣列，不是 `{"policies": [...]}` 包一層），更新政策。把 `GET` 原樣拿到的陣列整個送回去，內容(`retention_days`/`auto_delete`/`legal_basis`/`notes`)完全不變，只有 `updated_at`/`updated_by` 更新成呼叫者跟當下時間 |
| `GET` | `/gdpr/retention/policy/dry-run` | 唯讀：如果現在真的執行保留政策會刪掉幾筆資料（不會真的刪）。**Response：** `{"counts": {"benchmark_runs": 0, ...}, "as_of": "2026-11-23T..."}` |

### 資料主體請求（DSAR，GDPR Art.15 / CCPA §1798.100）

- `GET /api/0.5/gdpr/dsar` — 自助。你的資料結構化摘要（只有數量/分類，不含內容）。一般帳號直接 200 回報表；若索引項目超過一萬筆或組裝超時，回 202 `{"job_id": ..., "status": "pending"}`，用下面端點輪詢。**Response：** `{"user_id": "...", "generated_at": "...", "data_categories": {"identity": {...}, "uploaded_files": {...}, "indexed_content": {...}, "memory": {...}, "benchmark": {...}, "safety_violations": {...}}, "consent_status": {...}, "retention_summary": {...}, "ccpa_category": "...", "ccpa_source": "...", "ccpa_sells_data": false, "rights": {"erasure_endpoint": "...", "export_endpoint": "...", "consent_endpoint": "..."}, "unavailable_fields": {...}, "stores_failed": []}`。
- `GET /api/0.5/gdpr/dsar/jobs/{job_id}` — 自己或管理員，擁有權檢查跟匯出的 job 輪詢一樣。只有在上面 `GET /gdpr/dsar` 回 202（索引量夠大或組裝逾時）時才會真的用到這條。
- `GET /api/0.5/gdpr/dsar/{user_id}` — 管理員專用。回應跟自助版同一個結構；對一個剛被完整抹除的使用者查，`identity`/`consent_status` 底下的欄位會變成 `null`，可以拿來當「抹除真的生效了」的獨立佐證。

### 同意管理（Consent，GDPR Art.7，僅自助，沒有管理員版本）

### `GET /api/0.5/gdpr/consent`

取得你每個同意目的的狀態：`memory`、`analytics`、`benchmark`、`voice_recording`、`face_detection`、`voice_clone`，每項有 `granted`（bool）、`granted_at`、說明文字。**Response：** `{"purposes": [{"id": "memory", "granted": true, "granted_at": "...", "description": "..."}, ...]}`。

### `PUT /api/0.5/gdpr/consent/{purpose}`

**Request：** Body `{"granted": bool}`。`purpose` 是上面 6 個之一。撤回 `memory`/`benchmark`/`voice_recording`/`face_detection`/`voice_clone` 的同意會**連帶觸發該用途資料的部分抹除**（帳號本身不受影響）。例如撤回 `face_detection` 會刪掉你的人臉/人臉分組/已命名人物，停用人臉偵測/種子照片/People Search；撤回 `voice_clone` 會刪掉複製語音用的 speaker embedding，停用語音複製功能。

**Response：**

```json
{"id": "analytics", "granted": true, "changed": true, "erasure_job_id": null}
```

`changed` 是這次呼叫是否真的改變了狀態；`erasure_job_id` 只有撤回會觸發部分抹除的那 5 個用途、且真的觸發了才不是 `null`。

### 存取紀錄（Audit Log，GDPR Art.30）

- `GET /api/0.5/gdpr/audit-log` — 管理員專用。所有對個人資料的存取紀錄，最新在前。可用 `user_id`、`from`/`to` 篩選，`page`/`page_size`(上限 1000)分頁。
- `GET /api/0.5/gdpr/audit-log/my` — 自助，永遠只看自己的資料(無法指定別人的 id)。回傳存取者(非本人時是 SHA-256 雜湊)、端點、時間。

### 工作佇列總覽（管理員專用）

- `GET /api/0.5/gdpr/jobs` — 抹除/匯出/DSAR/同意撤回的工作全部列在一起，最新在前，每筆附 `failed_stores`。可用 `type`、`status`(如 `partial_failure`)、`user_id`、`from`/`to`、`stuck_hours`(卡住超過 N 小時仍未完成)篩選。
- `GET /api/0.5/gdpr/jobs/summary` — 依類型統計工作數量(`by_type` 是 `{類型: {狀態: 數量}}` 的巢狀物件)、`partial_failure` 總數、`stuck`(物件，列出卡住超過 `stuck_hours` 小時的工作，預設 24 小時;沒有卡住的工作時是空物件 `{}`，不是數字)。

### 更正／限制／異議（Art.16/18/21）

- `POST /api/0.5/gdpr/rectify` — 自助(Art.16)。Body `{"field": "email"|"firstName"|"lastName", "value": "..."}`(你的 Keycloak 個人資料，RAPTOR 本身不存這些欄位)。改 email 會標記未驗證；沒有 mail server 可寄驗證信時回 409。
- `POST /api/0.5/gdpr/restrict` — 自助(Art.18)。Body `{"reason": "..."}`(選填)。限制期間，`/gdpr/*` 以外的任何寫入(上傳/對話/處理...)一律回 451 `{"reason": "processing_restricted"}`；讀取跟你的 GDPR 權利端點仍可用。只有 `gdpr-admin` 能解除。重複呼叫只會回已存在的限制。**Response：** `{"user_id": "...", "restricted": true, "restricted_at": "...", "lifted_at": null, "reason": "..."}`。
- `POST /api/0.5/gdpr/restrict/lift` — 管理員專用。Body `{"user_id": "..."}`。該使用者沒被限制時回 404。**Response：** 同上，但 `restricted: false`、`lifted_at` 填上解除時間。
- `POST /api/0.5/gdpr/object` — 自助(Art.21)。Body `{"processing_purpose": "..."}`(6 個同意目的之一)。只停止該用途「未來」的處理，**不刪除過去的資料**(跟 `PUT consent` 設 false 不同，那個會連帶抹除)。Response:`{"purposes_stopped": [...], "data_deleted": false}`。

### 法律保留（Legal Hold，管理員專用）

- `POST /api/0.5/gdpr/admin/legal-hold` — Body(全部必填):`user_id`、`reason`(如 `"regulatory_audit"`)、`hold_until`(未來日期，UTC)、`authority`(如 `"DPO"`)。保留期間內，該使用者的抹除請求會被延後，直到每日的到期排程自動解除。錯誤:409(該使用者已有保留中的 hold，或正在被抹除，或已被抹除)。**Response：** `{"id": 1, "user_id": "...", "reason": "...", "authority": "...", "placed_at": "...", "hold_until": "2026-11-21", "released_at": null}`。對一個有保留中 hold 的使用者呼叫 `POST /gdpr/erasure` 會回 `status: "DEFERRED"`，`reason: "legal_hold"`，並附 `hold_until` 跟 `exception_fields`（依法保留項目跟法條依據的清單）。
- `GET /api/0.5/gdpr/admin/legal-holds` — 列出所有生效中的法律保留，依到期時間排序。**Response：** `{"holds": [...]}`；沒有任何生效中的保留時是 `{"holds": []}`。

### 處理活動記錄（RoPA，GDPR Art.30，管理員專用，控制者層級，非個人資源）

- `GET /api/0.5/gdpr/processing-records` — 完整的 Article 30 處理活動記錄。
- `GET /api/0.5/gdpr/processing-records/ccpa` — 同上但 CCPA 格式，`sells_data` 固定是 `false`。

---

<a id="docagent"></a>

## 文件代理 (Doc Agent)

### `GET`/`POST`/`PUT`/`PATCH`/`DELETE /api/0.5/docagent/{path}`

通用的已認證反向代理，轉發到 Module 29(doc-processing-agent)自己的 `/api/v1/docagent/{path}`——方法、header(去掉 Host/Content-Length)、body、query 參數全部原樣轉發，回應(狀態碼/header/body)也原樣轉回。這層本身不檢查路徑或內容，只套用標準 Bearer 認證。實際的請求/回應格式完全看 `{path}` 對應到 module 29 自己定義的哪個端點。

**錯誤：** 503 `{"error": "doc_agent_unavailable"}`——僅限 module 29 連不上時，其餘 4xx/5xx 都是 module 29 自己回的，原樣轉過來。

---

<a id="moderation"></a>

## 內容審核 (Moderation)

代理 Module 23 內部的 `/moderation/flags/*`，這個 router 額外檢查 Keycloak role `moderation-admin`，沒有一律 403。

「flag」是由 Module 09–12(各類型內容的審核檢查器)在自己的判定分數超過門檔時**自動建立**的 `moderation_flag` 記錄——這個 gateway 刻意不開放用 API 建立 flag，只能審查已存在的。

### `GET /api/0.5/moderation/flags`

列出審核佇列。**Request：** `page`(預設 1)、`page_size`(預設 50，上限 500)、`status`、`category`、`branch_id`、`version_id`、`asset_path`、`moment_id`、`chunk_id`、`from`/`to`。

**Response：**

```json
{
  "items": [{
    "id": "<uuid>", "branch_id": "...", "version_id": "...", "asset_path": "...",
    "moment_id": null, "chunk_id": null, "category": "...", "score": 0.91,
    "model": "...", "policy_version": null, "status": "flagged",
    "event_id": "<uuid>", "created_at": "...", "updated_at": "...",
    "asset_url": "https://..."
  }],
  "total": 1, "page": 1, "page_size": 50
}
```

`asset_url` 是這個 router 自己呼叫 module 04 補上的 presigned 下載連結，失敗時是 `null`(不會讓整個請求失敗)。

### `GET /api/0.5/moderation/flags/{flag_id}`

取得單一 flag，格式同上。

### `PATCH /api/0.5/moderation/flags/{flag_id}`

轉換 flag 的審查狀態(唯一能做的修改)。**Request：** `status`(`flagged`/`cleared`/`removed`，必填)、`note`(選填，上限 2000 字——會被記錄進稽核，但不會存進 flag 本身)。Response 同上(更新後)。

---

<a id="aiml"></a>

## AI/ML 模型與資料集 (AIML)

純代理，轉發到模型/資料集生命週期管理服務，方法/header(去 Host)/參數/body 原樣轉發，回應也原樣轉回(未定型的 passthrough)。

| Method | Path | 說明 |
|--------|------|------|
| `POST` | `/aiml/models/download` | Query `model_source`(`huggingface`/`ollama`)、`model_name`(必填，如 `google/gemma-3-270m-it`)——下載模型到本地暫存 |
| `GET` | `/aiml/models/local` | Query `model_source`(選填，省略列出全部)——列出已下載到本地的模型 |
| `POST` | `/aiml/datasets/download_from_network` | Body:`dataset_name`、`dataset_source`(如 `huggingface`)、`extract_multimedia`(預設 true)、`max_samples`(預設 1000)、`split`(預設 `train`)——下載資料集，支援文字/圖片/音訊/影片/多模態 |
| `GET` | `/aiml/datasets/local` | 列出已下載到本地的資料集 |

**錯誤：** 502(後端生命週期管理服務連不上)。

---

<a id="training"></a>

## 模型訓練 (Training)

純代理，轉發到微調訓練服務，同樣的「原樣轉發」模式。

### `POST /api/0.5/training/submit`

提交一個微調訓練工作。**Request：** `dataset_config`(`dataset_name_or_path`、`column_mapping` 必填，`max_length`/`train_size`/`val_size`/`train_split_name` 有預設值)、`training_config`(`experiment_name`、`model_name_or_path` 必填，`batch_size`/`learning_rate`/`max_epochs`/`gradient_accumulation_steps` 等有預設值，選填 `deepspeed_config`/`lora_config`/`quantization_config`)、`task_type`(必填，如 `"instruction"`)、`select_multiple_gpus`(預設 false)、`vram_budget_gb`(預設 5)。

**Response：**

```json
{"job_id": "...", "gpu_id": 0, "status": "queued", "config": {...}, "start_time": null, "end_time": null, "metrics": null, "model_path": null}
```

| Method | Path | 說明 |
|--------|------|------|
| `GET` | `/training/status/{job_id}` | 工作目前狀態(`queued`/`running`/`completed`/`failed`/`cancelled`)、進度指標、輸出路徑 |
| `GET` | `/training/list` | 列出所有工作，可用 `status` 篩選 |
| `POST` | `/training/cancel/{job_id}` | 中止執行中或排隊中的工作(已完成/失敗的無法取消) |
| `DELETE` | `/training/delete/{job_id}` | 刪除工作紀錄跟它的 log/checkpoint，不可逆。Query `force`(預設 false，強制刪除即使還在執行) |

---

<a id="identity"></a>

## 人物身分 (Identity)

這是 People Search / 人臉身分管理介面，大部分端點一對一代理 Module 25 的 `/personal/identity/*`(`profile-image` 除外，那條是真的跨三個模組的流程)。**每條路由都只作用在呼叫者自己的 JWT `sub` 上——沒有任何一個端點能存取別人的人物資料**(整個 router 沒有 `user_id` 參數)。

**資料模型**(理解端點前先知道):**identity**(人物)是一筆具名或待確認的「人」記錄(狀態:`pending`/`active`/`merged`/`deleted`)。**trace**(軌跡)是某人在單一影片裡一段連續出現(含起訖秒數跟代表畫面)。**group**(分組)是人臉比對流程在**同一部影片**裡認為是同一個人、但還沒綁定到任何具名人物的一群 trace(「未分配」直到被綁定)。綁定(bind)把一個 trace 或整個 group 掛到某個人物下；解除綁定(unbind)把 trace 移出。合併(merge)把一個人物併入另一個(分組/種子照片都會移過去)。刪除跟合併都有 24 小時內可撤銷/重做的機制(靠回應的 `merge_id`/`history_id`)。回應裡出現的人臉圖片(頭像、代表畫面)都會由這個 router 補上 presigned `image_url`(透過 module 04)，跟搜尋結果的 `asset_url` 是同一套做法。

### `GET /api/0.5/identity/identities`

列出人物，依最近更新排序。**Request：** `limit`(1-200，預設 50)、`offset`(預設 0)、`status`(`pending`/`active`/`merged`/`deleted`——預設列表會排除 merged/deleted，要看要明確指定)。

### `POST /api/0.5/identity/upload`

建立一個人物，或依 `identity_id` 更新(idempotent upsert)。**Request：** `name`(必填)、`aliases`(`{locale: 名稱}`，選填)、`identity_type`(預設 `"person"`)、`identity_id`(選填——給了就用那個 id 建立/更新，重試不會重複建立)。新建立的人物狀態是 `active`(有人特意命名，跟自動偵測的 pending 不同)。

### `GET /api/0.5/identity/file/{version_id}/pending-persons`

針對一部影片，列出它還沒分配的人臉分組，跟從這部影片建立出來的待確認人物。Response:`unassigned_groups`(含每個 trace 的起訖秒數跟代表畫面)、`pending_identities`。

### `POST /api/0.5/identity/file/{version_id}/pending-person`

從一部影片建立一個待確認人物，可同時綁定。**Request：** `name`(可空字串)，以下三者最多給一個:`trace_id`(單個 trace)、`trace_ids`(多個)、`group_id`(整個未分配分組)。任何 id 無效都會先擋下來，不會建立任何東西(404)。

### `GET /api/0.5/identity/{identity_id}`

取得單一人物。Response:`name`、`aliases`、`identity_status`、`source`(`auto`/`manual`)、`avatar`(含 `image_url`)、`video_count`、`group_count`。

### `PATCH /api/0.5/identity/{identity_id}`

改名、改別名，或確認一個待確認人物。**Request：** 全部選填——`name`、`aliases`(某個 locale 給 `null` 會移除)、`identity_status`(這裡唯一合法值是 `"active"`，用來確認 pending 人物；`merged`/`deleted` 只能透過 mergeinto/DELETE 產生)。錯誤:409(人物已是 merged 或 deleted)。

### `DELETE /api/0.5/identity/{identity_id}`

刪除一個人物(軟刪除)。它的人臉分組回到對應影片的「未分配」清單；trace/畫面/種子照片都保留。人物標記為 `deleted`:不出現在列表或 People Search 中，種子照片不再比對新影片。Response:`history_id`(24 小時內可用 `POST /identity/merge/{history_id}/undo` 復原)、`groups_unassigned`。

### `POST /api/0.5/identity/{identity_id}/bind/trace` / `/bind/group` / `/unbind`

**Request：** 皆為 `version_id` + (`trace_id` 或 `group_id`)。`bind/trace` 把一個 trace 移到此人物底下；`bind/group` 把整個分組移過來(一次做完，不用逐個 trace)；`unbind` 把一個 trace 移出，不綁定到任何人物(trace 目前不在此人物底下時回 409)。`bind/group` 若該分組不在指定影片裡回 404。

### `GET /api/0.5/identity/{identity_id}/traces`

列出此人物在所有影片裡的每一段連續出現。**Response：** `{"identity_id": "...", "traces": [{"trace_id": "...", "version_id": "...", "group_id": "...", "start_sec": 76.1, "end_sec": 76.7, "avg_confidence": 0.88, "representative": {...含 image_url}, "source_status": "active"}]}`——外層包一個物件,不是裸陣列。

### `POST /api/0.5/identity/{identity_id}/mergeinto`

把此人物合併進另一個，24 小時內可撤銷。**Request：** `target_id`(必填)、`keep_source`(預設 true)。別名會補到目標缺少的 locale(不會覆蓋既有的)；若目標原本沒有頭像，會繼承來源的。錯誤:409(合併進自己，或任一方已是 merged/deleted)。

### `POST /api/0.5/identity/merge/{merge_id}/undo` / `/redo`

`undo`:24 小時內復原一次合併或刪除。`redo`:24 小時內(從**原始動作**算起)重新套用已撤銷的合併/刪除；超過期限或狀態不是「已撤銷」回 409。

**Response：** `{"history_id": "...", "operation": "delete", "history_state": "reverted", "undo_deadline": "...", "applied": ["identity <id> identity_status"], "skipped": []}`——`applied` 列出這次真的改動了什麼，`skipped` 是在原動作之後又被改過、因此跳過不處理的項目。

### `POST /api/0.5/identity/{identity_id}/profile-image`

設定此人物的種子/檔案照片——之後的影片會拿這張去比對人臉。**這是這組裡唯一不是純代理的端點**:真正跑了三個模組的流程(module 11 做人臉 embedding、module 04 存檔、module 25 記錄 embedding 並設為頭像)。

**Auth：** 標準 Bearer **外加**呼叫者必須有 `face_detection` 的 GDPR 同意(在讀取照片之前就先檢查)。

**Request：** multipart 檔案欄位 `photo`(必填，JPEG/PNG/WebP，必須偵測到人臉)。

**Response：** 更新後的人物物件(含新頭像/`image_url`)加上 `photo: {"face_count": 1, "confidence": 0.98}`(偵測到多張臉時用最大/最清晰的那張)。

**錯誤：** 400(沒偵測到臉，或照片是空的)、403 `PERMISSION_DENIED`(缺 `face_detection` 同意——在讀取照片前就會回，不會先做運算)、503(同意檢查本身失敗，或人臉 embedding 服務連不上)、409(人物已是 merged 或 deleted)。

### `GET /api/0.5/identity/{identity_id}/profile-image`

取得此人物的照片(設定過的種子照片，沒有就用一張代表畫面)，含 presigned `image_url`。

---

<a id="temporal"></a>

## 時序推理 (Temporal)

代理 Module 25 的時序事實萃取/推理流程:RAPTOR 會從已處理內容的文字(文件/影片語音辨識/音訊逐字稿/圖片說明)萃取出「有日期的事實」，這組端點讓你送入新素材、查處理進度、讀取萃取結果、解析問題本身的時間範圍、以及針對一個答案的時序主張做確定性驗證(不是 LLM 判斷)。「TKG」(temporal knowledge graph)另外掛在 `/search/tkg`，這裡的 `capabilities` 端點只是互相參照。

### `GET /api/0.5/temporal/capabilities`

回報這個部署目前支援的時序功能版本，以下是回應的頂層欄位：

- `contract_versions`：支援的 contract 版本列表（如 `[1, 2]`）
- `temporal_v2`：bool，v2 功能是否真的對這個部署啟用
- `tkg`：支援的比對模式(`match`)、Allen 關係(`relations`)、排序方式(`sort`)、分頁方式(`pagination`)、時區設定、`reference`（引用解析的狀態列舉）、`unsupported`（規格定義但這個部署沒提供的功能，用了會 400，目前是 `["as_of"]`）
- `operations`：操作名稱對應的路由模板
- `extraction`：目前萃取流程用的模型(`model`)、萃取器/正規化/文法版本、支援的語言(`languages`)、支援的媒體類型
- `ingest`：目前實際支援的內容類型(`inline_content_types`)/副檔名(`asset_extensions`)/保留天數(`inline_retention_days`)/大小上限(`max_inline_bytes`)

### `POST /api/0.5/temporal/ingest`

提交新素材進時序處理流程。**Request：** 以下兩種恰好給一種——`content` + `content_type`(`text/plain`/`text/markdown`，上限 1MB，純文字會像一般上傳一樣建立 asset，但**7 天後過期**失效並刪除)或 `asset: {asset_path, version_id}`(已上傳過的檔案，保留期限照正常規則)。另外:`idempotency_key`(必填，同一個 key 重送回原結果，換成不同內容但用同 key 會 409)、`operation`(`ingest`預設=存入處理；`query`=只問既有素材的問題，不存任何東西，立即回答案；`ingest_and_query`=存入後等最多 `wait_seconds`[預設 60，上限 300]再回答，逾時則 `answer_status: "pending"`，之後用 `query` 搭配 `source_version_ids` 重問)。

**Response：** `{"operation": "ingest", "source_version_id": "...", "job_id": "...", "asset_path": "document/txt/inline", "status": "pending", "expires_at": "2026-11-27T11:00:18Z", "duplicate": false, "status_url": "/api/0.5/temporal/status/..."}`——`job_id` 跟 `source_version_id` 目前是同一個值，`status_url` 已經是完整路徑可直接拿來用。素材要先走完一般的上傳/分析流程（解析、分塊、索引）後才會建立時序工作本身——在那之前打 `GET /temporal/status/{id}` 會回 `404`，不是 `pending`；工作建立後 `status` 才會先後經過 `pending`/`running`，處理完成後變成 `complete`。想確定性地等到完成再拿結果，改用帶 `wait_seconds` 的 `operation: "ingest_and_query"`，或自行輪詢 `status` 直到不再是 `404`。

**錯誤：** 400 `INVALID_INGEST`(content/asset 兩個都給或都沒給)、413 `CONTENT_TOO_LARGE`、415 `UNSUPPORTED_CONTENT_TYPE`(HTML/email/聊天記錄/CSV/JSON/log 明確不支援，不會被誤當純文字處理)、409 `IDEMPOTENCY_KEY_REUSED`、409 `SOURCE_EXPIRED`、404(不是自己的 asset)、502 `ANSWER_UNAVAILABLE`(`query`/`ingest_and_query` 時下游回答服務失敗)。

### `GET /api/0.5/temporal/status/{version_id}`

查一個素材的時序萃取進度。Response:`status`(`pending`/`running`/`complete`/`partial`[部分片段失敗或語言不支援]/`failed`/`cancelled`)、片段數量、寫入的事實數、萃取器/正規化版本。錯誤:404(從沒跑過時序處理，或不是自己的素材)。

### `GET /api/0.5/temporal/reason/{version_id}` / `POST .../recompute`

`GET` 取得已算好的推理結果。**Response：** `{"source_version_id": "...", "status": "consistent", "network_version": "...", "stale": false, "job_status": "complete", "computed_at": 1795172441548, "derived_facts": [], "witness": [], "links_used": [{"from": "...", "relation": "after", "to": "...", "other": "..."}], "links_unresolved": []}`——`status` 是 `consistent`/`inconsistent`/`no_links`;`witness`、`links_used`、`links_unresolved` 永遠都是陣列欄位(沒有衝突/沒有連結時是空陣列,不是整個欄位消失);`computed_at` 是 Unix 毫秒整數;`stale` 是該素材的時序工作還在跑時為 true。`POST .../recompute` 重新排一次計算(202，非同步)。

### `POST /api/0.5/temporal/parse`

單純解析一段自然語言問題「本身帶的時間條件」，不碰任何已存的事實。**Request：** `text`(必填，1-2000 字)、`timezone`(選填)、`anchor`(選填，ISO 8601 帶時區——問題本身的「現在」，沒給的話「去年」之類的相對說法無法解析)。

**Response：** 三種結果之一——`RESOLVED`(含 `window` 起訖、`match`，或一個 `reference` 事件)、`NO_TEMPORAL`(沒有時間限制)、`UNRESOLVED`(無法解析，附 `reason`——這種情況不該忽略時間條件直接查)。

### `POST /api/0.5/temporal/verify`

針對一個答案所引用的時序主張，做確定性檢查(不是 LLM 判斷)。**Request：** `claims`(必填，上限 50，每項格式見下)、`evidence`(答案引用的事實 id 清單，必填，上限 500)、`complete`(bool，預設 false——證據檢索是否完整；`absence` 類主張只有 `complete=true` 時才能驗證)、`timezone`(選填)。

`claims` 每一項都是一個物件，用 `kind` 欄位區分種類：

| `kind` | 其他欄位 | 驗證內容 |
|---|---|---|
| `"date"` | `fact`(事實 id)、`date`(主張的日期字串)、`endpoint`(選填，`"whole"`/`"start"`/`"end"`，預設 `"whole"`) | 該事實的日期是否涵蓋主張的日期跟精確度 |
| `"ordering"` | `subject`、`object`(兩個事實 id)、`relations`(Allen 關係名稱陣列)、`wording`(選填，`"certain"`/`"possible"`，預設 `"certain"`) | 兩個事實的日期允許的 Allen 關係，是否涵蓋(`certain`)或至少命中一個(`possible`)主張的 `relations` |
| `"absence"` | （無） | 只有 `complete=true` 才能驗證 |

`date`/`ordering` 引用的事實 id 都必須出現在 `evidence` 裡且仍然有效，否則整筆主張直接判定失敗。Allen 關係名稱：
`before`/`meets`/`overlaps`/`starts`/`during`/`finishes`/`equals`/`finished-by`/`contains`/`started-by`/`overlapped-by`/`met-by`/`after`。

**Request 範例（最簡單的 `absence` 主張，不需要引用任何事實）：**

```json
{"claims": [{"kind": "absence"}], "evidence": [], "complete": true}
```

**Response：** `verdicts`(每個主張對應一筆，含 `index`、`verified`、`reason`，以及依種類不同的額外欄位，例如 `ordering` 會多帶 `relations`)、`all_verified`、引用的 `facts`。上面的 `absence` 範例會回：

```json
{"verdicts": [{"index": 0, "verified": true, "reason": null}], "all_verified": true, "facts": {}}
```

### `GET /api/0.5/temporal/expiry/{version_id}`

查一個「用純文字送入」(非檔案上傳)的素材何時過期。**Response：** `{"version_id": "...", "expires_at": 1795777218000, "state": "pending", "expired": false}`——`expires_at` 是 Unix epoch **毫秒**整數，不是 ISO 字串；`state`(`pending`/`expired`[已封存，永不再復活]/`deleted`)跟 `expired`(bool)是兩個分開的欄位，不是同一件事的兩種寫法。錯誤:404(是上傳的檔案，有自己獨立的生命週期，不適用這條；或不是自己的素材)。

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

BASE_URL = "http://raptor_open_0_5_api.dhtsolution.com:8012"

# 1. 登入
resp = requests.post(f"{BASE_URL}/api/0.5/sso/login",
                     data={"username": "<your_username>", "password": "<your_password>"})
token = resp.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# 2. 上傳影片
with open("川普習近平會面.mp4", "rb") as f:
    resp = requests.post(
        f"{BASE_URL}/api/0.5/asset/fileupload_analysis",
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
            f"{BASE_URL}/api/0.5/processing/cache/video/{correlation_id}",
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
    f"{BASE_URL}/api/0.5/search/hybrid",
    headers={**headers, "Content-Type": "application/json"},
    json={"query": "川普習近平 會面內容", "top_k": 5},
)
for r in resp.json()["results"]:
    print(f"score={r['score']:.4f} | {r['payload'].get('filename','?')}")

# 5. RAG 問答
resp = requests.post(
    f"{BASE_URL}/api/0.5/a2a/query",
    headers={**headers, "Content-Type": "application/json"},
    json={"question": "川普和習近平會面討論了哪些議題？", "top_k": 5, "mode": "direct"},
    timeout=120,
)
print("\nRAG 答案:", resp.json()["answer"])
```
