# Hybrid Search API

OpenSearch + Qdrant 混合搜尋 API，支援向量搜尋、BM25 搜尋、混合搜尋（RRF 融合）以及動態數據模式 (Payload Schemas)。

## 目錄

- [架構概覽](#架構概覽)
- [支援的數據模式 (Payload Schemas)](#支援的數據模式-payload-schemas)
- [新增自訂 Payload Schema](#新增自訂-payload-schema)
- [環境設定](#環境設定)
- [API 端點](#api-端點)
  - [健康檢查](#健康檢查)
  - [文件匯入 (Ingest)](#文件匯入-ingest)
  - [搜尋功能 (Search)](#搜尋功能-search)
- [配置說明](#配置說明)

---

## 架構概覽

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Application                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Ingest API │  │   Search API │  │  Health API  │      │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘      │
│         │                │                                  │
│         └────────┬───────┘                                  │
│                  ▼                                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Service Container (依賴注入)              │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐            │  │
│  │  │Embedding │  │OpenSearch│  │ Qdrant   │            │  │
│  │  │ Manager  │  │ Service  │  │ Service  │            │  │
│  │  └──────────┘  └──────────┘  └──────────┘            │  │
│  │  ┌──────────┐  ┌──────────────────────┐              │  │
│  │  │ Reranker │  │  PayloadSchemaManager │              │  │
│  │  │ Manager  │  │  (filter + BM25 + ex- │              │  │
│  │  └──────────┘  │   tractor 統一管理)   │              │  │
│  │                └──────────────────────┘              │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 服務說明

| 服務 | 用途 |
|------|------|
| **Embedding Manager** | 文本向量化（BGE-M3 模型） |
| **Reranker Manager** | 重新排序（bge-reranker-v2-m3），透過 schema 的 `content_extractor` 取出文字 |
| **PayloadSchemaManager** | 統一管理各 schema 的 BM25 欄位、filter 行為、rerank 文字提取邏輯 |
| **OpenSearch Service** | BM25 搜尋、全文檢索（支援 Nested 結構） |
| **Qdrant Service** | 向量搜尋、相似度比對 |
| **Task Manager** | 背景任務管理 |

---

## 支援的數據模式 (Payload Schemas)

`payload_schema` 放在 **request body** 裡，控制搜尋的 filter 行為、BM25 欄位與 rerank 文字提取邏輯。

| Schema | BM25 欄位 | Filter 行為 | 適用場景 |
|--------|-----------|-------------|---------|
| `contextual`（預設） | `payload.text`, `payload.summary` | 加 `status=active`，支援 type/filename/speaker/source 過濾 | 標準文件、音訊片段 |
| `temporal` | `payload.events.description`（nested） | 只加 `payload_schema=temporal`，忽略其他 req filters | 影片事件、時間序列 |

### PayloadSchema 欄位說明

每個 schema 定義以下屬性（均在 `app/schemas.py` 的 `DEFAULT_EXTRACTORS` 裡設定）：

| 欄位 | 說明 |
|------|------|
| `name` | schema 識別名稱 |
| `description` | 說明 |
| `content_extractor` | Callable，從 payload dict 提取文字供 reranker 使用 |
| `bm25_fields` | OpenSearch BM25 搜尋的欄位清單（支援 nested path） |
| `nested_paths` | 需要用 nested query 的欄位前綴 |
| `skip_status_filter` | `True` 時不加 `status=active` 條件（預設 `False`） |
| `skip_req_filters` | `True` 時忽略 req 的 type/filename/speaker/source 條件（預設 `False`） |

---

## 新增自訂 Payload Schema

只需修改 `app/schemas.py`，在 `DEFAULT_EXTRACTORS` 加入新項目即可，**不需要改其他檔案**：

```python
def my_extractor(payload: Dict[str, Any]) -> str:
    return payload.get("my_field", "")

DEFAULT_EXTRACTORS["my_schema"] = PayloadSchema(
    name="my_schema",
    description="自訂 schema 說明",
    content_extractor=my_extractor,
    bm25_fields=["payload.my_field"],
    nested_paths=[],
    skip_status_filter=False,
    skip_req_filters=False,
)
```

---

## 環境設定

### 1. 設定環境變數

建立 `.env` 檔案（Docker Compose 內使用 container name 作為 host）：

```env
# OpenSearch Settings
OPENSEARCH_HOST=opensearch-node1       # Docker: container name; 本地測試: localhost
OPENSEARCH_PORT=9200
OPENSEARCH_INDEX=hybrid_index
OPENSEARCH_USER=admin
OPENSEARCH_PASSWORD=your_password
OPENSEARCH_DASHBOARDS_HOST=opensearch-dashboards
OPENSEARCH_DASHBOARDS_PORT=5601
VERIFY_CERTS=False                     # 自簽憑證時設 False；OpenSearch 預設啟用 HTTPS

# RRF Fusion Settings
RRF_K_FACTOR=60

# Qdrant Settings
QDRANT_HOST=qdrant                     # Docker: container name; 本地測試: localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=raptor

# Embedding Model Settings
EMBEDDING_MODEL=BAAI/bge-m3
VECTOR_DIM=1024

# Reranker Model Settings
RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# HuggingFace Cache（模型快取路徑）
HF_HOME=/home/.cache/huggingface       # container 內路徑
HF_CACHE=/path/to/host/.cache/huggingface  # host 掛載路徑

# App Settings
DEBUG=false
```

> **注意：** OpenSearch 預設啟用 HTTPS（security plugin 開啟），API 使用 `use_ssl=True` + `verify_certs=False` 連線，請勿設定 `plugins.security.disabled: true`。

### 2. 啟動服務

```bash
docker compose up -d
```

啟動順序：`opensearch-node1` → healthcheck 通過後 → `api` 容器才會啟動（由 `depends_on` 控制）。

---

## API 端點

### 健康檢查

```http
GET /api/v1/health/live
GET /api/v1/health/ready
```

`/ready` 回傳 embedding model、reranker、OpenSearch、Qdrant 各自的 ready 狀態。

---

### 文件匯入 (Ingest)

`payload_schema` 作為 **query param** 傳入，決定匯入時使用的文字提取邏輯。

#### 匯入 JSON Body

```http
POST /api/v1/ingest/json?payload_schema=contextual
Content-Type: application/json
```

**Temporal 模式範例：**

```http
POST /api/v1/ingest/json?payload_schema=temporal
```
```json
[
  {
    "id": "video1",
    "payload": {
      "video_id": "vid_001",
      "filename": "demo.mp4",
      "duration": 120.5,
      "events": [
        {
          "start_timestamp": 10.2,
          "end_timestamp": 15.8,
          "description": "一個人走進房間",
          "objects": ["person", "door"]
        }
      ]
    }
  }
]
```

#### 匯入 JSON 檔案

```http
POST /api/v1/ingest/file?payload_schema=contextual
POST /api/v1/ingest/file-background?payload_schema=contextual
```

背景匯入回傳 `task_id`，可用 `GET /api/v1/ingest/task/{task_id}` 查詢進度。

---

### 搜尋功能 (Search)

`payload_schema` 放在 **request body** 裡（所有搜尋端點一致）。

#### 1. 混合搜尋 (Hybrid Search)

```http
POST /api/v1/search/hybrid
Content-Type: application/json
```

流程：BM25（OpenSearch）+ Vector（Qdrant）→ RRF 融合 → Reranker → Top-K 結果

**Request Body：**

| 欄位 | 類型 | 必填 | 說明 |
|------|------|------|------|
| `query` | string | ✅ | 搜尋查詢字串 |
| `top_k` | integer | ❌ | 返回結果數量，預設 10 |
| `payload_schema` | string | ❌ | Schema 名稱，預設 `contextual` |
| `embedding_type` | string | ❌ | 向量化欄位，`text` 或 `summary` |
| `type` | string/list | ❌ | 文件類型過濾 |
| `filename` | list | ❌ | 指定檔名過濾 |
| `speaker` | list | ❌ | 說話者過濾（audio/video） |
| `source` | string | ❌ | 來源檔案類型（e.g., `pdf`, `docx`） |

**範例：**

```json
{
  "query": "會議中提到的預算問題",
  "top_k": 10,
  "payload_schema": "contextual",
  "type": ["documents", "audios"]
}
```

**Temporal 範例：**

```json
{
  "query": "一個人走進房間",
  "top_k": 5,
  "payload_schema": "temporal"
}
```

#### 2. 向量搜尋 (Vector Search)

```http
POST /api/v1/search/vector
Content-Type: application/json
```

只走 Qdrant，不做 BM25 和 rerank。Request body 格式與 hybrid 相同。

#### 3. BM25 搜尋 (BM25 Search)

```http
POST /api/v1/search/bm25
Content-Type: application/json
```

只走 OpenSearch BM25，不做 vector 和 rerank。Request body 格式與 hybrid 相同。

---

## 配置說明

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `OPENSEARCH_HOST` | localhost | OpenSearch 位址（Docker 內用 container name） |
| `OPENSEARCH_PORT` | 9200 | OpenSearch 連接埠 |
| `OPENSEARCH_INDEX` | hybrid_index | OpenSearch 索引名稱 |
| `OPENSEARCH_USER` | admin | OpenSearch 帳號 |
| `OPENSEARCH_PASSWORD` | admin | OpenSearch 密碼 |
| `VERIFY_CERTS` | False | SSL 憑證驗證，自簽憑證設 False |
| `QDRANT_HOST` | localhost | Qdrant 位址（Docker 內用 container name） |
| `QDRANT_PORT` | 6333 | Qdrant 連接埠 |
| `QDRANT_COLLECTION` | raptor | Qdrant collection 名稱 |
| `RRF_K_FACTOR` | 60 | RRF 融合常數（越大越平滑） |
| `RERANK_DEPTH` | 30 | 送進 reranker 的最大文件數（降低可縮短延遲，不得低於 top_k） |
| `EMBEDDING_MODEL` | BAAI/bge-m3 | 向量化模型 |
| `VECTOR_DIM` | 1024 | 向量維度 |
| `RERANKER_MODEL` | BAAI/bge-reranker-v2-m3 | Reranker 模型 |
| `HF_HOME` | — | HuggingFace cache 在 container 內的路徑 |
| `HF_CACHE` | — | HuggingFace cache 在 host 的掛載路徑 |
| `DEBUG` | false | 開啟後 error response 附帶 traceback |
