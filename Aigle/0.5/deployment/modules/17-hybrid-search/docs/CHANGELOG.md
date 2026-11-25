# Hybrid Search — Changelog

## 2026-04-15 OpenSearch Mapping — Asset Lifecycle Fields

### `app/services/opensearch_service.py`
- `_ensure_index()` 的 `payload.properties` 新增兩個 `keyword` 欄位：
  - `asset_path`：對應 `02-object-storage` 的 `asset_path`，用於 asset lifecycle 狀態同步
  - `version_id`：對應 `02-object-storage` 的 `version_id`，用於版本精確過濾
- 背景說明：`02-object-storage` 的 `SearchSync` 在 asset archive / delete / clone 時，會對 `hybrid_index` 執行 `update_by_query` / `delete_by_query`，filter 條件為 `payload.asset_path` + `payload.version_id`；若這兩個欄位未在 mapping 中宣告為 `keyword`，OpenSearch 會動態將其 map 為 `text`，導致 `term` query 失效。

---

## 2026-04-13 OpenSearch Warm-up on Startup

### `app/services/opensearch_service.py`
- `initialize()` 新增 `_warmup()` 步驟：啟動時送一次 `match_all` 暖機查詢
  - 預熱 JVM JIT、OS page cache、query cache，消除第一次真實查詢的 cold start（約 1s）
  - index 尚無資料時例外被靜默吞掉，不影響啟動流程

---

## 2026-04-13 Reranker Latency Optimization

### `app/core/rerank.py`
- `model.predict` 的 `show_progress_bar` 從 `True` 改為 `False`：tqdm 初始化在小 batch 有 overhead，節省約 50–100ms
- 移除 hot path 裡的 `torch.cuda.empty_cache()`：此 call 會觸發 CUDA sync，改移至 `unload_model` 保留（模型卸載時仍執行）

### `app/core/config.py`
- 新增 `RERANK_DEPTH: int = 30`：控制送進 reranker 的最大文件數
  - 修正前：`hybrid_search` 固定傳 `top_k * 5`（預設 50 份）進 reranker
  - 修正後：最多傳 `RERANK_DEPTH`（預設 30）份，可透過 `.env` 調整

### `app/services/search.py`
- `hybrid_search` 的 rerank 前截斷改為 `min(len(fused_results), settings.RERANK_DEPTH)`
  - 保證 `rerank_depth >= top_k`（避免結果不足）

---

## 2026-04-13 Code Cleanup & Robustness Fixes

### `app/services/opensearch_service.py`
- 移除 `import uuid`（已無使用）
- `batch_insert` 移除 `item.id or str(uuid.uuid4())` fallback，改為直接使用 `item.id`
  - `ingest_docs` 已保證寫入前必有 ID；保留 fallback 會掩蓋直接呼叫 `batch_insert` 時 ID 不一致的問題

### `app/main.py`
- `asyncio.create_task` 的模型載入 task 加入 `add_done_callback(_log_task_error)`
  - 修正前：模型載入例外被 asyncio 靜默吞掉，API 啟動但所有請求回 503，無 log 可查
  - 修正後：例外會以 `ERROR` 等級記錄在 log 內

### `app/api/v1/search.py`
- 移除未使用的 `get_payload_schema_manager` import

### `app/core/embedding.py`
- 修正 `encode_async` method body 多一格縮排（9 spaces → 8 spaces）

---

## 2026-04-10 Cleanup & ID Consistency Fix

### `app/schemas.py`
- 移除未使用的 `Tuple` import

### `app/services/search.py`
- `hybrid_search_by_opensearch`：BM25 部分改用 `_build_bm25_clauses`，與 `bm25_search_by_opensearch` 一致，正確處理 nested 欄位（例如 `temporal` schema 的 `payload.events.description`）

### `app/services/ingest.py`
- `ingest_docs` 新增 ID 補齊邏輯：在寫入前，若 `doc.id is None` 則統一補一個 `uuid4()` string
- 修正前：OpenSearch 自行生成 UUID，Qdrant 收到 `None`（行為未定義），兩邊 ID 不同導致 RRF fusion 永遠無法 merge 同一文件
- 修正後：同一份文件在 Qdrant 和 OpenSearch 使用相同 ID，RRF deduplication 正常運作

---

## 2026-04-10 Dynamic PayloadSchema Refactor

### 問題背景
原設計中 PayloadSchema 只管了 `content_extractor` 和 `bm25_fields`，但 filter 邏輯卻硬寫在 `OpenSearchService.build_filters` 和 `QdrantService.build_filters` 裡（if/else 判斷 schema name）。新增 schema 必須同時修改三個檔案，違背了動態設計的初衷。`payload_schema` 也以 query param 形式散落在各 endpoint，與 `SearchRequest` body 裡的其他搜尋參數不一致。

### `app/schemas.py`
- `PayloadSchema` 新增兩個 flag：
  - `skip_status_filter: bool = False` — 控制是否略過 `status=active` filter 條件
  - `skip_req_filters: bool = False` — 控制是否略過 req 裡的 `embedding_type/type/filename/speaker/source` 條件
- `temporal` schema 設定 `skip_status_filter=True, skip_req_filters=True`（與原 hardcode 行為一致）
- `SearchRequest` 新增 `payload_schema: str = "contextual"` 欄位，統一放在 request body

### `app/core/payload.py`
- 新增 `get_schema(name)` 方法，回傳完整 `PayloadSchema` 物件
- `get_extractor` 和 `get_bm25_fields` 改為委派給 `get_schema`
- 三個方法的錯誤處理統一改為 `raise ValidationError(...)` (HTTP 400)，取代原本的 `ValueError`（會造成 500）

### `app/services/opensearch_service.py`
- `build_filters(req, payload_schema: str)` 改為 `build_filters(req, schema: PayloadSchema)`
- filter 邏輯改由 `schema.skip_status_filter` 和 `schema.skip_req_filters` 控制，移除 if/else hardcode
- import 新增 `PayloadSchema`

### `app/services/qdrant_service.py`
- 同 opensearch_service.py，`build_filters` 改為接收 `PayloadSchema` 物件
- import 新增 `PayloadSchema`

### `app/services/search.py`
- 所有搜尋方法移除 `payload_schema: str` 參數，改從 `req.payload_schema` 讀取
- `build_filters` 呼叫改為先 `self.schema_manager.get_schema(req.payload_schema)` 取得物件再傳入
- `hybrid_search_by_opensearch` 的 `bm25_fields` 改從 `schema.bm25_fields` 取得（同時修正原本 `self._get_bm25_fields()` 不存在的 bug）

### `app/services/ingest.py`
- 移除手動 `if payload_schema not in self.schema_manager.schemas: raise ValueError`
- 改為直接呼叫 `self.schema_manager.get_extractor(payload_schema)`，驗證邏輯統一在 `PayloadSchemaManager.get_schema` 處理

### `app/api/v1/search.py`
- `/hybrid`, `/vector`, `/bm25` 三個 endpoint 移除 `payload_schema: str = "contextual"` query param
- `payload_schema` 現在統一在 `SearchRequest` body 裡傳入

### `app/schemas.py`（補充修正）
- `PayloadSchema.bm25_fields` 從 `Optional[List[str]] = None` 改為 `List[str] = []`
- `PayloadSchema.nested_paths` 從 `Optional[List[str]] = None` 改為 `List[str] = []`
- 防止新增 schema 時未指定這兩個欄位導致 `_build_bm25_clauses` 在 `for field in fields` 拋出 `TypeError`

### `app/services/search.py`（補充修正）
- `rrf_fusion` 的 `k` 參數移除多餘的 `hasattr(settings, 'RRF_K_FACTOR')` 檢查，直接用 `settings.RRF_K_FACTOR`

### `README.md`
- 更新 `payload_schema` 說明：search endpoint 改為 request body 欄位，ingest endpoint 仍為 query param
- 新增「新增自訂 Payload Schema」章節，說明只需修改 `schemas.py` 一個地方
- 新增 `PayloadSchema` 欄位對照表（含 `skip_status_filter` / `skip_req_filters`）
- 更新 API 端點範例 request body

---

## 2026-04-10 Bug Fixes

### `app/schemas.py`
**問題：** `PayloadSchema` 使用 `Callable` 作為 Pydantic 欄位，在 Pydantic v2 預設設定下會於 import 時拋出 `PydanticUserError`，因為 `DEFAULT_EXTRACTORS` 在模組層級被建立。

**修改：**
- 新增 `from pydantic import ConfigDict`
- 在 `PayloadSchema` 加入 `model_config = ConfigDict(arbitrary_types_allowed=True)`

---

### `app/services/opensearch_service.py`
**注意：** 此 docker-compose 的 OpenSearch 3.4.0 使用預設 security（HTTPS 啟用），`use_ssl=True` 為正確設定，不修改。`verify_certs=settings.VERIFY_CERTS`（預設 `False`）處理自簽憑證，`ssl_show_warn=False` 抑制警告。

---

### `app/services/search.py`
**問題：** `hybrid_search_by_opensearch` 方法內呼叫 `self._get_bm25_fields(payload_schema)`，但 `SearchService` 上不存在此方法，會拋出 `AttributeError`。（此方法目前未被任何 API endpoint 呼叫，為死碼 bug）

**修改：**
- `self._get_bm25_fields(payload_schema)` → `self.schema_manager.get_bm25_fields(payload_schema)[0]`
- `get_bm25_fields` 回傳 `(fields, nested_paths)` tuple，`[0]` 取 fields list

---

### `app/core/embedding.py`
**問題：** `encode_async` 的型別標注為 `list[str]`，但呼叫端（search router）傳入的是單一字串 `req.query: str`。雖然 `SentenceTransformer.encode` 兩者都接受，標注不一致容易誤導。

**修改：**
- `text: list[str]` → `text: str | list[str]`
