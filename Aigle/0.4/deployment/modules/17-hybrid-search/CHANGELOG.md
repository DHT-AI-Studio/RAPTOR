# Hybrid Search — Changelog

## 2026-04-15 OpenSearch Mapping — Asset Lifecycle Fields

### `app/services/opensearch_service.py`
- Added two `keyword` fields to `_ensure_index()`'s `payload.properties`:
  - `asset_path`: corresponds to `02-object-storage`'s `asset_path`, used for asset lifecycle state sync
  - `version_id`: corresponds to `02-object-storage`'s `version_id`, used for exact version filtering
- Background: `02-object-storage`'s `SearchSync` runs `update_by_query` / `delete_by_query` against `hybrid_index` on asset archive / delete / clone, filtering on `payload.asset_path` + `payload.version_id`; if these two fields aren't declared as `keyword` in the mapping, OpenSearch dynamically maps them as `text`, breaking `term` queries.

---

## 2026-04-13 OpenSearch Warm-up on Startup

### `app/services/opensearch_service.py`
- Added a `_warmup()` step to `initialize()`: sends a `match_all` warm-up query at startup
  - Warms up the JVM JIT, OS page cache, and query cache, eliminating the cold start (~1s) on the first real query
  - The exception when the index has no data yet is silently swallowed, without affecting startup

---

## 2026-04-13 Reranker Latency Optimization

### `app/core/rerank.py`
- Changed `model.predict`'s `show_progress_bar` from `True` to `False`: tqdm initialization has overhead on small batches, saving ~50–100ms
- Removed `torch.cuda.empty_cache()` from the hot path: this call triggers a CUDA sync; moved to `unload_model` instead (still runs on model unload)

### `app/core/config.py`
- Added `RERANK_DEPTH: int = 30`: controls the maximum number of documents passed to the reranker
  - Before the fix: `hybrid_search` always passed `top_k * 5` (50 documents by default) into the reranker
  - After the fix: at most `RERANK_DEPTH` (default 30) documents are passed, adjustable via `.env`

### `app/services/search.py`
- The pre-rerank truncation in `hybrid_search` changed to `min(len(fused_results), settings.RERANK_DEPTH)`
  - Guarantees `rerank_depth >= top_k` (avoiding insufficient results)

---

## 2026-04-13 Code Cleanup & Robustness Fixes

### `app/services/opensearch_service.py`
- Removed `import uuid` (no longer used)
- Removed the `item.id or str(uuid.uuid4())` fallback from `batch_insert`, using `item.id` directly
  - `ingest_docs` already guarantees an ID exists before writing; keeping the fallback would mask ID-inconsistency issues when `batch_insert` is called directly

### `app/main.py`
- Added `add_done_callback(_log_task_error)` to the `asyncio.create_task` model-loading task
  - Before the fix: a model-loading exception was silently swallowed by asyncio — the API would start but every request returned 503, with no log to check
  - After the fix: the exception is logged at `ERROR` level

### `app/api/v1/search.py`
- Removed the unused `get_payload_schema_manager` import

### `app/core/embedding.py`
- Fixed an extra level of indentation in the `encode_async` method body (9 spaces → 8 spaces)

---

## 2026-04-10 Cleanup & ID Consistency Fix

### `app/schemas.py`
- Removed the unused `Tuple` import

### `app/services/search.py`
- `hybrid_search_by_opensearch`: the BM25 portion now uses `_build_bm25_clauses`, consistent with `bm25_search_by_opensearch`, correctly handling nested fields (e.g. the `temporal` schema's `payload.events.description`)

### `app/services/ingest.py`
- Added ID-backfill logic to `ingest_docs`: before writing, if `doc.id is None`, a `uuid4()` string is generated and used uniformly
- Before the fix: OpenSearch generated its own UUID while Qdrant received `None` (undefined behavior) — the same document ended up with different IDs in each store, so RRF fusion could never merge them
- After the fix: the same document uses the same ID in both Qdrant and OpenSearch, so RRF deduplication works correctly

---

## 2026-04-10 Dynamic PayloadSchema Refactor

### Background
In the original design, PayloadSchema only managed `content_extractor` and `bm25_fields`, but the filter logic was hard-coded into `OpenSearchService.build_filters` and `QdrantService.build_filters` (if/else branches on schema name). Adding a new schema required changing three files at once, defeating the point of a dynamic design. `payload_schema` was also scattered across endpoints as a query param, inconsistent with the other search parameters living in the `SearchRequest` body.

### `app/schemas.py`
- Added two flags to `PayloadSchema`:
  - `skip_status_filter: bool = False` — controls whether the `status=active` filter condition is skipped
  - `skip_req_filters: bool = False` — controls whether the request's `embedding_type/type/filename/speaker/source` conditions are skipped
- The `temporal` schema sets `skip_status_filter=True, skip_req_filters=True` (matching the original hard-coded behavior)
- Added a `payload_schema: str = "contextual"` field to `SearchRequest`, unifying it into the request body

### `app/core/payload.py`
- Added a `get_schema(name)` method, returning the full `PayloadSchema` object
- `get_extractor` and `get_bm25_fields` now delegate to `get_schema`
- Error handling in all three methods now uniformly `raise ValidationError(...)` (HTTP 400), replacing the original `ValueError` (which resulted in a 500)

### `app/services/opensearch_service.py`
- Changed `build_filters(req, payload_schema: str)` to `build_filters(req, schema: PayloadSchema)`
- Filter logic is now controlled by `schema.skip_status_filter` and `schema.skip_req_filters`, removing the hard-coded if/else
- Added `PayloadSchema` to imports

### `app/services/qdrant_service.py`
- Same as opensearch_service.py — `build_filters` now takes a `PayloadSchema` object
- Added `PayloadSchema` to imports

### `app/services/search.py`
- Removed the `payload_schema: str` parameter from every search method, reading from `req.payload_schema` instead
- `build_filters` calls now first fetch the object via `self.schema_manager.get_schema(req.payload_schema)` before passing it in
- `hybrid_search_by_opensearch`'s `bm25_fields` now comes from `schema.bm25_fields` (also fixes a bug where the original `self._get_bm25_fields()` didn't exist)

### `app/services/ingest.py`
- Removed the manual `if payload_schema not in self.schema_manager.schemas: raise ValueError`
- Now calls `self.schema_manager.get_extractor(payload_schema)` directly, with validation logic unified in `PayloadSchemaManager.get_schema`

### `app/api/v1/search.py`
- Removed the `payload_schema: str = "contextual"` query param from the `/hybrid`, `/vector`, `/bm25` endpoints
- `payload_schema` is now passed uniformly in the `SearchRequest` body

### `app/schemas.py` (follow-up fix)
- `PayloadSchema.bm25_fields` changed from `Optional[List[str]] = None` to `List[str] = []`
- `PayloadSchema.nested_paths` changed from `Optional[List[str]] = None` to `List[str] = []`
- Prevents a `TypeError` being raised in `_build_bm25_clauses`'s `for field in fields` when a new schema doesn't specify these two fields

### `app/services/search.py` (follow-up fix)
- Removed the redundant `hasattr(settings, 'RRF_K_FACTOR')` check from `rrf_fusion`'s `k` parameter, using `settings.RRF_K_FACTOR` directly

### `README.md`
- Updated the `payload_schema` description: now a request-body field for search endpoints, still a query param for ingest endpoints
- Added an "Adding a Custom Payload Schema" section, explaining that only `schemas.py` needs to change
- Added a `PayloadSchema` field reference table (including `skip_status_filter` / `skip_req_filters`)
- Updated the API endpoint example request bodies

---

## 2026-04-10 Bug Fixes

### `app/schemas.py`
**Problem:** `PayloadSchema` uses `Callable` as a Pydantic field, which raises `PydanticUserError` at import time under Pydantic v2's default settings, because `DEFAULT_EXTRACTORS` is constructed at module level.

**Fix:**
- Added `from pydantic import ConfigDict`
- Added `model_config = ConfigDict(arbitrary_types_allowed=True)` to `PayloadSchema`

---

### `app/services/opensearch_service.py`
**Note:** This docker-compose's OpenSearch 3.4.0 uses default security (HTTPS enabled), so `use_ssl=True` is the correct setting and was not changed. `verify_certs=settings.VERIFY_CERTS` (default `False`) handles self-signed certs, and `ssl_show_warn=False` suppresses the warning.

---

### `app/services/search.py`
**Problem:** The `hybrid_search_by_opensearch` method calls `self._get_bm25_fields(payload_schema)`, but `SearchService` has no such method, raising `AttributeError`. (This method is currently never called by any API endpoint — it was dead code with a latent bug.)

**Fix:**
- `self._get_bm25_fields(payload_schema)` → `self.schema_manager.get_bm25_fields(payload_schema)[0]`
- `get_bm25_fields` returns a `(fields, nested_paths)` tuple, `[0]` takes the fields list

---

### `app/core/embedding.py`
**Problem:** `encode_async`'s type annotation is `list[str]`, but the caller (the search router) passes a single string, `req.query: str`. Although `SentenceTransformer.encode` accepts both, the inconsistent annotation is misleading.

**Fix:**
- `text: list[str]` → `text: str | list[str]`
