# Module 26 — Memory Service

Raptor 0.4's persistent memory layer. Each user has an independent per-session conversation record (`.mv2`) and cross-session long-term memory (`.mv2`), with semantic search provided via BM25 + HNSW vector indexing.

---

## Table of Contents

- [Architecture](#architecture)
- [Environment Configuration](#environment-configuration)
- [Starting the Service](#starting-the-service)
- [API Reference](#api-reference)
  - [Session Memory](#session-memory-memorysessions)
  - [Long-term Memory](#long-term-memory-memorylongterm)
  - [Multimedia Memory](#multimedia-memory-memorymultimedia)
  - [Search API](#search-api)
  - [Timeline / Management](#timeline--management)
  - [Memory Extraction (Automatic)](#memory-extraction-automatic)
  - [Memory Compaction](#memory-compaction)
- [Testing](#testing)
  - [Unit Tests](#unit-tests-no-ollama--redis--nfs-required)
  - [Latency Benchmark](#latency-benchmark)
  - [Manual Testing (curl)](#manual-testing--module-15-integration-verification)
- [GUI](#guidemo_guipy)
- [Module Dependencies](#module-dependencies)

---

## Architecture

```
User request
    │
    ▼
Module 13 (API Gateway)  ← JWT verification
    │
    ▼
Module 26 (Memory Service :8026)
    │
    ├── /memory/sessions/*    → session_{id}.mv2          per-turn conversation record
    ├── /memory/longterm/*    → long_term_N.mv2           cross-session facts / preferences (sharded)
    ├── /memory/multimedia/*  → media/{type}_{hash}.mv2   video / audio / image index
    ├── /memory/search        → global search (sessions + longterm + multimedia)
    ├── /memory/stats         → per-user memory statistics
    └── /memory/export        → full memory export (streamed JSON)
```

```
Each conversation turn's input
    │
    ├── [Tier 1] Redis  last 5 turns · 1-hour TTL · millisecond-level reads
    │
    └── [Tier 2] MemVID .mv2
          ├── session_{id}.mv2    full conversation record, BM25 + HNSW full-text search
          └── long_term_N.mv2     important facts / preferences / entities, BAAI/bge-m3 local semantic search
```

---

## Environment Configuration

| Variable | Default | Description |
|---|---|---|
| `MEM_STORAGE_ROOT` | `/mnt/memvid-storage` | `.mv2` root directory (NFS mount point) |
| `MEM_EMBEDDING_MODEL` | `BAAI/bge-m3` | SentenceTransformer local embedding model (HuggingFace name) |
| `MEM_REDIS_HOST` | `raptor-redis-standalone` | Redis hostname |
| `MEM_REDIS_PORT` | `6379` | Redis port |
| `MEM_REDIS_PASSWORD` | — | Redis password |
| `MEM_EXTRACTION_MODEL` | `qwen2.5:7b` | LLM name used for extraction (must be registered in Module 07 / Ollama) |
| `MEM_MODULE07_URL` | `http://raptor-ai-lifecycle-api:8010` | Module 07 AI Lifecycle API address (extraction + ASR) |
| `PORT_MEMORY_SERVICE` | `8026` | Public service port |
| `MEM_COMPACT_BUFFER_TOKENS` | `13000` | Safety buffer for compaction, reserved for the system prompt / token-count estimation error |
| `MEM_COMPACT_RESERVED_OUTPUT_TOKENS` | `20000` | Space reserved for the LLM's output response |
| `MEM_COMPACT_MIN_TAIL_TOKENS` | `10000` | Minimum token count for the tail (retained region); extends further back if not met |
| `MEM_COMPACT_MIN_TEXT_MESSAGES` | `5` | Minimum number of conversation turns retained in the tail |
| `MEM_COMPACT_MAX_TAIL_TOKENS` | `40000` | Maximum token count for the tail; only the excess beyond this is summarized |
| `MEM_COMPACT_SUMMARY_MAX_TOKENS` | `12000` | Maximum tokens when the LLM generates a summary |
| `MEM_COMPACT_MAX_SECTION_TOKENS` | `2000` | Cap per `# heading` section of an existing summary; only truncated if it exceeds this before being reused for incremental summarization |
| `MEM_COMPACT_KEEP_TURNS` | `10` | The most recent complete turns that are always retained, regardless of token budget |
| `MEM_COMPACT_TOOL_RESULT_RETENTION_HOURS` | `24.0` | Time window (hours) during which a tool/result pair is always retained; only eligible for summarization after this |

---

## Starting the Service

### Production deployment

```bash
cd deployment/modules
python build.py -m 26 --build
```

### Direct local start (for development)

```bash
conda activate CIE
pip install -r deployment/modules/26-memory-service/requirements.txt
```

```bash
cd deployment/modules/26-memory-service

MEM_STORAGE_ROOT=/tmp/mv_demo \
MEM_REDIS_HOST=localhost \
MEM_REDIS_PORT=6381 \
MEM_REDIS_PASSWORD=dht888888 \
MEM_MODULE07_URL=http://localhost:8010 \
PYTHONPATH=app \
python -m uvicorn main:app --host 0.0.0.0 --port 8026
```

```bash
curl http://localhost:8026/health
# {"status":"ok","memvid_version":"2.0.160"}
```

`GET /metrics` returns Prometheus-format metrics (generic HTTP request count/latency from
`prometheus-fastapi-instrumentator`, plus custom business metrics for compaction:
`memory_compact_pre_tokens`, `memory_compact_post_tokens`, `memory_compact_turns_compacted`,
`memory_compact_turns_kept`, `memory_compact_summary_write_seconds`,
`memory_compact_llm_call_seconds{path=}`, `memory_compact_llm_output_tokens{path=}`,
`memory_compact_threshold_exceeded_total`).

---

## API Reference

Every endpoint requires a `user_id` in the `X-User-ID` header.

### Session Memory (`/memory/sessions`)

| Method | Path | Description |
|---|---|---|
| `POST` | `/memory/sessions/{session_id}/turns` | Write one conversation turn |
| `GET` | `/memory/sessions/{session_id}/recent?n=10` | Get the last N turns (ascending time order) |
| `GET` | `/memory/sessions` | List all sessions for the current user |
| `DELETE` | `/memory/sessions/{session_id}` | Delete a session (`404` if it doesn't exist) |

**POST `/memory/sessions/{session_id}/turns`**

```json
{
  "user_message": "Hi, I'd like to look up the project we discussed last time",
  "assistant_response": "Based on the previous record, last time we discussed...",
  "search_results": [],
  "tool_calls": [],
  "timestamp": 1718000000.0
}
```

---

### Long-term Memory (`/memory/longterm`)

| Method | Path | Description |
|---|---|---|
| `POST` | `/memory/longterm/facts` | Write a fact / preference / entity |
| `GET` | `/memory/longterm/facts` | List all `preference` + `fact` + `entity` entries (descending time order) |
| `POST` | `/memory/longterm/search` | BM25 + HNSW semantic search (supports Unix-timestamp date filtering) |
| `DELETE` | `/memory/longterm` | Clear all long-term memory (`404` if none exists) |

**POST `/memory/longterm/facts`**

```json
{
  "text": "User prefers responses in Traditional Chinese",
  "frame_type": "preference",
  "session_id": "sess_abc"
}
```

`frame_type` allowed values: `conversation` | `preference` | `entity` | `fact`

**POST `/memory/longterm/search`**

```json
{
  "query": "user's preferred language setting",
  "top_k": 5,
  "from_date": 1700000000.0,
  "to_date": 1750000000.0
}
```

`from_date` / `to_date` are Unix timestamps, optional.

---

### Multimedia Memory (`/memory/multimedia`)

| Method | Path | Description |
|---|---|---|
| `POST` | `/memory/multimedia/video` | Index a video moment (transcript + time range) |
| `POST` | `/memory/multimedia/audio` | Index an audio clip (omitting `transcription` automatically calls Module 07 ASR) |
| `POST` | `/memory/multimedia/image` | Index an image (description + OCR text) |
| `POST` | `/memory/multimedia/search` | Cross-media semantic search (filterable by `media_type`) |

**POST `/memory/multimedia/video`**

```json
{
  "asset_path": "videos/cooling_system.mp4",
  "version_id": "abc123",
  "start_sec": 120.0,
  "end_sec": 145.0,
  "transcription": "The cooling system lowers engine temperature by circulating coolant",
  "session_id": "sess_001",
  "context_query": "look up how the cooling system works"
}
```

---

### Search API

| Method | Path | Description |
|---|---|---|
| `POST` | `/memory/sessions/{session_id}/search` | Hybrid search within a single session |
| `POST` | `/memory/search` | Global search across all sessions + long-term + multimedia |

**POST `/memory/sessions/{session_id}/search`**

```json
{
  "query": "cooling system",
  "top_k": 5,
  "from_date": "2026-01-01T00:00:00Z",
  "to_date": "2026-12-31T23:59:59Z"
}
```

`from_date` / `to_date` are ISO 8601 strings, optional.

**POST `/memory/search`** (global)

```json
{
  "query": "engine cooling",
  "top_k": 5,
  "scope": ["sessions", "longterm", "multimedia"]
}
```

`scope` defaults to all; can be narrowed to any subset.

---

### Timeline / Management

| Method | Path | Description |
|---|---|---|
| `GET` | `/memory/sessions/{session_id}/timeline` | Paginated browsing of conversation history; supports time-travel |
| `GET` | `/memory/stats` | Memory statistics |
| `GET` | `/memory/export` | Streamed export of full memory (JSON) |
| `DELETE` | `/memory` | GDPR erasure: deletes all memory for the current user |

**GET `/memory/sessions/{session_id}/timeline`**

Query parameters:

| Parameter | Default | Description |
|---|---|---|
| `page` | `1` | Page number (1-based) |
| `page_size` | `20` | Items per page (max 100) |
| `at` | — | ISO 8601 timestamp; when set, only returns frames **before** that point in time (time-travel) |

**GET `/memory/stats`**

```json
{
  "session_count": 12,
  "total_turns": 347,
  "total_media_items": 23,
  "long_term_frame_count": 58,
  "storage_bytes_used": 4194304
}
```

**GET `/memory/export`**

Returns streamed JSON in a stable, versioned format:

```json
{
  "export_schema_version": "1.0",
  "user_id": "user_abc",
  "exported_at": "2026-07-02T10:00:00+00:00",
  "sessions": [
    {"session_id": "sess_001", "turns": [...]}
  ],
  "longterm": [...],
  "multimedia": [...]
}
```

`multimedia` contains only metadata (asset_path, timestamps, text), not the original media files.

**DELETE `/memory`**

Permanently deletes all of the current user's `.mv2`, `.meta.json`, and Redis index keys. Returns `204 No Content`.

---

### Memory Extraction (Automatic)

After every conversation write (`POST /sessions/{id}/turns`), the system runs an LLM extraction pass on that turn **automatically in the background**, without blocking the response.

#### Extraction flow

```
append_turn() writes to .mv2
    │
    └─ asyncio.create_task (fire-and-forget)
         │
         ├─ read the user's existing long-term facts (up to 10, with frame_id)
         │
         ├─ send to Module 07 /inference/infer (extraction_model, temperature=0)
         │     engine: ollama; system prompt: output a list of ADD / DELETE / UPDATE JSON operations
         │
         └─ execute the operations
               ADD    → add a new preference / entity / fact
               DELETE → soft-delete an old frame (removed from the search index)
               UPDATE → DELETE the old frame + ADD the new version
```

#### Cross-session persistence

Extraction results are stored in `user_{id}/long_term_N.mv2`, independent of session.
Preferences a user states in any session accumulate and can override each other (UPDATE/DELETE handle contradicting preferences).

```
Session A (turn 3): "I prefer responses in English"
    └─ extraction → ADD preference "user prefers English"

Session C (turn 1): "switch to Traditional Chinese"
    └─ extraction → DELETE frame_42 + ADD preference "user prefers Traditional Chinese"

Session D (any): retrieve_long_term search
    └─ finds "user prefers Traditional Chinese" → included in the LLM context
```

#### Verifying extraction goes through Module 07

```bash
# record the baseline
curl -s http://localhost:8010/inference/stats | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print('total_inferences:', d['stats']['total_inferences'])"

# trigger one extraction
curl -s -X POST http://localhost:8026/memory/longterm/extract \
  -H "X-User-ID: $USER_ID" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test-sess","turn":{"user_message":"I like using Traditional Chinese","assistant_response":"Got it"}}'

# check again — the counter should be +1
curl -s http://localhost:8010/inference/stats | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print('total_inferences:', d['stats']['total_inferences'])"
```

The memory service log will also show:
```
Module 07 extraction call succeeded (model=qwen2.5:7b, processing_time=X.XXXs)
```

#### Internal endpoint (not exposed via the Gateway)

| Method | Path | Description |
|---|---|---|
| `POST` | `/memory/longterm/extract` | Manually trigger extraction (requires the `X-User-ID` header; does not accept `user_id` in the body) |

**POST `/memory/longterm/extract`**

```json
{
  "session_id": "sess_001",
  "turn": {
    "user_message": "I prefer Traditional Chinese",
    "assistant_response": "Understood, I've noted your preference."
  }
}
```

> `user_id` is taken by the service from the `X-User-ID` header — the body does not need, and does not accept, a `user_id` field.

`stored` is the number of ADD/UPDATE operations actually executed this call (DELETE is also counted).

---

### Memory Compaction

When a single session's conversation history exceeds the LLM's context window, older conversation turns are summarized into a `summary` frame, while a high-fidelity recent tail is preserved, to avoid an overlong prompt.

| Method | Path | Description |
|---|---|---|
| `POST` | `/memory/compact/evaluate` | Estimates the token usage of the `messages` passed in the request body, returns whether compaction is needed (writes nothing) |
| `POST` | `/memory/sessions/{session_id}/compact/evaluate` | Aggregates the session's turns + long-term facts + multimedia to estimate actual token usage (writes nothing) |
| `POST` | `/memory/sessions/{session_id}/compact` | Compacts a session, writing a summary frame |
| `GET` | `/memory/sessions/{session_id}/summaries` | Lists all summary frames for the session |
| `DELETE` | `/memory/sessions/{session_id}/summaries/{summary_id}` | Deletes a summary frame (`404` if it doesn't exist) |

#### Decision logic

```
threshold = context_window - MEM_COMPACT_RESERVED_OUTPUT_TOKENS - MEM_COMPACT_BUFFER_TOKENS

session's total token count < threshold → no compaction (source="under_budget")
session's total token count ≥ threshold → trigger compaction:
    1. Find the last compaction boundary (last_summarized_frame_id, or the source_turn_range.to
       of the most recent summary frame)
    2. Starting from the turns after the boundary, accumulate backward until
       MEM_COMPACT_MIN_TAIL_TOKENS / MEM_COMPACT_MIN_TEXT_MESSAGES is satisfied, but not
       exceeding MEM_COMPACT_MAX_TAIL_TOKENS → this segment is kept verbatim (the tail)
    3. Turns after the boundary but outside the tail → handed to the LLM to summarize
         - If a summary frame already exists with non-empty content that isn't just a bare
           heading skeleton (quality gate), prefer "incremental summarization" (the existing
           summary is truncated per-section by MEM_COMPACT_MAX_SECTION_TOKENS, then combined
           with the new turns → an updated summary)
         - If the existing summary is an empty string or just a `# heading` with no content
           (e.g. the LLM previously returned something malformed), it's treated as unusable
           and a completely fresh LLM summary is generated instead, with no incremental attempt
         - If the LLM call fails, fail open: return compacted=false, source="llm_failed",
           without affecting the existing conversation
    4. The summary result is written back into the session's .mv2 as a frame with
       frame_type="summary", plus an additional frame_type="compact_boundary" frame recording
       this compaction's preserved_segment (head/anchor/tail frame_id)
```

**POST `/memory/compact/evaluate`**

Only evaluates the `messages` passed in the request body — it does not itself query Redis / the session's `.mv2` / long-term / multimedia to aggregate. This is a general-purpose version for callers that already have their own message list; to evaluate the actual archived volume for a specific session, use the session-scoped version below instead.

```json
{
  "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
  "context_window": 128000,
  "extra_tokens": 0
}
```

**POST `/memory/sessions/{session_id}/compact/evaluate`**

Aggregates the token count of three things for the session — its already-archived turns, the user's long-term facts (`get_facts`, not an embedding search), and the multimedia index (`list_all`, not an embedding search) — returning a `CompactEvaluation` in the same format as above. `extra_tokens` is used to account for the current turn's content that hasn't been written to any store yet (e.g. the user's current question). Module 21 calls this endpoint before assembling a grounded prompt; when `should_compact=false` it skips the `/compact` call outright, and only calls `/compact` when `should_compact=true` or the call itself fails (fail open).

```json
{
  "context_window": 128000,
  "extra_tokens": 0
}
```

**POST `/memory/sessions/{session_id}/compact`**

```json
{
  "trigger": "auto",
  "context_window": 128000,
  "last_summarized_frame_id": null,
  "custom_instructions": null,
  "dry_run": false
}
```

`source` may be: `under_budget` (threshold not reached), `no_session`, `no_turns`, `nothing_to_compact`, `session_memory` (incrementally updated an existing summary), `llm_compact` (a completely fresh LLM summary), `llm_failed` (LLM failed, fail open, not compacted).

`dry_run=true` returns only the evaluation result, without writing a summary frame.

`threshold_exceeded`: whether the token count is still ≥ threshold after compaction (including summary-truncation attempts and other reduction efforts). `true` means fail-open — the summary has already been trimmed as much as possible to fit within the remaining budget of `threshold - tail_tokens`, but the tail itself is already at or beyond the threshold (e.g. `context_window` is set smaller than `reserved_output + buffer`), so the target can't be fully met; the caller may need to trim further itself or adjust `context_window`. This does not mean compaction failed — `compacted` is still `true` and the summary frame is still written.

#### Module 15 / Module 21 integration

Both sides already call Module 26's `/memory/sessions/{session_id}/compact` (which preserves summaries and is auditable), rather than only doing local truncation. Local truncation is only a fail-open fallback for when Module 26 is unavailable.

Module 15 calls `/compact` directly, relying on that endpoint's internal threshold check to no-op when unnecessary. Module 21 instead first calls
`/memory/sessions/{session_id}/compact/evaluate`: it skips `/compact` when `should_compact=false`;
it calls `/compact` only when `should_compact=true`, or when the evaluate call itself fails
(returns `None`, fail open) — still backstopped by `/compact`'s own internal threshold check, so a
transient evaluate failure won't cause a genuinely-needed compaction to be missed.

Module 15 (Chat Service): the LangGraph flow is `load_memory → retrieve_long_term → retrieve_session_history → compact_context → prepare_context → ... → save_memory → archive_memory`.

- `retrieve_long_term`: calls `/memory/longterm/search` before answering, pulling cross-session preferences/facts into the system prompt, fail open.
- `retrieve_session_history`: calls `/memory/sessions/{id}/search`, semantically searching conversation further back than Redis's `context_window`.
- `archive_memory`: fire-and-forget write to `/memory/sessions/{id}/turns` after each turn completes.
- `compact_context`: first calls `MemoryClient.compact_session()` → Module 26's `/memory/sessions/{session_id}/compact`; on successful compaction, the summary text retrieved via `get_latest_summary()` is inserted at the very front of `context_window` with a `[history summary]` prefix. When Module 26 is unavailable, there's no session to compact, or it returns "not compacted," it falls back to trimming Redis's short-term `context_window` from oldest to newest based on `COMPACT_CONTEXT_BUDGET` (default 20000 tokens, estimated via `chars/4`); fail open throughout.

Module 21 (Agent Protocol): Step 7 of `run_rag_pipeline` calls
`MemoryClient.evaluate_compact()` → Module 26's `/memory/sessions/{session_id}/compact/evaluate`
before assembling the grounded prompt. If it returns `should_compact=false`, this round's
`compact_session()` call is skipped; if it returns `should_compact=true`, or evaluate itself
fails (`None`, fail open), then `MemoryClient.compact_session()` +
`get_latest_summary()` are called, and on successful compaction the summary is inserted at the
front of `memory_lines` with a `[history summary]` prefix, combined with the long-term/multimedia
retrieval results into the prompt; when `session_id` is omitted it falls back to using `user_id` as
the session key. A call failure only logs a warning and doesn't affect the answer (fail open).
`COMPACT_PROMPT_BUDGET_TOKENS` (default 80000) is an independent second line of defense that trims
the retrieved chunks, not the session history.

Each integration has its own unit tests:

```bash
# Module 15: ChatService._compact_context() (covers Module 26 success/declined/error paths) + retrieve_long_term / MemoryClient.search_longterm, compact_session, get_latest_summary
cd deployment/modules/15-chat-service
python -m pytest test_compact_context.py test_retrieve_long_term.py -v

# Module 21: MemoryClient (including compact_session/get_latest_summary fail-open) + pipeline.py Step 5's chunk token-budget trimming
cd deployment/modules/21-agent-protocol
python -m pytest test_memory_integration.py test_compact_budget.py -v
```

---

## Testing

### Unit Tests (no Ollama / Redis / NFS required)

```bash
cd deployment/modules/26-memory-service

# full quick test suite
conda run -n CIE python -m pytest \
  test_memvid_store.py \
  test_session_memory.py \
  test_api.py \
  test_long_term_memory.py \
  test_multimedia_memory.py \
  test_search.py \
  test_timeline_export.py \
  test_extraction.py \
  test_compact_memory.py \
  -v -m "not slow"

# a single module
conda run -n CIE python -m pytest test_search.py -v -m "not slow"
conda run -n CIE python -m pytest test_timeline_export.py -v
conda run -n CIE python -m pytest test_extraction.py -v
conda run -n CIE python -m pytest test_compact_memory.py -v
```

| Test file | # tests | Coverage |
|---|---|---|
| `test_memvid_store.py` | 9 | `.mv2` CRUD, BM25 search, timeline |
| `test_session_memory.py` | 8 | Redis sorted set, turn index, delete |
| `test_api.py` | 8 | HTTP status codes, response schema, 404 |
| `test_long_term_memory.py` | 15 | 100 frames, type filtering, date range, deletion |
| `test_multimedia_memory.py` | 12 | video/audio/image indexing, cross-media search, metadata, type filtering, `list_all` |
| `test_search.py` | 18 (+2 slow) | Session search (date filtering, user isolation), global search (scope, total_frames_searched), API integration |
| `test_timeline_export.py` | 23 | Timeline pagination, time-travel, stats, export JSON format, GDPR deletion, API integration |
| `test_extraction.py` | 23 | `_parse_ops` for each operation type, ADD/DELETE/UPDATE execution, LLM failure doesn't crash, background trigger |
| `test_compact_memory.py` | 11 | Token estimation, tail-retention-range calculation, compaction trigger threshold, summary frame read/write/delete, LLM failure fail-open |

**Total: 124 unit / integration tests**

---

### Latency Benchmark

Run separately from the unit tests (larger seed data volume, takes longer):

```bash
# Long-term search SLA (seeds 10,000 frames, ~5 minutes)
conda run -n CIE python -m pytest test_latency.py -v -m slow

# Search SLA (seeds 50 sessions × 100 turns, ~12 minutes)
conda run -n CIE python -m pytest test_search.py -v -m slow
```

| Test | Scale | SLA |
|---|---|---|
| `test_search_latency_under_200ms` | 10,000 long-term frames, 5 query types | P95 < 200ms |
| `test_search_latency_worst_case_top50` | 10,000 frames, top_k=50 | < 200ms |
| `test_search_latency_with_date_filter` | 10,000 frames, with date filtering | < 200ms |
| `test_session_search_latency_50_sessions` | 50 sessions × 100 turns | P95 < 300ms |
| `test_global_search_latency_50_sessions` | 50 sessions + 50 long-term facts | P95 < 300ms |

---

### Manual Testing / Module 15 Integration Verification

For complete curl examples covering manual testing of every endpoint and verifying Module 15's integration, see [TESTING.md](TESTING.md).

---

## GUI (demo_gui.py)

```bash
conda run -n CIE streamlit run demo_gui.py
```

Open `http://localhost:8501` in a browser

### Tab overview

| Tab | Function |
|---|---|
| Session Memory | Write conversation turns, get the last N turns, list sessions, delete a session |
| Long-term Memory | Write a fact/preference, list facts, semantic search, clear all |
| Multimedia Memory | Index video/audio/image, cross-media search |
| **Search** | Session search (ISO 8601 date filtering), global search (multi-select scope) |
| **Timeline / Management** | Timeline pagination + time-travel, stats cards, export JSON download, GDPR clear |
| **Memory Extraction** | Manually trigger extraction, view existing long-term facts (with frame_id), explanation of the automatic extraction mechanism |
| **Memory Compaction** | Quickly generate test data, estimate token usage, trigger compaction and view the summary frame |

### End-to-end test flow

1. Sidebar → Health Check (confirm ✅) → Login (get a token)
2. **Session Memory** → write 3 turns into `sess_test_001`
3. **Long-term Memory** → write `User prefers responses in Traditional Chinese` (preference), `User's car model is a Toyota Camry 2022` (fact)
4. **Search → Session Search** → query=`engine cooling`, expect turn #1 to score highest, `total_frames_searched=3`
5. **Search → Global Search** → query=`engine`, all scopes selected, expect results from both sessions + longterm
6. **Timeline / Management → Timeline** → `sess_test_001`, page_size=2, confirm `has_next=true`
7. **Timeline / Management → Timeline** → check Time-Travel, set the time to today at `00:00:00`, confirm `total=0`
8. **Timeline / Management → Stats** → confirm `session_count=1`, `total_turns=3`
9. **Timeline / Management → Export** → download JSON, confirm `export_schema_version="1.0"`
10. **Timeline / Management → Clear** → confirm and clear, check Stats to confirm everything is zeroed
11. **Memory Extraction → Manual Trigger** → enter conversation content, click "Trigger Extraction," confirm `stored >= 1`
12. **Memory Extraction → Reload** → view the extracted preference / fact, confirm it includes a frame_id

---

## Module Dependencies

| Module | Purpose |
|---|---|
| 01 NFS Server | Persistent storage for `.mv2` + `.meta.json` |
| 02 Redis | Session sorted-set index |
| 07 AI/ML | `/inference/infer` (port 8010) calls `qwen2.5:7b` for preference extraction; `BAAI/bge-m3` SentenceTransformer loaded locally for HNSW vector embedding |
| 13 API Gateway | JWT verification; `/api/0.3/sso/login` to get a token (port 8012) |
