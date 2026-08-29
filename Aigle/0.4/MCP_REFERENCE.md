# Raptor 0.4 MCP Reference

How to talk to Raptor over the [Model Context Protocol](https://modelcontextprotocol.io/) — a different shape of API from [`API_REFERENCE.md`](API_REFERENCE.md)'s plain REST endpoints, documented separately for that reason. This is the interface definition (every tool's, resource's, and prompt's exact schema) plus a user guide for connecting a client — aimed at anyone building an LLM agent, IDE integration, or script against Raptor via MCP rather than raw REST.

Module 27 (`27-mcp-server`) is the actual MCP server — it exposes Raptor's search/chat/upload/graph/memory capabilities as MCP tools, resources, and prompts. Module 13's gateway proxies it at `/api/{version}/mcp` so an MCP client never needs to reach module 27 directly.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Two ways to authenticate](#auth)
- [The protocol endpoint](#endpoint)
- [Tool Catalog](#tools)
  - [Search Tools](#tools-search)
  - [Asset Management Tools](#tools-asset)
  - [Conversational & RAG Tools](#tools-rag)
  - [Knowledge Graph Tools](#tools-graph)
  - [Memory Tools](#tools-memory)
- [Resources](#resources)
- [Prompts](#prompts)
- [Agent registration (autonomous agents)](#agent-registration)
- [Error handling & notes for client authors](#notes)

---

<a id="getting-started"></a>

## Getting Started

Pick one of three ways to connect, depending on what you're building.

### 1. Official MCP SDK (recommended for a real client)

Raptor speaks standard MCP Streamable HTTP — any client built on an official SDK works unmodified, you only need to supply the URL and a Bearer token.

**Python** (`pip install mcp`) — adapted from [`deployment/modules/27-mcp-server/examples/python_client.py`](deployment/modules/27-mcp-server/examples/python_client.py), runnable as-is:

```python
import asyncio, json
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

JWT = "<bearer token from POST /api/0.4/sso/login>"
SERVER_URL = "http://<host>:8012/api/0.4/mcp"  # via the gateway; or http://<mcp_host>:8027/mcp direct

def text_of(result) -> str:
    return "".join(block.text for block in result.content if hasattr(block, "text"))

async def main():
    async with streamablehttp_client(SERVER_URL, headers={"Authorization": f"Bearer {JWT}"}) as (read, write, _):
        async with ClientSession(read, write) as session:
            init_result = await session.initialize()
            print(f"Connected to {init_result.serverInfo.name} v{init_result.serverInfo.version}")

            tools = await session.list_tools()
            print(f"Available tools ({len(tools.tools)}): {[t.name for t in tools.tools]}")

            result = await session.call_tool("raptor_search", {"query": "video", "top_k": 3})
            for hit in json.loads(text_of(result)):
                print(f"  [{hit.get('score', 0):.3f}] {hit.get('asset_path', '')}")

            capabilities = await session.read_resource("raptor://capabilities")
            print(capabilities.contents[0].text[:200], "...")

asyncio.run(main())
```

**TypeScript** (`npm install @modelcontextprotocol/sdk tsx`) — same shape, see [`deployment/modules/27-mcp-server/examples/typescript_client.ts`](deployment/modules/27-mcp-server/examples/typescript_client.ts):

```bash
npx tsx examples/typescript_client.ts --jwt <token> --server-url http://<host>:8012/api/0.4/mcp
```

### 2. Raw JSON-RPC over curl (no SDK, for debugging)

See [The protocol endpoint](#endpoint) below for the full `initialize` → `notifications/initialized` → `tools/list` → `tools/call` handshake, or run [`deployment/modules/27-mcp-server/examples/curl_mcp.sh`](deployment/modules/27-mcp-server/examples/curl_mcp.sh), which can log in for you (`KEYCLOAK_USERNAME=... KEYCLOAK_PASSWORD=... ./curl_mcp.sh`).

> The script's `GATEWAY_BASE_URL` default is stale (still points at the retired `raptor_open_0_3_api` host) — pass `GATEWAY_BASE_URL=http://<your_host>:8012` explicitly rather than relying on the default.

### 3. stdio / local process (e.g. Claude Desktop)

Module 27 also runs as a local subprocess speaking MCP over stdio (`MCP_TRANSPORT=stdio python -m app`, run from `deployment/modules/27-mcp-server/` with its `requirements.txt` installed) — the shape an app like Claude Desktop expects for a locally-configured MCP server:

```json
{
  "mcpServers": {
    "raptor": {
      "command": "python",
      "args": ["-m", "app"],
      "cwd": "/path/to/deployment/modules/27-mcp-server",
      "env": {
        "MCP_TRANSPORT": "stdio",
        "MCP_KEYCLOAK_USERNAME": "<service_account_username>",
        "MCP_KEYCLOAK_PASSWORD": "<service_account_password>",
        "MCP_API_GATEWAY_URL": "http://<host>:8012/api/0.4"
      }
    }
  }
}
```

> **Important — this is a shared identity, not per-user.** stdio has no HTTP layer, so there is no per-call `Authorization` header to forward. The whole subprocess authenticates once as a single configured Keycloak service account (`MCP_KEYCLOAK_USERNAME`/`MCP_KEYCLOAK_PASSWORD`) for its entire life — every tool call any user makes through this local server acts under that one identity, not their own. Don't wire this up for multiple people expecting their own isolated Module 25 data; use the HTTP transport with each person's own Bearer token for that instead.

---

<a id="auth"></a>

## Two ways to authenticate

**1. Human — Bearer token, same as every other endpoint in `API_REFERENCE.md`.** Works for everything below, including the main protocol endpoint. Fine for testing or a client that already manages its own token lifecycle.

**2. Autonomous agent — register once, then reuse an `agent_token`.** For an agent that runs unattended and can't re-authenticate interactively. A human registers the agent's Keycloak service-account credentials once; the MCP server then refreshes the underlying token itself via `client_credentials`, so the agent never has to log in again. See [Agent registration](#agent-registration) below. A registered agent's token is always prefixed `mcp-agent-` — that's how the server tells it apart from a raw Keycloak JWT and routes it through the stored-credential refresh path instead of forwarding it as-is.

**Over HTTP transport there is no third, silent option.** A tool call with no `Authorization` header at all is rejected outright (`"Missing Authorization: Bearer <token> — MCP tool calls require a caller JWT"`) — it never falls back to acting as the server's own service account. (That fallback only exists for [stdio transport](#getting-started) and for the public [resource](#resources) reads below, both deliberately.)

---

<a id="endpoint"></a>

## The protocol endpoint

### `POST /api/{version}/mcp`

Raw byte-for-byte proxy to Module 27's Streamable HTTP transport — no gateway-level JWT/UMA check (Module 27 extracts and validates the bearer token itself). Every call needs:

```
Authorization: Bearer <token>
Content-Type: application/json
Accept: application/json, text/event-stream
```

Responses come back as Server-Sent Events (`event: message` / `data: {...}`), whether the body is JSON or streamed — the gateway passes both through unmodified.

### Handshake (live-tested end to end against the real deployment)

**1. `initialize`** — no session header yet; the response carries the session id you'll need for every call after this one.

```bash
TOKEN=$(curl -s -X POST http://<host>:8012/api/0.4/sso/login -d "username=<user>&password=<pass>" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['access_token'])")

curl -s -D headers.txt -X POST http://<host>:8012/api/0.4/mcp \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{
        "protocolVersion":"2024-11-05","capabilities":{},
        "clientInfo":{"name":"my-client","version":"1.0"}}}'

SESSION=$(grep -i '^mcp-session-id:' headers.txt | cut -d' ' -f2 | tr -d '\r')
```

Real response body:

```
event: message
data: {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05","capabilities":{"experimental":{},"prompts":{"listChanged":false},"resources":{"subscribe":false,"listChanged":false},"tools":{"listChanged":false}},"serverInfo":{"name":"raptor-mcp","version":"1.29.1"}}}
```

Real response header carrying the session id: `mcp-session-id: 93ef18d70eec46eda7ae2b11615d6e8a`.

**2. `notifications/initialized`** — required handshake step before any other call; every request from here on needs `Mcp-Session-Id`.

```bash
curl -s -X POST http://<host>:8012/api/0.4/mcp \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" -H "Mcp-Session-Id: $SESSION" \
  -d '{"jsonrpc":"2.0","method":"notifications/initialized"}'
# → HTTP 202, empty body — this is correct, not an error
# (the mcp SDK's own handshake response; not one of Module 13/27's routes)
```

**3. `tools/list`** — discover what's callable.

```bash
curl -s -X POST http://<host>:8012/api/0.4/mcp \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" -H "Mcp-Session-Id: $SESSION" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list"}'
```

**4. `tools/call`** — invoke one. Real, live-tested example and its actual (trimmed) response:

```bash
curl -s -X POST http://<host>:8012/api/0.4/mcp \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" -H "Mcp-Session-Id: $SESSION" \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{
        "name":"raptor_list_assets","arguments":{"page":1,"page_size":3}}}'
```

```
event: message
data: {"method":"notifications/message","params":{"level":"info","data":"raptor_list_assets: page=1 page_size=3 keyword=None"},"jsonrpc":"2.0"}

event: message
data: {"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"{\n  \"total_count\": 8,\n  \"page\": 1,\n  \"commits\": [{\"asset_path\": \"image/jpg/extracted_frame\", \"primary_filename\": \"extracted_frame.jpg\", \"status\": \"active\", ...}]\n}"}]}}
```

A `tools/call` typically streams one or more `notifications/message` progress events before the final `result` event — a client needs to keep reading the SSE stream until it sees the `result` with the matching `id`, not stop at the first event.

**Resources and prompts use the same envelope**, with `resources/list`, `resources/read` (`params: {"uri": "raptor://..."}`), `prompts/list`, and `prompts/get` (`params: {"name": "...", "arguments": {...}}`) in place of `tools/list`/`tools/call`.

---

<a id="tools"></a>

## Tool Catalog

22 tools, live-queried via `tools/list` against the real deployment (2026-08-28, after that day's Module 27 restart). Every tool's actual `inputSchema` (returned by `tools/list`) is authoritative — the tables below are a documentation layer on top of it, not a substitute for calling `tools/list` yourself.

`raptor_trigger_processing` was removed (its target route, `POST /processing/process-file`, doesn't exist on the gateway — `raptor_upload_asset` already triggers processing on upload) and the 7 `raptor_memory_*` tools were added, covering the gateway's full non-destructive Module 26 memory surface.

**Quick reference:**

| Tool | Required params | Optional params | What it does |
|---|---|---|---|
| `raptor_search` | `query` | `top_k`, `type`, `speaker`, `source`, `embedding_type` | Hybrid search across all asset types |
| `raptor_search_bm25` | `query` | `top_k`, `type`, `speaker`, `source` | Keyword-only search |
| `raptor_search_vector` | `query` | `top_k`, `type`, `speaker`, `source`, `embedding_type` | Semantic-only search |
| `raptor_video_search` | `query` | `top_k`, `asset_path`, `candidate_multiplier`, `score_threshold` | Video-specific multi-retriever search |
| `raptor_list_assets` | — | `keyword`, `start_date`, `end_date`, `page`, `page_size` | List the caller's uploaded assets |
| `raptor_get_asset_url` | `asset_path`, `version_id` | — | Presigned 24h download URL |
| `raptor_upload_asset` | `filename`, `content_base64` | `content_type`, `archive_ttl_days`, `destroy_ttl_days` | Upload + trigger AI processing |
| `raptor_check_status` | `correlation_id` | `m_type` | Poll async processing status |
| `raptor_chat` | `message` | `session_id`, `history` | Conversational RAG with memory |
| `raptor_a2a_direct` | `question` | `top_k`, `session_id` | Deterministic RAG pipeline (classify → search → rerank → answer) |
| `raptor_a2a_agent` | `question` | `top_k`, `session_id` | Agentic RAG (LLM decides which tools to call) |
| `raptor_query_orchestrate` | `query` | `top_k` | Auto-routed query through the full RAG pipeline |
| `raptor_graph_query` | `query` | `entity`, `max_depth`, `limit`, `strategy` | GraphRAG entity/relationship query |
| `raptor_tkg_query` | `query` | `time_start`, `time_end`, `max_depth`, `limit` | Temporal knowledge graph query |
| `raptor_memory_retrieve` | `query` | `top_k` | Search the caller's Module 26 memory: session history + long-term facts + multimedia, combined |
| `raptor_memory_store` | `text` | `frame_type`, `session_id` | Write a standalone memory node not tied to a session |
| `raptor_memory_timeline` | — | `page`, `page_size` | Paginated, time-ascending timeline of the caller's turns across all sessions |
| `raptor_memory_multimedia_search` | `query` | `top_k`, `media_type` | Search only the caller's indexed video/audio/image memory |
| `raptor_memory_session_summaries` | `session_id` | — | List the summary frames produced by compacting a session |
| `raptor_memory_compact` | — | `session_id`, `trigger`, `context_window`, `custom_instructions` | Summarise a session's older turns to free up context window space |
| `raptor_memory_compact_evaluate` | — | `session_id`, `context_window`, `extra_tokens` | Estimate the token budget a compaction would need, without running it |
| `raptor_memory_archive` | `user_message`, `assistant_response` | `session_id` | Archive one conversation turn into a session |

**Known stale wording:** `raptor_search_bm25`/`raptor_search_vector`'s own descriptions (surfaced verbatim by `tools/list` to any MCP client, including an LLM deciding which tool to call) still say "using OpenSearch"/"using Qdrant" — those backends were retired (see root [`README.md`](README.md#deprecated-modules)); the real backend for all `raptor_search*` tools is Module 25 (ArcadeDB). Trust this doc and `API_REFERENCE.md`'s [Search](API_REFERENCE.md#search) section over the tool's own description text until that's fixed.

**Reshaped, not raw, responses.** Unlike the REST endpoints in `API_REFERENCE.md`, most of these tools reshape the gateway's JSON before returning it to the MCP client — trimmed to what an LLM actually needs, and (for the search family) using different field names than the REST `payload`. Don't assume a tool's output matches its underlying REST endpoint's response shape byte-for-byte; the per-tool sections below document the tool's own shape.

---

<a id="tools-search"></a>

### Search Tools

All four call Module 25 (ArcadeDB) through Module 13's `/search/*` gateway routes and reshape each hit from the REST `payload` shape into a flatter one: `{id, content, score, metadata, asset_path, start_time, end_time}`, where `content` is the REST payload's `text` field and `metadata` holds whatever's left over (e.g. `filename`, `type`, `embedding_type`, `speaker`).

#### `raptor_search`

Hybrid semantic + keyword search (BM25 + vector, RRF-fused, cross-encoder-reranked) across all asset types.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `query` | string | ✓ | — | Search query text |
| `top_k` | int | | `10` | Maximum results to return (1–50) |
| `type` | string | | null | Media type filter: `videos` / `audios` / `documents` / `images` |
| `speaker` | string | | null | Filter by speaker name |
| `source` | string | | null | Filter by file format, e.g. `mp4`, `pdf` |
| `embedding_type` | string | | null | `text` or `summary` |

```json
{"name": "raptor_search", "arguments": {"query": "quarterly earnings call", "top_k": 5, "type": "documents"}}
```

```json
[
  {
    "id": "2386aa1d...",
    "content": "ocr:{...} / asr:{...} / lvlm:{...}",
    "score": 0.5000,
    "metadata": {"filename": "q3-earnings.pdf", "type": "documents", "embedding_type": "text"},
    "asset_path": "document/pdf/q3-earnings",
    "start_time": null,
    "end_time": null
  }
]
```

#### `raptor_search_bm25`

Keyword-only search. Score is a BM25 relevance score (higher = more relevant). Same parameters as `raptor_search` minus `embedding_type`. Use when exact keyword precision matters more than semantic recall.

```json
{"name": "raptor_search_bm25", "arguments": {"query": "trade tariff negotiations", "top_k": 5}}
```

#### `raptor_search_vector`

Semantic-only search. Score is cosine similarity (0–1). Same parameters as `raptor_search`. Use for conceptual/paraphrase queries where exact wording doesn't matter.

```json
{"name": "raptor_search_vector", "arguments": {"query": "AI competition between world powers", "top_k": 5}}
```

#### `raptor_video_search`

Video-specific multi-retriever search (BM25 + vector + GraphRAG + TKG → RRF → cross-encoder rerank). Returns individual matched **moments** (segments), not raw video records — reshaped from the REST endpoint's per-video/per-segment nesting into a flat, score-sorted list.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `query` | string | ✓ | — | Natural-language description of video content to find |
| `top_k` | int | | `5` | Maximum video files to consider (1–50) |
| `asset_path` | string | | null | LakeFS path to restrict results to a single video |
| `candidate_multiplier` | int | | `5` | Fan-out factor per retriever (1–20) |
| `score_threshold` | float | | `0.52` | Minimum segment score (0.0–1.0) |

```json
{"name": "raptor_video_search", "arguments": {"query": "trade summit handshake", "top_k": 5}}
```

```json
[
  {
    "clip_url": "http://<host>:8333/lakefs/data/...",
    "start_time": 0.0,
    "end_time": 10.0,
    "filename": "trade-summit-2026.mp4",
    "video_id": "d9243fa1...",
    "score": 0.7102,
    "text": "contextual:{...} / ocr:{...} / asr:{...} / lvlm:{...}",
    "sources": ["bm25", "vector"]
  }
]
```

---

<a id="tools-asset"></a>

### Asset Management Tools

#### `raptor_list_assets`

Lists assets uploaded by the authenticated user, with optional keyword and date filters.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `keyword` | string | | null | Partial filename filter (case-insensitive) |
| `start_date` | string | | null | Return assets uploaded on or after this date (ISO 8601) |
| `end_date` | string | | null | Return assets uploaded on or before this date (ISO 8601) |
| `page` | int | | `1` | Page number, 1-based |
| `page_size` | int | | `10` | Results per page (1–100) |

Returns the gateway's `/asset/users/commits` response unchanged: `total_count`, `total_pages`, and a `commits` array (each entry: `asset_path`, `version_id`, `primary_filename`, `upload_date`, `status`, `checksum`).

```json
{"name": "raptor_list_assets", "arguments": {"page": 1, "page_size": 3}}
```

#### `raptor_get_asset_url`

Gets a presigned 24-hour download URL for a specific asset version.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `asset_path` | string | ✓ | LakeFS asset path, e.g. `video/mp4/my-video`. From `raptor_list_assets`. |
| `version_id` | string | ✓ | 64-char hex version ID. From `raptor_list_assets`. |

Returns `metadata`, `primary_file` (filename, version_id, content_type, url), and `associated_file_N` entries for derived files (e.g. video frames) — same shape as `API_REFERENCE.md`'s [`GET /asset/filedownload/...`](API_REFERENCE.md#asset), unmodified.

#### `raptor_upload_asset`

Uploads a file to Raptor and triggers async AI processing. File is stored in LakeFS; a Kafka message starts the appropriate worker (video → chunking/OCR/ASR/LVLM/graph indexing; audio → diarization/ASR/classification; document → layout analysis/summarization/graph indexing; image → feature extraction/indexing).

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `filename` | string | ✓ | — | Original filename including extension, e.g. `interview.mp4` |
| `content_base64` | string | ✓ | — | File contents as standard Base64 (RFC 4648) |
| `content_type` | string | | `application/octet-stream` | MIME type, e.g. `video/mp4`, `application/pdf` |
| `archive_ttl_days` | int | | null | Days until asset is archived. Omit for permanent storage. |
| `destroy_ttl_days` | int | | null | Days until asset is deleted. Must exceed `archive_ttl_days`. |

Rejects with a plain error (not a silent truncation) if the decoded file exceeds the server's upload cap (`MCP_MAX_UPLOAD_BYTES`, 50 MB by default). Returns immediately — poll `raptor_check_status` with the returned `correlation_id` to track processing:

```json
{"asset_path": "video/mp4/interview", "version_id": "93d3389e...", "size_bytes": 10485760, "exists": false, "correlation_id": "41644469-7492-44b6-8ae5-4af3c343c035"}
```

#### `raptor_check_status`

Polls the async AI processing pipeline status for an uploaded asset. Status values: `queued` → `transcribing` → `extracting` → `indexing` → `complete` | `failed`.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `correlation_id` | string | ✓ | Returned by `raptor_upload_asset`. Poll until status is `complete` or `failed`. |
| `m_type` | string | | `document` / `video` / `image` / `audio`. Omit to auto-detect (slower — scans all types). |

---

<a id="tools-rag"></a>

### Conversational & RAG Tools

#### `raptor_chat`

Sends a message to Raptor's conversational RAG system (Module 15). Pipeline: intent classification → hybrid search → context window → LLM generation. Typical latency 10–60 seconds.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `message` | string | ✓ | User message to send to the RAG chat system |
| `session_id` | string | | Session ID for conversation continuity (Redis-backed). Omit to start a new session — the server assigns one. |
| `history` | array of `{role, content}` | | Explicit conversation history, for injecting context when server-side session memory is unavailable |

Returns `response`, `session_id`, `search_triggered`, `search_results` (present when `search_triggered` is true, same shape as the [Search Tools](#tools-search) above), and `tool_calls` when the pipeline invoked any.

#### `raptor_a2a_direct`

Deterministic RAG pipeline: intent classification → multi-path search → rerank → LLM answer. Latency 15–60 seconds. Use when you need a grounded, reproducible answer.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `question` | string | ✓ | — | Question to answer (at least 1 character) |
| `top_k` | int | | `5` | Number of chunks to retrieve (1–50) |
| `session_id` | string | | null | Reserved; not yet used by the pipeline |

Returns `answer`, `sources` (retrieved chunks), `graph_context`.

#### `raptor_a2a_agent`

Agentic RAG: a smolagents `CodeAgent` autonomously selects and calls Raptor tools, plans, searches, and synthesizes an answer with a tool-call trace. Latency 30–120 seconds — use for complex multi-hop questions. Same parameters as `raptor_a2a_direct`. Returns `answer`, `sources`, and `agent_trace` (the agent's reasoning/tool-call steps).

#### `raptor_query_orchestrate`

Auto-routed query — sends the question through Raptor's full RAG pipeline, which internally classifies intent (Module 18) and picks the best retrieval path (VideoRAG / DocumentRAG / GraphRAG / TKG / RDBMS) before generating an answer. Use when you don't want to choose a search tool yourself.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `query` | string | ✓ | — | Natural-language query to auto-route and answer |
| `top_k` | int | | `10` | Number of chunks to retrieve (1–50) |

Returns `answer`, `sources`, `graph_context`, plus `pipeline_used` and `confidence` — both currently always `null` (the underlying `/a2a/query` endpoint doesn't yet surface Module 18's classification result; reserved for when it does).

---

<a id="tools-graph"></a>

### Knowledge Graph Tools

#### `raptor_graph_query`

GraphRAG entity and relationship query — finds matched entities and expands a subgraph. Use for "Who is X?" or "How are X and Y related?" questions.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `query` | string | ✓ | — | Natural language query |
| `entity` | string | | null | Optional entity name to focus the query on |
| `max_depth` | int | | `2` | Subgraph expansion depth (1–4) |
| `limit` | int | | `50` | Maximum nodes to return (1–200) |
| `strategy` | string | | `hybrid` | `hybrid` / `literal` / `semantic` |

Returns semantic triples (`subject`–`relation`–`object`), a short summary, entity count, and co-occurring entities (capped at `MCP_COOCCUR_LIMIT`, 20 by default) — reshaped from the raw graph edges by `_map_edge`, which reads the relation text from `properties.predicate` (module 17/20-style edges) or `properties.relation` (Module 25 edges) depending on which store produced it, falling back to the edge's structural `type` only when neither is present.

#### `raptor_tkg_query`

Temporal Knowledge Graph query — time-indexed entity events and relationships. Use for "What happened to X between date A and date B?" questions.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `query` | string | ✓ | — | Natural language query about time-ordered events |
| `time_start` | string | | null | Start date filter (ISO 8601, e.g. `2025-01-01`) |
| `time_end` | string | | null | End date filter (ISO 8601, e.g. `2026-12-31`) |
| `max_depth` | int | | `2` | Subgraph expansion depth (1–4) |
| `limit` | int | | `50` | Maximum nodes to return (1–200) |

Returns time-ordered temporal facts, semantic triples, and a short summary.

---

<a id="tools-memory"></a>

### Memory Tools

All eight proxy Module 26 through Module 13's `/memory/*` gateway routes (see `API_REFERENCE.md`'s [Memory Service](API_REFERENCE.md#memory) section for the underlying REST contract) — each tool below returns that endpoint's response unmodified.

| Tool | Maps to | Parameters |
|---|---|---|
| `raptor_memory_retrieve` | `GET /memory/retrieve` | `query` (✓), `top_k` (default `5`, 1–50) |
| `raptor_memory_store` | `POST /memory/store` | `text` (✓), `frame_type` (`fact` default / `conversation` / `preference` / `entity`), `session_id` (optional, traceability only) |
| `raptor_memory_timeline` | `GET /memory/timeline` | `page` (default `1`), `page_size` (default `20`, 1–100) |
| `raptor_memory_multimedia_search` | `POST /memory/multimedia/search` | `query` (✓), `top_k` (default `5`), `media_type` (`video`/`audio`/`image`, omit for all) |
| `raptor_memory_session_summaries` | `GET /memory/sessions/{id}/summaries` | `session_id` (✓) |
| `raptor_memory_compact` | `POST /memory/compact` | `session_id` (omit for `default`), `trigger` (`auto`/`manual`/`reactive`, default `manual`, logging only), `context_window` (default `128000`), `custom_instructions` (optional, extra guidance for the summarizing LLM) |
| `raptor_memory_compact_evaluate` | `POST /memory/compact/evaluate` | `session_id` (omit to estimate only the messages you pass in), `context_window` (default `128000`), `extra_tokens` (default `0`, size of the current not-yet-archived turn) |
| `raptor_memory_archive` | `POST /memory/archive` | `user_message` (✓), `assistant_response` (✓), `session_id` (omit for `default`) |

`raptor_memory_retrieve` is the one to reach for "search everything I've told you before" — it combines session history, long-term facts/preferences, and multimedia memory in one hybrid (semantic + BM25) query. Use `raptor_memory_multimedia_search` only when you specifically want to exclude text memory and search video/audio/image indexing alone.

```json
{"name": "raptor_memory_store", "arguments": {"text": "User prefers concise answers with bullet points", "frame_type": "preference"}}
```

---

<a id="resources"></a>

## Resources

Read with `resources/read` (`params: {"uri": "raptor://..."}`), or an SDK's `session.read_resource(uri)`.

| URI | MIME type | Description |
|---|---|---|
| `raptor://capabilities` | `text/markdown` | Static Markdown describing Raptor's five RAG pipelines (VideoRAG, DocumentRAG, GraphRAG, TKG, RDBMS) — intended to be injected into an LLM's system context to guide which tool it picks for a given question. |
| `raptor://assets` | `application/json` | JSON list of assets uploaded by **the server's own service account** (not the calling user — see note below), via `GET /asset/users/commits`. |
| `raptor://assets/{asset_path}/{version_id}` | `application/json` | Metadata and presigned download URL for a specific asset version, via `GET /asset/filedownload/{asset_path}/{version_id}`. |

**Resources authenticate differently from tools.** Unlike tool calls, resource reads use a server-level Keycloak token (`MCP_KEYCLOAK_USERNAME`/`MCP_KEYCLOAK_PASSWORD` on Module 27 itself) when configured, falling back to an unauthenticated call otherwise — they do **not** use the calling client's own Bearer token the way every tool above does. In practice this means `raptor://assets` lists the server's own uploads, not the calling user's; treat it as a demo/reference resource rather than a per-user data source. `raptor://capabilities` is static and needs no auth at all.

The `raptor://capabilities` text is worth reading once up front — it's the single clearest explanation of when to reach for GraphRAG vs. TKG vs. plain search that exists anywhere in the docs, written specifically to prime an LLM's tool choice.

---

<a id="prompts"></a>

## Prompts

10 ready-made prompt templates (`prompts/list` to enumerate, `prompts/get` with `{"name": ..., "arguments": {...}}` to render one into a `PromptMessage` list). Each renders to a single user-role message that walks the model through calling the right tool(s) in order — useful as a starting point for a client that wants a sensible default flow without hand-writing one.

| Prompt | Arguments | What it does |
|---|---|---|
| `raptor_search_and_summarise` | `topic` (✓), `top_k` (default `10`) | Search and return a concise, sourced summary |
| `raptor_video_analysis` | `topic` (✓), `top_k` (default `5`) | Find and analyze video clips, listing timestamps and key points |
| `raptor_document_qa` | `question` (✓) | Document-grounded Q&A, answered strictly from retrieved content |
| `raptor_temporal_query` | `entity` (✓), `start_date`, `end_date` | Time-range knowledge-graph query for a date window |
| `raptor_quick_answer` | `question` (✓) | Ask anything, get a plain-language answer via `raptor_chat` |
| `raptor_explore_topic` | `topic` (✓) | Comprehensive overview across all media types, grouped by type |
| `raptor_find_in_video` | `what` (✓) | Find specific moments in videos with jump-to timestamps |
| `raptor_upload_workflow` | `filename` (✓), `content_base64` (✓), `content_type` (✓) | *[Dev]* Upload → poll status → verify searchable, end to end |
| `raptor_search_strategy` | `query` (✓), `media_type` | *[Dev]* Run hybrid/BM25/vector (and video) side by side and compare |
| `raptor_rag_pipeline` | `question` (✓), `mode` (`direct` default / `agent`), `top_k` (default `5`) | *[Dev]* Run `raptor_a2a_direct` or `raptor_a2a_agent` with a structured report-back format |

```json
{"method": "prompts/get", "params": {"name": "raptor_find_in_video", "arguments": {"what": "the trade summit handshake"}}}
```

---

<a id="agent-registration"></a>

## Agent registration (autonomous agents)

**Live-verified end to end** (2026-08-27, against the real deployment, after PR#134/#135/#137/#138 and a Module 13 + Module 27 restart) — register → call a tool with the returned `agent_token` → revoke → call again with the now-revoked `agent_token` and confirm it's rejected. All four steps behaved as documented below. Re-verified 2026-08-28 after Module 06/27 restarts, this time sweeping all 22 tools individually with a registered agent_token (search/asset/graph/processing/memory read+write/LLM-RAG families) — all returned `isError:false` with real, non-empty content; two negative-path tools (`raptor_get_asset_url`, `raptor_check_status`) were also checked against bad input and correctly returned `isError:true`.

### `POST /api/{version}/mcp/auth/register`

A human's Bearer token registers an agent's Keycloak **confidential client** (service account) credentials — not a username/password pair.

**Server-side prerequisite (added 2026-08-27):** Module 27 encrypts the registered `client_secret` at rest in Redis (Fernet). This requires `MCP_SECRET_ENCRYPTION_KEY` to be set in the deployment's **root** `.env` (module-local `.env` is overridden by root per Module 27's `env_file: [.env, ../.env]` merge order) to a real generated key — `python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`. The `.env.example` ships a literal placeholder; leaving it unchanged makes every `register` call fail with `"Fernet key must be 32 url-safe base64-encoded bytes"`.

```bash
curl -X POST http://<host>:8012/api/0.4/mcp/auth/register \
  -H "Authorization: Bearer <human_access_token>" -H "Content-Type: application/json" \
  -d '{"client_id": "<keycloak_confidential_client_id>", "client_secret": "<client_secret>"}'
```

Response:

```json
{"agent_token": "<opaque token, Module 27-issued>", "expires_in": 86400}
```

Use the returned `agent_token` as `Authorization: Bearer <agent_token>` on the protocol endpoint above — Module 27 resolves it to a real Keycloak token and refreshes it automatically via `client_credentials`, so the agent never needs the human's token again.

### `DELETE /api/{version}/mcp/auth/register/{agent_token}`

Revoke a previously-registered agent (also needs a human Bearer token). Returns `204 No Content`.

---

<a id="notes"></a>

## Error handling & notes for client authors

- **Session id is mandatory after `initialize`.** Every call except `initialize` itself needs `Mcp-Session-Id`; omitting it gets rejected.
- **Read past the first SSE event.** `tools/call` on a slow tool (e.g. `raptor_chat`, `raptor_a2a_agent`) emits `notifications/message` progress events before the real `result` — match on `id`, don't stop at the first `data:` line.
- **Timeouts:** the gateway's proxy uses a 150s timeout specifically for this route (`raptor_chat`/`raptor_a2a_agent` can legitimately take up to ~120s) — a client with a shorter timeout will see a connection drop, not a clean error.
- **Check `isError`, don't just parse the content for an `"error"` key.** Before PR#141 every tool swallowed its own exceptions and returned them as a normal-looking `{"error": "..."}` JSON string, so `isError` stayed `false` even on failure — a client had to know to inspect the content for that shape. That's fixed: a failed tool call now comes back with `isError: true` and a plain-text message in `content`, e.g. `{"content":[{"type":"text","text":"Error executing tool raptor_list_assets: Unknown or revoked agent_token: ..."}],"isError":true}` (live-tested against an expired agent_token, 2026-08-27).
- **No Authorization header over HTTP is a hard error, not a silent downgrade.** A tool call with neither a raw JWT nor a registered `agent_token` (prefix `mcp-agent-`) is rejected with `"Missing Authorization: Bearer <token> — MCP tool calls require a caller JWT"` — it never quietly falls back to running as the server's own service account. (Contrast with [stdio transport](#getting-started) and the `raptor://assets*` [resources](#resources), which deliberately do use a server-level identity, for different reasons each.)
