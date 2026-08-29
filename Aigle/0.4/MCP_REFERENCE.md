# Raptor 0.4 MCP Reference

How to talk to Raptor over the [Model Context Protocol](https://modelcontextprotocol.io/) — a different shape of API from [`API_REFERENCE.md`](API_REFERENCE.md)'s plain REST endpoints, documented separately for that reason.

Module 27 (`27-mcp-server`) is the actual MCP server — it exposes Raptor's search/chat/upload/graph capabilities as MCP tools. Module 13's gateway proxies it at `/api/{version}/mcp` so an MCP client never needs to reach module 27 directly.

---

## Two ways to authenticate

**1. Human — Bearer token, same as every other endpoint in `API_REFERENCE.md`.** Works for everything below, including the main protocol endpoint. Fine for testing or a client that already manages its own token lifecycle.

**2. Autonomous agent — register once, then reuse an `agent_token`.** For an agent that runs unattended and can't re-authenticate interactively. A human registers the agent's Keycloak service-account credentials once; the MCP server then refreshes the underlying token itself via `client_credentials`, so the agent never has to log in again. See [Agent registration](#agent-registration) below.

---

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

---

## Tool catalog

22 tools, live-queried via `tools/list` against the real deployment (2026-08-28, after today's Module 27 restart). Every tool's actual `inputSchema` is authoritative — this table is a quick-reference, not a substitute for calling `tools/list` yourself.

`raptor_trigger_processing` was removed (its target route, `POST /processing/process-file`, doesn't exist on the gateway — `raptor_upload_asset` already triggers processing on upload) and the 7 `raptor_memory_*` tools below `raptor_memory_retrieve` were added, covering the gateway's full non-destructive Module 26 memory surface.

| Tool | Required params | Optional params | What it does |
|---|---|---|---|
| `raptor_search` | `query` | `top_k`, `type`, `speaker`, `source`, `embedding_type` | Hybrid search across all asset types |
| `raptor_search_bm25` | `query` | `top_k`, `type`, `speaker`, `source` | Keyword-only search |
| `raptor_search_vector` | `query` | `top_k`, `type`, `speaker`, `source`, `embedding_type` | Semantic-only search |
| `raptor_video_search` | `query` | `top_k`, `asset_path`, `candidate_multiplier`, `score_threshold` | Video-specific multi-retriever search |
| `raptor_chat` | `message` | `session_id`, `history` | Conversational RAG with memory |
| `raptor_list_assets` | — | `keyword`, `start_date`, `end_date`, `page`, `page_size` | List the caller's uploaded assets |
| `raptor_get_asset_url` | `asset_path`, `version_id` | — | Presigned 24h download URL |
| `raptor_upload_asset` | `filename`, `content_base64` | `content_type`, `archive_ttl_days`, `destroy_ttl_days` | Upload + trigger AI processing |
| `raptor_check_status` | `correlation_id` | `m_type` | Poll async processing status |
| `raptor_graph_query` | `query` | `entity`, `max_depth`, `limit`, `strategy` | GraphRAG entity/relationship query |
| `raptor_tkg_query` | `query` | `time_start`, `time_end`, `max_depth`, `limit` | Temporal knowledge graph query |
| `raptor_a2a_direct` | `question` | `top_k`, `session_id` | Deterministic RAG pipeline (classify → search → rerank → answer) |
| `raptor_a2a_agent` | `question` | `top_k`, `session_id` | Agentic RAG (LLM decides which tools to call) |
| `raptor_query_orchestrate` | `query` | `top_k` | Auto-routed query through the full RAG pipeline |
| `raptor_memory_retrieve` | `query` | `top_k` | Search the caller's Module 26 memory: session history + long-term facts + multimedia, combined |
| `raptor_memory_store` | `text` | `frame_type` (`fact` default / `conversation` / `preference` / `entity`), `session_id` | Write a standalone memory node not tied to a session |
| `raptor_memory_timeline` | — | `page`, `page_size` | Paginated, time-ascending timeline of the caller's turns across all sessions |
| `raptor_memory_multimedia_search` | `query` | `top_k`, `media_type` (`video`/`audio`/`image`) | Search only the caller's indexed video/audio/image memory |
| `raptor_memory_session_summaries` | `session_id` | — | List the summary frames produced by compacting a session |
| `raptor_memory_compact` | — | `session_id`, `trigger`, `context_window`, `custom_instructions` | Summarise a session's older turns to free up context window space |
| `raptor_memory_compact_evaluate` | — | `session_id`, `context_window`, `extra_tokens` | Estimate the token budget a compaction would need, without running it |
| `raptor_memory_archive` | `user_message`, `assistant_response` | `session_id` | Archive one conversation turn into a session |

**Known stale wording:** `raptor_search_bm25`/`raptor_search_vector`'s own descriptions still say "using OpenSearch"/"using Qdrant" — those backends were retired (see root [`README.md`](README.md#deprecated-modules)); the real backend for all `raptor_search*` tools is Module 25 (ArcadeDB). The descriptions themselves haven't been updated to match — trust this doc and `API_REFERENCE.md`'s [Search](API_REFERENCE.md#search) section over the tool's own text until that's fixed.

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

## Notes for client authors

- **Session id is mandatory after `initialize`.** Every call except `initialize` itself needs `Mcp-Session-Id`; omitting it gets rejected.
- **Read past the first SSE event.** `tools/call` on a slow tool (e.g. `raptor_chat`, `raptor_a2a_agent`) emits `notifications/message` progress events before the real `result` — match on `id`, don't stop at the first `data:` line.
- **Timeouts:** the gateway's proxy uses a 150s timeout specifically for this route (`raptor_chat`/`raptor_a2a_agent` can legitimately take up to ~120s) — a client with a shorter timeout will see a connection drop, not a clean error.
- **Check `isError`, don't just parse the content for an `"error"` key.** Before PR#141 every tool swallowed its own exceptions and returned them as a normal-looking `{"error": "..."}` JSON string, so `isError` stayed `false` even on failure — a client had to know to inspect the content for that shape. That's fixed: a failed tool call now comes back with `isError: true` and a plain-text message in `content`, e.g. `{"content":[{"type":"text","text":"Error executing tool raptor_list_assets: Unknown or revoked agent_token: ..."}],"isError":true}` (live-tested against an expired agent_token, 2026-08-27).
