# Raptor 0.4 A2A Reference

**Short answer: yes — Agent-to-Agent (A2A) is available and live in Aigle 0.4**, and it runs deeper than the one REST endpoint documented in [`API_REFERENCE.md`](API_REFERENCE.md#rag). This document is the interface definition and user guide for the rest of it: how an external agent discovers Raptor, sends it a task, and gets served — the A2A equivalent of `API_REFERENCE.md`.

Two layers make up "A2A" in this release:

1. **The orchestrator** (Module 21, `agent-protocol`) — a custom JSON-RPC 2.0 dialect (`agent.discover` / `agent.query` / `agent.delegate`), standard A2A message forwarding, agent-card discovery, and a Redis-backed peer-agent registry. Reached through the gateway at `/api/0.4/a2a/*`, or directly on port `8030`.
2. **Five specialist sub-agents** — `vector_search_agent`, `keyword_search_agent`, `graphrag_agent`, `tkg_agent`, `reranker_agent` — each a genuine, independent, spec-compliant A2A server built on the official [`a2a-sdk`](https://pypi.org/project/a2a-sdk/) (`a2a-sdk[http-server]==0.3.26`), each on its own port (52001–52005), each with its own real `AgentCard`. These are what the orchestrator itself calls internally to answer a query — an external A2A client can call them exactly the same way.

**Most of this is not in the gateway's Swagger docs.** Every route below except `/api/0.4/a2a/query` is registered with `include_in_schema=False` on Module 13 — it works, but nothing auto-generates docs for it. This file is the only place it's written down.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Agent Discovery — three coexisting conventions](#discovery)
- [The Orchestrator's A2A Surface (Module 21, via `/api/{version}/a2a/*`)](#orchestrator)
  - [JSON-RPC 2.0: `agent.discover` / `agent.query` / `agent.delegate`](#jsonrpc)
  - [Standard A2A message forwarding](#message-forward)
  - [Peer agent registration & discovery](#peer-registry)
  - [Plain REST query (`/query`)](#rest-query)
- [The Specialist Sub-Agents](#sub-agents)
- [Known limitations — read before you build on this](#limitations)
- [Client examples](#examples)

---

<a id="getting-started"></a>

## Getting Started

**If you just want Raptor to answer a question and don't care about the A2A protocol machinery**, use the REST endpoint already documented in `API_REFERENCE.md`: [`POST /api/0.4/a2a/query`](API_REFERENCE.md#rag). Everything below exists for the case where your caller *is itself an agent* speaking A2A — discovering Raptor's capabilities via an agent card, sending a `message/send`, registering itself as a peer, and so on.

**If your caller is a generic, spec-compliant A2A client** (built on `a2a-sdk` or an equivalent), point it at one of the five specialist sub-agents (see [below](#sub-agents)) rather than the orchestrator — they follow the standard SDK conventions exactly, whereas the orchestrator's own `/.well-known/*` paths use two different, non-standard shapes (see [Agent Discovery](#discovery)).

**If you're building a custom integration and are fine hand-rolling JSON-RPC**, the orchestrator's `agent.query` method (via `/api/0.4/a2a/jsonrpc`) is the simplest single call that runs Raptor's full RAG pipeline and hands back a grounded answer — see [JSON-RPC](#jsonrpc) below.

All endpoints in this document require the same Bearer JWT as everything in `API_REFERENCE.md` (`POST /api/0.4/sso/login`), except the specialist sub-agents when reached directly on their own ports (52001–52005) — those have no auth layer of their own; they're meant to sit behind the gateway/orchestrator on the internal `raptor` network. If you expose their published host ports externally, put your own auth in front.

---

<a id="discovery"></a>

## Agent Discovery — three coexisting conventions

Three different `/.well-known/*` shapes exist in this release — know which one you're hitting:

| Path | Served by | Shape |
|---|---|---|
| `/.well-known/agent-card` (no extension) | Gateway (Module 13) and Module 21, each independently | Raptor's own legacy shape: `{id, name, version, description, endpoint, capabilities: [...strings], protocols, authentication}` |
| `/.well-known/agent.json` | Gateway and Module 21, each independently | A closer-to-spec shape: `{name, description, version, url, defaultInputModes, defaultOutputModes, capabilities: {}, skills: [...]}` |
| `/.well-known/agent-card.json` | **Each of the 5 specialist sub-agents** (this is the `a2a-sdk`'s own default path, not something Raptor built) | Full `a2a-sdk` `AgentCard` object |

**The gotcha:** a generic A2A client library doing auto-discovery expects `/.well-known/agent-card.json` — that's the one path the orchestrator and gateway do *not* serve. Pointed at the orchestrator, such a client's auto-discovery will 404. Pointed at any of the five specialist sub-agents, it works exactly as the spec expects. If you need the orchestrator/gateway's capabilities via a generic client, fetch `/.well-known/agent.json` manually rather than relying on SDK auto-discovery.

```bash
# Gateway-level card (describes the whole platform)
curl http://<host>:8012/.well-known/agent.json

# Module 21 orchestrator's own card (same shape, reachable directly too)
curl http://<host>:8030/.well-known/agent.json

# A specialist sub-agent's spec-standard card
curl http://<host>:52001/.well-known/agent-card.json
```

---

<a id="orchestrator"></a>

## The Orchestrator's A2A Surface (Module 21, via `/api/{version}/a2a/*`)

Gateway base URL: `http://<host>:8012/api/0.4/a2a` (also reachable at `/api/0.3/a2a/*` via the legacy-alias middleware, which rewrites to the canonical path and marks the response deprecated). Direct base URL, bypassing the gateway: `http://<host>:8030`.

<a id="jsonrpc"></a>

### JSON-RPC 2.0: `agent.discover` / `agent.query` / `agent.delegate`

`POST /api/0.4/a2a/jsonrpc`

Standard JSON-RPC 2.0 envelope (`{"jsonrpc": "2.0", "id": ..., "method": ..., "params": {...}}`), three supported methods:

| Method | Params | What it does |
|---|---|---|
| `agent.discover` | `{}` | Returns the orchestrator's own agent card (same shape as `/.well-known/agent-card`) |
| `agent.query` | `{"query": "...", "top_k": 5}` | Runs the **full RAG pipeline** (same as `/query` mode=direct) and returns an `OrchestratorResult` |
| `agent.delegate` | `{"task": "...", "payload": {...}, "target_agent_id": null}` | **Stub — see [Known limitations](#limitations).** Accepts and echoes the request; does not actually route or execute anything yet. |

```bash
curl -X POST http://<host>:8012/api/0.4/a2a/jsonrpc \
  -H "Authorization: Bearer <token>" -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"agent.query","params":{"query":"How does the cooling system work?","top_k":5}}'
```

```json
{"jsonrpc":"2.0","id":1,"result":{"answer":"...","sources":[...],"graph_context":"","chunks_used":3}}
```

`agent.query` accepts an optional `session_id` in `params` to archive the turn via Module 26; if omitted, it falls back to the JSON-RPC request `id` as the session identity — so repeated calls with the same numeric/string `id` share memory, and calls with a fresh `id` (or an explicit `session_id`) don't.

Errors follow standard JSON-RPC 2.0 codes: `-32601` method not found, `-32603` internal error (with the exception message in `error.data`).

<a id="message-forward"></a>

### Standard A2A message forwarding

`POST /api/0.4/a2a/message?target=<agent_name>` and `POST /api/0.4/a2a/agents/{agent_name}/message` both forward a standard A2A `message/send` envelope to one of the specialist sub-agents (`vector`, `keyword`, `graphrag`, `tkg`, `reranker`), by proxying the raw request body to that agent's own root endpoint.

```json
{
  "jsonrpc": "2.0",
  "id": "req-1",
  "method": "message/send",
  "params": {
    "message": {
      "role": "user",
      "messageId": "msg-001",
      "parts": [{"type": "text", "text": "{\"query\": \"cooling system\", \"top_k\": 3, \"type\": \"documents\"}"}]
    }
  }
}
```

The `text` part is itself a JSON string matching that sub-agent's own request model (see [The Specialist Sub-Agents](#sub-agents) below) — this is the A2A convention this codebase uses throughout: a plain-text message part carrying a JSON payload, packed/unpacked with `model_dump_json()`/`model_validate_json()` rather than structured message parts.

**`target` must name one of the five specialist agents.** The endpoint defaults to `"orchestrator"` when no `target` query param or `X-A2A-Target-Agent` header is given, but — see [Known limitations](#limitations) — that default doesn't currently resolve to a working handler. Always pass an explicit target.

<a id="peer-registry"></a>

### Peer agent registration & discovery

A Redis-backed (`a2a:agents:*`, TTL `AGENT_TTL`, default 300s) registry any external agent can add itself to, making it visible to anything that lists registered agents.

| Endpoint | Method | Description |
|---|---|---|
| `/api/0.4/a2a/agents` | `GET` | List all currently-registered agents (self + any registered peers) |
| `/api/0.4/a2a/agents/register` | `POST` | Register an external peer agent's own card: `{id, name, version, description, endpoint, capabilities: [...], protocols: ["a2a/1.0"]}` |
| `/api/0.4/a2a/agents/{agent_id}` | `DELETE` | Deregister a peer immediately |
| `/api/0.4/a2a/agents/{agent_id}/heartbeat` | `POST` | Renew a peer's TTL — call at roughly `AGENT_TTL × 0.8` seconds |

```bash
curl -X POST http://<host>:8012/api/0.4/a2a/agents/register \
  -H "Authorization: Bearer <token>" -H "Content-Type: application/json" \
  -d '{"id": "my-external-agent", "name": "My Agent", "endpoint": "http://my-agent.example.com:9000", "capabilities": ["custom_skill"]}'
```

A registered entry disappears automatically if its owner stops heartbeating — there is no explicit "offline" push, only TTL expiry.

<a id="rest-query"></a>

### Plain REST query (`/query`)

Already documented in full in `API_REFERENCE.md` as [`POST /api/0.4/a2a/query`](API_REFERENCE.md#rag) — the same `direct`/`agent`/`tool` modes, the same request/response shape. It's the non-protocol equivalent of `agent.query` above; use it when your caller just wants an HTTP+JSON answer and has no reason to speak A2A/JSON-RPC at all.

---

<a id="sub-agents"></a>

## The Specialist Sub-Agents

Five independent `a2a-sdk` servers. Each is directly callable — by the orchestrator (internally, for `mode=direct`/`agent` RAG), by another Raptor sub-agent, or by any standard A2A client, using the standard `message/send` method with a JSON-string text part matching the request model below. Every result comes back the same way: JSON text in the response message, matching the listed result model.

| Agent | Port | Card `name` | Backs onto |
|---|---|---|---|
| Vector Search | `52001` (`PORT_VECTOR_AGENT`) | `vector_search_agent` | Module 25 `/personal/search/vector` |
| Keyword Search | `52002` (`PORT_KEYWORD_AGENT`) | `keyword_search_agent` | Module 25 `/personal/search/bm25` |
| GraphRAG | `52003` (`PORT_GRAPHRAG_AGENT`) | `graphrag_agent` | Module 25 `/personal/search/graphrag` |
| Temporal KG | `52004` (`PORT_TKG_AGENT`) | `tkg_agent` | Module 25 `/personal/search/tkg` |
| Reranker | `52005` (`PORT_RERANKER_AGENT`) | `reranker_agent` | Module 25 rerank endpoint, falling back to a local cross-encoder |

All five auto-provision the caller's Module 25 database on first use (`X-Branch-ID`-scoped, via `POST /internal/db/init`) — a provisioning failure doesn't block the call, it just surfaces as whatever the underlying search then returns.

### Vector Search Agent (`vector_search_agent`, port 52001)

Semantic vector search. Skill id: `vector_search`.

| Field | Type | Default | Description |
|---|---|---|---|
| `query` | string | required | Search text |
| `top_k` | int | `10` | Max results |
| `type` | string \| string[] | null | `videos`/`audios`/`documents`/`images` |
| `score_threshold` | float | `0.3` | Minimum score to include a hit |
| `branch_id` | string | null | Tenant/user scope (Module 25 database selector) |

```json
{"query": "cooling system", "top_k": 5, "type": "documents"}
```

### Keyword Search Agent (`keyword_search_agent`, port 52002)

BM25 full-text search. Skill id: `keyword_search`. Same fields as Vector Search minus `score_threshold`.

```json
{"query": "error code 404", "top_k": 5, "type": "documents"}
```

### GraphRAG Agent (`graphrag_agent`, port 52003)

Entity/relationship retrieval. Skill id: `graphrag`.

| Field | Type | Default | Description |
|---|---|---|---|
| `query` | string | required | Natural-language query |
| `entity` | string | null | Seed entity name to focus traversal on |
| `max_depth` | int | `2` | Subgraph expansion depth |
| `score_threshold` | float | `0.3` | Minimum score to include an edge |
| `branch_id` | string | null | Tenant/user scope |

```json
{"query": "how does A relate to B", "max_depth": 2}
```

Result shape (`GraphResult`): `{triples: [{subject, predicate, object, valid_from?, valid_to?}], summary, entity_count, source}`.

### Temporal KG Agent (`tkg_agent`, port 52004)

Same as GraphRAG plus a time window. Skill id: `tkg`.

| Field | Type | Default | Description |
|---|---|---|---|
| `query` | string | required | Natural-language query about time-ordered events |
| `max_depth` | int | `2` | Subgraph expansion depth |
| `score_threshold` | float | `0.3` | Minimum score |
| `time_start` | string | null | ISO 8601 |
| `time_end` | string | null | ISO 8601 |
| `branch_id` | string | null | Tenant/user scope |

```json
{"query": "events involving entity X", "time_start": "2024-01-01", "max_depth": 2}
```

### Reranker Agent (`reranker_agent`, port 52005)

Re-scores a caller-supplied candidate list — this one doesn't search anything itself, it just reorders results you already have (e.g. from combining the four search agents above). Skill id: `rerank`.

| Field | Type | Default | Description |
|---|---|---|---|
| `query` | string | required | The original query the candidates were retrieved for |
| `candidates` | array of objects | required | The hits to rerank (any dict shape carrying at least enough for the cross-encoder to score against `query`) |
| `top_k` | int | `5` | How many reranked candidates to return |

```json
{"query": "cooling system", "candidates": [{"...": "..."}], "top_k": 5}
```

Common result shape for the four search agents (`SearchResult`): `{hits: [{id, doc_id, score, content, metadata, start_time, end_time, asset_path, version_id, storage_uri}], total, source}`.

---

<a id="limitations"></a>

## Known limitations — read before you build on this

- **`agent.delegate` is a stub.** It returns `{"status": "accepted", "task": ..., "target_agent_id": ..., "timestamp": ...}` and does nothing else — no routing, no execution, whatever you "delegate" simply isn't run. Use `agent.query` (which does run the real pipeline) or call a specialist sub-agent directly instead.
- **Peer registration doesn't feed delegation.** Registering an external agent via `/a2a/agents/register` makes it listable via `/a2a/agents` — nothing in the current pipeline automatically routes tasks to a registered peer. Today it's a discovery mechanism only.
- **`/a2a/message`'s default target ("orchestrator") isn't wired to a working endpoint.** The five specialist sub-agents each expose a root `message/send` handler (that's what the standard `a2a-sdk` app gives you), but the orchestrator itself has no equivalent root handler — it has `/a2a/jsonrpc` for its own dialect instead. Always pass an explicit `target=<vector|keyword|graphrag|tkg|reranker>` (or the `X-A2A-Target-Agent` header) when using this endpoint.
- **Three incompatible `/.well-known/*` conventions coexist** — see [Agent Discovery](#discovery). A generic client's auto-discovery only works against the five specialist agents, not the orchestrator or gateway.
- **Most of this surface is hidden from the gateway's Swagger UI** (`include_in_schema=False` on every route except `/query`) — this document, not `/docs`, is the source of truth for it.
- **The specialist sub-agents have no auth of their own.** They're designed to live behind the internal `raptor` Docker network; their ports are published to the host by default (52001–52005) for convenience during development — put your own access control in front before exposing them past a single trusted host.

---

<a id="examples"></a>

## Client examples

### Python — calling a specialist sub-agent directly with the official `a2a-sdk`

Adapted directly from Raptor's own internal caller (`deployment/modules/21-agent-protocol/app/pipeline.py::_call_agent`), so this is exactly the pattern the orchestrator itself uses — not a hypothetical.

```python
# pip install "a2a-sdk[http-server]==0.3.26" httpx
import asyncio, json, uuid
import httpx
from a2a.client import A2AClient
from a2a.types import AgentCard, SendMessageRequest, MessageSendParams
from a2a.client.helpers import create_text_message_object

AGENT_URL = "http://<host>:52001"  # vector_search_agent

async def call_vector_agent(query: str, top_k: int = 5) -> dict:
    async with httpx.AsyncClient(timeout=30.0) as client:
        card_resp = await client.get(f"{AGENT_URL}/.well-known/agent-card.json")
        card = AgentCard(**card_resp.json())

        a2a = A2AClient(httpx_client=client, agent_card=card)
        payload = json.dumps({"query": query, "top_k": top_k})
        req = SendMessageRequest(
            id=str(uuid.uuid4()),
            params=MessageSendParams(message=create_text_message_object(content=payload)),
        )
        resp = await a2a.send_message(req)

        # resp.root.result is either a Message (text reply) or a Task (check .history)
        result = resp.root.result
        text = getattr(result, "parts", None)  # Message case: iterate .parts for TextPart
        return json.loads(text[0].root.text) if text else {}

print(asyncio.run(call_vector_agent("cooling system")))
```

### curl — raw JSON-RPC to the orchestrator

```bash
TOKEN=$(curl -s -X POST http://<host>:8012/api/0.4/sso/login -d "username=<user>&password=<pass>" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['access_token'])")

curl -X POST http://<host>:8012/api/0.4/a2a/jsonrpc \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"agent.query","params":{"query":"Summarize the uploaded report","top_k":5}}'
```

### curl — message/send straight to a specialist sub-agent (bypassing the orchestrator)

```bash
curl -X POST http://<host>:52002/ -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":"req-1","method":"message/send","params":{
        "message":{"role":"user","messageId":"msg-001",
                   "parts":[{"type":"text","text":"{\"query\": \"error code 404\", \"top_k\": 3}"}]}}}'
```
