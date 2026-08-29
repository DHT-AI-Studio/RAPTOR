# Module 21 — agent-protocol

Agent-to-Agent (A2A) orchestrator: JSON-RPC discovery/query/delegate, standard A2A `message/send` forwarding, and a Redis-backed peer-agent registry, fronting five independent, spec-compliant A2A specialist agents (vector / keyword / GraphRAG / TKG / reranker, all searching Module 25) plus a smolagents-based `agent`/`tool` RAG mode. See [`../../../A2A_REFERENCE.md`](../../../A2A_REFERENCE.md) for the full interface definition and client guide — this README is a quick-start/orientation layer on top of it, not a replacement for it.

**Key dependencies:** 02 (Redis, peer registry), 04 (presigned asset URLs), 07 (LLM answer generation), 13 (gateway proxy; also how this module reaches Module 26), 18 (intent classification), 25 (searched by the 5 built-in sub-agents below), 26 (turn archiving, via 13)

## Quick start

```bash
cd deployment/modules/21-agent-protocol
cp .env.example .env
docker compose up -d
curl http://localhost:8030/health
curl -X POST http://localhost:8030/query -H "Content-Type: application/json" -d '{"question": "How does the cooling system work?"}'
```

This module runs **six containers**, not one:

| Service | Port | Role |
| --- | --- | --- |
| `agent-protocol` (orchestrator) | `PORT_AGENT_PROTOCOL` (default `8030`) | REST `/query`, A2A JSON-RPC, peer registry, agent-card discovery |
| `vector-agent` | `PORT_VECTOR_AGENT` (default `52001`) | Specialist A2A agent — semantic search |
| `keyword-agent` | `PORT_KEYWORD_AGENT` (default `52002`) | Specialist A2A agent — BM25 search |
| `graphrag-agent` | `PORT_GRAPHRAG_AGENT` (default `52003`) | Specialist A2A agent — entity/relationship search |
| `tkg-agent` | `PORT_TKG_AGENT` (default `52004`) | Specialist A2A agent — temporal knowledge graph |
| `reranker-agent` | `PORT_RERANKER_AGENT` (default `52005`) | Specialist A2A agent — cross-encoder rerank |

The orchestrator's `depends_on` waits for all five sub-agents to report `service_healthy` before it starts.

## What this module actually does

`POST /query` (mode `direct` / `agent` / `tool`) runs the full RAG pipeline: intent classification (Module 18) → fan out to the five sub-agents above → RRF fusion → rerank → LLM answer (Module 07). Reached externally via the API Gateway's `POST /api/0.4/a2a/query` ([`API_REFERENCE.md`](../../../API_REFERENCE.md#rag)).

For the Agent-to-Agent protocol itself — agent cards, JSON-RPC `agent.discover` / `agent.query` / `agent.delegate`, standard `message/send` to a specific sub-agent, and the peer-agent registry — see [`A2A_REFERENCE.md`](../../../A2A_REFERENCE.md), which documents this module in full, including its known limitations (`agent.delegate` is currently a stub; the registry doesn't yet feed automatic delegation).

## Endpoint groups (orchestrator — the `agent-protocol` service)

| Prefix | Purpose |
| --- | --- |
| `/query` | Plain REST RAG query, `mode=direct\|agent\|tool` |
| `/a2a/jsonrpc` | JSON-RPC 2.0: `agent.discover` / `agent.query` / `agent.delegate` |
| `/a2a/message`, `/agents/{name}/message` | Standard A2A `message/send`, forwarded to a named specialist sub-agent |
| `/a2a/agents`, `/a2a/agents/register`, `/a2a/agents/{id}`, `/a2a/agents/{id}/heartbeat` | Redis-backed peer-agent registry (list / register / deregister / heartbeat) |
| `/agents`, `/agents/cards`, `/agents/{name}/card` | List/inspect the 5 built-in sub-agents and their AgentCards |
| `/.well-known/agent-card`, `/.well-known/agent.json` | This module's own agent card (two custom shapes — neither is the `a2a-sdk`'s own `/.well-known/agent-card.json` default, which the 5 sub-agents use instead) |
| `/health` | Liveness of all 5 sub-agents plus downstream services |

Each sub-agent (`agents/*_agent.py`) exposes only the standard `a2a-sdk` surface — an `AgentCard` at `/.well-known/agent-card.json` and a `message/send` handler at its root — there's no REST surface of their own to list here.

Full request/response schemas, worked client examples, and known limitations: [`A2A_REFERENCE.md`](../../../A2A_REFERENCE.md).
