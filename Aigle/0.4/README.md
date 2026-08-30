# Raptor 0.4

![CI](https://github.com/DHT-AI-Studio/Raptor_0.4/actions/workflows/ci.yml/badge.svg)

Multimodal AI framework and agent harness for building agentic applications over video, audio, documents, and images. Provides MCP tools, A2A orchestration, persistent memory, hybrid BM25/vector RAG, GraphRAG, pipeline evaluation, LLM guardrails, and model lifecycle management — all as independently deployable Docker microservices, built entirely in Python.

API Reference: [`API_REFERENCE.md`](API_REFERENCE.md) · MCP Reference: [`MCP_REFERENCE.md`](MCP_REFERENCE.md) · Setup, Build & Configuration: [`BUILD.md`](BUILD.md) · Demo Frontend: [`raptor-demo-frontend/`](raptor-demo-frontend/)

---

## Architecture

27 modules under `deployment/modules/`, numbered `01`–`27` in dependency order. Each module is an independent Docker Compose stack sharing the `raptor` bridge network. Three of the 27 (`17`, `19`, `20`) are **deprecated** — kept in the tree for rollback but not part of the live pipeline; see [Deprecated Modules](#deprecated-modules).

### Media Processing Flow

```
File Upload (API Gateway :8012)
  → Asset Management (LakeFS + SeaweedFS, Module 04)
  → Kafka topic
  → Processing workers (audio / video / image / document, Modules 09–12)
    → Transcription (WhisperX) + OCR + Frame description (InternVL) + Summary
    → optional Guardrail content-moderation check (Module 23, called from Module 07's inference gateway)
  → Per-user indexing (Module 25, backed by ArcadeDB/Module 24): hybrid BM25 + vector search, knowledge graph, temporal facts
  → Accessible via Search / RAG / A2A APIs (Modules 13, 15, 21)
```

Modules 17 (OpenSearch/Qdrant-based hybrid search) and 20 (Neo4j-based graph reasoning, on top of Module 19's database) were the original backing stores for this flow; both were retired in favor of Module 25's per-user ArcadeDB index (see [Deprecated Modules](#deprecated-modules)).

---

## Modules

### CPU-only

| ID | Module             | Description                                                              |
| -- | ------------------ | ------------------------------------------------------------------------ |
| 01 | nfs-server         | NFS Server — shared storage backing SeaweedFS volumes                   |
| 02 | redis-cluster      | Redis standalone (default, used platform-wide) + optional 6-node cluster via `COMPOSE_PROFILES=cluster` |
| 03 | database           | PostgreSQL (shared, used by all services except Keycloak); also runs a Qdrant container, unused since Module 17's retirement (kept for rollback) |
| 04 | object-storage     | SeaweedFS distributed object storage + LakeFS + Asset Management API     |
| 05 | kafka-cluster      | Kafka KRaft cluster (3 controllers + 3 brokers) + AKHQ UI                |
| 06 | authentication     | Keycloak identity provider with embedded PostgreSQL                      |
| 13 | api-services       | API Gateway (`:8012`) + Asset Management + Search + A2A proxy APIs      |
| 14 | monitoring         | Prometheus + Grafana + Alertmanager + Node Exporter + Loki               |
| 15 | chat-service       | Chat service — LangGraph RAG pipeline, search via Module 25, Redis memory |
| 21 | agent-protocol     | Agent Protocol — A2A discovery, orchestration and RAG pipeline (search/graph agents proxy to Module 25) (`:8030`) |
| 22 | benchmark-service  | Benchmark Service — user-defined marking schemas, pluggable scoring (incl. LLM-as-judge + pairwise), run history, and a `local_infer` pipeline that scores fine-tuned checkpoints via Module 16 (train → serve → score loop) (`:8022`). `local_infer` and `/optimize` (AutoTune) need Module 16, not deployed here; every other pipeline is live-verified — see [API_REFERENCE.md](API_REFERENCE.md#benchmark). |
| 23 | guardrail-service  | Guardrail Service — content moderation (Llama Guard 3 / Granite Guardian / GPT-OSS-Safeguard), policy engine, audit logging; shares Module 02/03's Redis/Postgres (`:8023`, disabled by default — see Module 07/13/23's `.env.example` for the enable switches). In practice, Module 13/07's actual checks go through the policy-free `/guard/check/*` guard-model group (each model's own built-in safety judgment) — no policy has ever been configured/activated, so the policy engine isn't driving any real decision yet. |
| 24 | arcadedb-server    | ArcadeDB — multi-model engine (graph + vector + BM25) backing Module 25 (`:2480`) |
| 25 | personal-db-service | Personal DB — per-user isolated hybrid/graph/temporal search + lifecycle API; the platform's only content index (`:8025`) |
| 26 | memory-service     | Memory Service — MemVID session / long-term memory, reached through Module 13's gateway (`:8026`) |
| 27 | mcp-server         | MCP Server — exposes Raptor APIs as MCP tools/resources (`:8027`) |

### GPU required

| ID | Module              | Description                                                                   |
| -- | ------------------- | ----------------------------------------------------------------------------- |
| 07 | ai-ml-services      | MLflow tracking server + AI Lifecycle API (also hosts the Module 23 guardrail hook on `/inference/infer`) |
| 08 | media-worker        | Shared GPU base image (torch 2.7.1 cu128 sm_120/Blackwell + PaddlePaddle 3.3.0 + WhisperX + docling) |
| 09 | audio-processing    | WhisperX STT / diarization / PANNs / audio summary workers                    |
| 10 | image-processing    | InternVL image analysis + hybrid search workers                               |
| 11 | video-processing    | Video chunking / frame description / OCR / summary workers                    |
| 12 | document-processing | PDF/Office document analysis + LLM summary workers                            |
| 16 | training-service    | **Optional** — GPU training orchestration via FastAPI + local model inference (`/api/v1/inference/infer`) for serving fine-tuned checkpoints. Only needed for Module 22's `local_infer` pipeline and `/optimize` (AutoTune); not part of a standard deployment and commonly excluded (`deploy.sh -e 16,...`) — everything else in the platform works without it. |
| 18 | query-orchestrator  | Query Orchestrator — intent classification (FAISS+BM25), signal extraction, multi-backend RAG routing |

### Deprecated Modules

Retired from the live pipeline (see [Media Processing Flow](#media-processing-flow)) but kept in the tree for rollback — **do not start** unless specifically reviving one of them.

| ID | Module          | Description                                                              | Replaced by |
| -- | --------------- | ------------------------------------------------------------------------- | ----------- |
| 17 | hybrid-search   | OpenSearch cluster (2 nodes) + Dashboards + Hybrid Search API             | Module 25 (per-user ArcadeDB hybrid search) |
| 19 | graph-database  | Neo4j graph database                                                      | Module 24/25 (ArcadeDB graph store) |
| 20 | graph-service   | Graph Service — LLM-powered knowledge graph query and reasoning           | Module 25's `/personal/search/graphrag` |

Other modules' `build.py`/`.env.example` entries may still reference 17/19/20 (e.g. Module 18's dependency list, Module 15's `HYBRID_SEARCH_URL`) — those are leftover wiring from before the retirement, not evidence the modules are still called at runtime; the actual code paths (Modules 04, 09–12, 15, 21) all call Module 25 directly, with the old calls commented out in place rather than deleted.

### In Development

Not part of `deploy.sh`/`build.py` yet — no `Dockerfile`, `docker-compose.yml`, or `build.py` entry exists for these, so they **cannot be started** the way modules 01–27 can.

| ID | Module | Status |
| -- | ------ | ------ |
| 29 | doc-processing-agent (DocAgent) | Scaffolding tickets DA-1–DA-5 (containerization, format readers, VLM extraction) are still planned, not built. Only DA-6 (ad-hoc document vector search over its own Qdrant collection, called via smolagents tool functions rather than a FastAPI router) has actual code, under `app/`. See `deployment/modules/29-doc-processing-agent/docs/` for the implementation plan and `Module_Overlap_Analysis.md`'s assessment of DA-6 vs Module 17 and planned DA-2/DA-5 vs Module 12 — the analysis recommends DocAgent call Module 12's `/analyze` for extraction and Module 17's reranker for ranking rather than rebuilding either. |

---

## Testing Status

Not tracked as a single point-in-time table here — the module set and code have changed too much since the last full pass for a static table to stay honest (it previously listed a status per module as of 2026-05-18, before Modules 23/24/25/26/27 existed and before the Module 17/19/20 retirement). Rely on the CI badge above and each module's own tests; PR descriptions on `main` record what was live-tested and when for that specific change.

---

## Deployment

`deploy.sh` at the repo root is a thin wrapper around `deployment/modules/build.py` — every flag is identical.

**Before the first run**:

1. **Create the `.env` files** and fill in required values (`HOST_IP`, `NFS_*`, `OLLAMA_HOST`, credentials, `HF_TOKEN`, …) — see [`BUILD.md` §3](BUILD.md#3-configuration-env) for the exact commands and for which values need editing in the root `.env` vs. a module's own local `.env`. **Then verify it**: `bash deploy.sh --check-env` catches values still left as an unfilled `<placeholder>` (a genuinely missing required value fails loudly on its own, but a placeholder-shaped one just starts the service with a broken credential and fails somewhere unrelated), and `bash deploy.sh --check-ports` catches host-port conflicts ahead of time.
2. **Build everything first** — recommended, especially for the first run on a new host:
   ```bash
   bash deploy.sh --build-only   # build every module's image, start nothing
   ```
   This confirms every image builds successfully — including module 08's shared GPU base image (used by modules 09–12) — before anything actually starts.

```bash
# Start modules
bash deploy.sh                        # start all modules in dependency order
bash deploy.sh -m 05                  # start a single module
bash deploy.sh -m 02,03,04            # start specific modules (comma-separated)
bash deploy.sh --cpu-only             # skip all GPU modules
bash deploy.sh -e 11                  # start all except module 11

# Lifecycle
bash deploy.sh -m 05 --stop           # stop a module (keep volumes)
bash deploy.sh -m 05 --delete         # stop + remove volumes
bash deploy.sh -m 05 --restart        # stop then start

# Build
bash deploy.sh -m 18 --build          # rebuild image then start
bash deploy.sh -m 18 --build-only     # rebuild the image only, don't start the container

# Validation
bash deploy.sh --check-ports          # detect host-port conflicts + verify container-port comments
bash deploy.sh --check-env            # find unfilled <placeholder> values in the env file

# Info
bash deploy.sh --list                 # list all modules
bash deploy.sh --status               # show running / stopped status
bash deploy.sh -m 18 --logs           # follow docker compose logs

# Logging
bash deploy.sh -l build.log           # save output to file
```

See `deployment/README.md` for the full dependency graph and module reference.

---

## Configuration

- Each module has its own `.env` file under `deployment/modules/<module>/` — create it from the committed `.env.example` template (see [`BUILD.md`](BUILD.md))
- All services share the `raptor` external Docker bridge network
- Shared PostgreSQL (`raptor-postgres`) is used by all services except Keycloak
- Shared Redis: all modules connect to the **standalone** instance (`raptor-redis-standalone`). Module 02's 6-node cluster is **not started by default** — enable it with `COMPOSE_PROFILES=cluster`. Most modules are hard-wired to the standalone client; **Module 16 auto-detects** standalone vs cluster from `REDIS_HOST` (see `16-training-service/src/core/redis_factory.py`) and adapts its client accordingly, so it works against either without a code change
- GPU allocation is managed by module 07 (`07-ai-ml-services`)

---

## Key Service Ports

All ports below are the shipped `.env.example` defaults (`PORT_*`/`*_PORT` env
vars) — every one is overridable and should be re-verified against the actual
`.env` in use, not assumed.

| Service            | Port   |
| ------------------ | ------ |
| API Gateway        | `8012` |
| AI Lifecycle API   | `8010` |
| Grafana            | `3000` |
| AKHQ (Kafka UI)    | `8180` |
| Keycloak           | `8080` |
| MLflow             | `5555` |
| Query Orchestrator | `8843` |
| Agent Protocol     | `8030` |
| Guardrail Service  | `8023` |
| ArcadeDB (HTTP)    | `2480` |
| Personal DB        | `8025` |
| Memory Service     | `8026` |
| MCP Server         | `8027` |

Deprecated modules' ports (Neo4j `7474`, Graph Service `8844`, Hybrid Search) are omitted — see [Deprecated Modules](#deprecated-modules).
