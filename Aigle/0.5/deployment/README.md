# Deployment Package

## Overview

36 module IDs under `modules/` (`01`–`36`), each an independent Docker Compose
stack. All services share the `raptor` external Docker bridge network.

Three modules (`17`, `19`, `20`) are **deprecated** — retired from the live pipeline
in favor of Module 25's ArcadeDB-backed index, kept in the tree for rollback only.
Four more (`28`, `30`, `31`, `32`) are empty scaffolding — no application code yet.
Don't start either group. See the root [`README.md`](../README.md#deprecated-modules)
for detail.

---

## Module Reference

| ID | Directory              | GPU | Description                                                       |
| -- | ---------------------- | --- | ----------------------------------------------------------------- |
| 01 | 01-nfs-server          | No  | NFS Server — shared storage backing SeaweedFS volumes            |
| 02 | 02-redis-cluster       | No  | Redis standalone (default) + optional 6-node cluster via `COMPOSE_PROFILES=cluster` |
| 03 | 03-database            | No  | PostgreSQL (shared, used by all services except Keycloak); also runs a Qdrant container, unused since Module 17's retirement |
| 04 | 04-object-storage      | No  | SeaweedFS + LakeFS + Asset Management API                         |
| 05 | 05-kafka-cluster       | No  | Kafka KRaft cluster (3 controllers + 3 brokers) + AKHQ UI         |
| 06 | 06-authentication      | No  | Keycloak identity provider                                        |
| 07 | 07-ai-ml-services      | Yes | MLflow tracking server + AI Lifecycle API                         |
| 08 | 08-media-worker        | Yes | Shared GPU base image (torch + PaddlePaddle + WhisperX + docling) |
| 09 | 09-audio-processing    | Yes | WhisperX STT / diarization / PANNs / audio summary workers        |
| 10 | 10-image-processing    | Yes | InternVL image analysis + hybrid search workers                   |
| 11 | 11-video-processing    | Yes | Video chunking / frame description / OCR / summary workers        |
| 12 | 12-document-processing | Yes | PDF/Office document analysis + LLM summary workers                |
| 13 | 13-api-services        | No  | API Gateway (`:8012`) + Asset Management + Search APIs          |
| 14 | 14-monitoring          | No  | Prometheus + Grafana + Alertmanager + Node Exporter + Loki        |
| 15 | 15-chat-service        | No  | LangGraph RAG pipeline, search via Module 25, Redis memory        |
| 16 | 16-training-service    | Yes | GPU training orchestration via FastAPI                            |
| 17 | 17-hybrid-search       | Yes | **Deprecated** — OpenSearch cluster (2 nodes) + Dashboards + Hybrid Search API. Replaced by Module 25. Do not start. |
| 18 | 18-query-orchestrator  | Yes | Query Orchestrator — intent classification (FAISS+BM25), signal extraction, multi-backend RAG routing |
| 19 | 19-graph-database      | No  | **Deprecated** — Neo4j graph database. Replaced by Module 24/25's ArcadeDB graph store. Do not start. |
| 20 | 20-graph-service       | No  | **Deprecated** — Graph Service, LLM-powered knowledge graph query and reasoning. Replaced by Module 25's `/personal/search/graphrag`. Do not start. |
| 21 | 21-agent-protocol      | No  | Agent Protocol — A2A discovery, orchestration and RAG pipeline (search/graph agents proxy to Module 25) (`:8030`) |
| 22 | 22-benchmark-service   | No  | Benchmark Service — user-defined marking schemas, LLM-as-judge/pairwise scoring, run history, AutoTune (`:8022`) |
| 23 | 23-guardrail-service   | Yes | Guardrail Service — content moderation (Llama Guard 3 / Granite Guardian / GPT-OSS-Safeguard), policy engine, audit logging; shares Module 02/03's Redis/Postgres (`:8023`, disabled by default) |
| 24 | 24-arcadedb-server     | No  | ArcadeDB — multi-model engine (graph + vector + BM25) backing Module 25 (`:2480`) |
| 25 | 25-personal-db-service | Yes | Personal DB — per-user isolated hybrid/graph/temporal search + lifecycle API; the platform's only content index (`:8025`) |
| 26 | 26-memory-service      | Yes | Memory Service — MemVID session / long-term memory, reached through Module 13's gateway (`:8026`) |
| 27 | 27-mcp-server          | No  | MCP Server — exposes Raptor APIs as MCP tools/resources (`:8027`) |
| 28 | 28-carbon-review-agent | — | Placeholder — no application code yet |
| 29 | 29-doc-processing-agent | No | DocAgent — document AI service (classify/extract/summarize/compare/qa/parse_table/search) (`:8290`) |
| 30 | 30-museum-cms          | — | Placeholder — no application code yet |
| 31 | 31-comfyui             | — | Placeholder — no application code yet |
| 32 | 32-3d-rig              | — | Placeholder — no application code yet |
| 33 | 33-privacy-compliance-service | No | Privacy Compliance Service — cross-store GDPR/CCPA erasure, export, DSAR, consent, retention, legal hold (`:8033`, internal) |
| 34 | 34-realtime-audio      | Yes | Realtime audio — WebSocket/gRPC voice chat, ticket auth, VAD, streaming STT/TTS (`:8035` WS, `:8044` gRPC) |
| 35 | 35-inference-service   | Yes | LLM Inference Service — text (translate/summarize/complete/transform) + voice (diarize/STT/TTS/clone) APIs on top of Module 07; only the voice profile reserves a GPU (`:8036`) |
| 36 | 36-grpc-gateway        | No  | gRPC Gateway — RaptorService on `:8825` (health, auth/metrics interceptors, search, assets, jobs) |

---

## Testing Status

Not tracked as a single point-in-time table here — the module set and code have changed too much since the last full pass (2026-05-18, before Modules 23–27 existed and before the Module 17/19/20 retirement) for a static table to stay honest. Rely on CI and each module's own tests; PR descriptions on `main` record what was live-tested and when for that specific change.

---

## build.py Reference

All modules are deployed via `modules/build.py`. The repo-root `deploy.sh` is a thin wrapper — flags are identical.

### Prerequisites

- Docker Engine 24.0+
- Docker Compose v2.20+
- NVIDIA Container Toolkit (for GPU modules)

### Commands

```bash
cd modules

# Start modules
python build.py                        # start all in dependency order
python build.py -m 05                  # start a single module
python build.py -m 02,03,04            # start specific modules (comma-separated)
python build.py --cpu-only             # skip all GPU modules
python build.py -e 11                  # start all except module 11

# Lifecycle
python build.py -m 05 --stop           # stop (keep volumes)
python build.py -m 05 --delete         # stop + remove volumes
python build.py -m 05 --restart        # stop then start

# Build
python build.py -m 18 --build          # rebuild image then start
python build.py -m 18 --build-only     # rebuild the image only, don't start the container

# Validation
python build.py --check-ports          # detect host-port conflicts + verify container-port comments
python build.py --check-env            # find unfilled <placeholder> values in the env file

# Info
python build.py --list                 # list all modules
python build.py --status               # show running / stopped status
python build.py --mem-usage            # VRAM per GPU module/physical GPU + host RAM used by all Raptor containers
python build.py -m 18 --logs           # follow docker compose logs

# Logging
python build.py -l build.log           # save output to file
```

### Module dependencies

Derived from the `deps` field in `build.py`. Modules with no listed dependencies can start in any order.

| ID | Depends on |
|----|------------|
| 01 | — |
| 02 | — |
| 03 | — |
| 04 | 01, 03 |
| 05 | 02 |
| 06 | — |
| 07 | 01, 03, 04 |
| 08 | — |
| 09 | 02, 03, 04, 05, 08 |
| 10 | 02, 03, 04, 05, 08 |
| 11 | 02, 03, 04, 05, 08 |
| 12 | 02, 03, 04, 05, 08 |
| 13 | 02, 03, 04, 06 |
| 14 | 03 |
| 15 | 02 |
| 16 | 01, 02, 07 |
| 17 | — |
| 18 | 03, 04, 13 |
| 19 | — |
| 20 | 02, 19 |
| 21 | 03, 13 |
| 22 | 02, 03, 07, 13 |
| 23 | 02, 03 |
| 24 | — |
| 25 | 02, 03, 05, 07, 13, 24 |
| 26 | 01, 02, 07, 13 |
| 27 | 02, 06, 13 |
| 29 | 02, 03, 07, 12 |
| 33 | 02, 03, 04, 06, 25, 26 |
| 34 | 02, 06, 08 |
| 35 | 02, 07, 08 |
| 36 | 02, 04, 05, 25 |

Modules 17/19/20 are deprecated (see the table above). 15/18 previously
declared a dependency on 17 as leftover wiring from before the retirement —
removed from `build.py` since (their actual search calls go to Module 25).
28/30/31/32 have no `deps` entry because they have no `build.py` entry at
all yet — placeholders, not deployable.

### GPU modules

Modules 07, 08, 09, 10, 11, 12, 16, 18, 23, 25, 26, 34, 35 require a
CUDA-capable GPU (confirmed against each module's own `docker-compose.yml`
device reservation, not just `build.py`'s `gpu=` flag, which can drift out of
sync with it). Module 17 also reserves one but is deprecated — don't start
it. 35's GPU reservation sits only on its `voice` profile. None of 13, 14,
15, 19–22, 24, 27, 29, 33, 36 do.

Module 08 (`media-worker`) is a shared base image used by 09, 10, 11, and 12. Build it once before deploying those workers:

```bash
python build.py -m 08 --build
```


---

## Configuration

- Each module has its own `.env` under `modules/<module>/`
- Shared PostgreSQL (`raptor-postgres`, module 03) is used by all services except Keycloak
- GPU allocation is managed by module 07 (`07-ai-ml-services`)

---

## Key Service Ports

All ports below are the shipped `.env.example` defaults — every one is
overridable and should be re-verified against the actual `.env` in use.

| Service            | Port     |
| ------------------ | -------- |
| API Gateway        | `8012` |
| AI Lifecycle API   | `8010` |
| Grafana            | `3000` |
| AKHQ (Kafka UI)    | `8180` |
| Keycloak           | `8080` |
| MLflow             | `5555` |
| Query Orchestrator | `8843` |
| Agent Protocol     | `8030` |
| Guardrail Service  | `8023` ¹ |
| ArcadeDB (HTTP)    | `2480` |
| Personal DB        | `8025` ¹ |
| Memory Service     | `8026` ¹ |
| MCP Server         | `8027` |
| DocAgent           | `8290` |
| Privacy Compliance | `8033` ¹ |
| Realtime Audio     | `8035` (WS), `8044` (gRPC) |
| Inference Service  | `8036` |
| gRPC Gateway       | `8825`, `9825` (metrics) |

¹ Internal service: published only on `INTERNAL_BIND_IP` (default `127.0.0.1`), so it is reachable from the docker host but not from other machines. Clients go through the API Gateway (`8012`). See [BUILD.md](../docs/BUILD.md) for multi-host deployments.

Deprecated modules' ports (Neo4j `7474`, Graph Service `8844`, Hybrid Search) are omitted — see the Module Reference table above.

---

## Troubleshooting

**Network already exists**

```bash
docker network rm raptor
docker network create raptor
```

**Port conflict**

```bash
ss -tulpn | grep <port>
```

**GPU not available**

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```
