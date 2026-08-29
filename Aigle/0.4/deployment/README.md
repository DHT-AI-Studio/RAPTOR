# Deployment Package

## Overview

27 modules under `modules/`, numbered `01`–`27` in dependency order. Each module is an independent Docker Compose stack. All services share the `raptor` external Docker bridge network.

Three modules (`17`, `19`, `20`) are **deprecated** — retired from the live pipeline in favor of Module 25's ArcadeDB-backed index, kept in the tree for rollback only. Module `22` is still a **placeholder**, not finished. Don't start either group. See the root [`README.md`](../README.md#deprecated-modules) for detail.

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
| 22 | 22-benchmark-service   | No  | Benchmark Service — scoring/marking pipelines, LLM-as-judge, `local_infer` via Module 16 (`:8022`). **Still a placeholder — not finished, do not start.** |
| 23 | 23-guardrail-service   | No  | Guardrail Service — content moderation (Llama Guard 3 / Granite Guardian / GPT-OSS-Safeguard), policy engine, audit logging; shares Module 02/03's Redis/Postgres (`:8023`, disabled by default) |
| 24 | 24-arcadedb-server     | No  | ArcadeDB — multi-model engine (graph + vector + BM25) backing Module 25 (`:2480`) |
| 25 | 25-personal-db-service | No  | Personal DB — per-user isolated hybrid/graph/temporal search + lifecycle API; the platform's only content index (`:8025`) |
| 26 | 26-memory-service      | No  | Memory Service — MemVID session / long-term memory, reached through Module 13's gateway (`:8026`) |
| 27 | 27-mcp-server          | No  | MCP Server — exposes Raptor APIs as MCP tools/resources (`:8027`) |

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
| 15 | 02, 17 |
| 16 | 01, 02, 07 |
| 17 | — |
| 18 | 03, 04, 13, 17 |
| 19 | — |
| 20 | 02, 19 |
| 21 | 03, 13 |
| 22 | 02, 03, 07, 13 |
| 23 | 02, 03 |
| 24 | — |
| 25 | 02, 03, 05, 07, 13, 24 |
| 26 | 01, 02, 07, 13 |
| 27 | 02, 06, 13 |

Modules 17/19/20 are deprecated (see the table above); 15/18's declared
dependency on 17 is leftover wiring from before the retirement, not evidence
17 is still called at runtime — their actual search calls go to Module 25.

### GPU modules

Modules 07, 08, 09, 10, 11, 12, 16, 18 require a CUDA-capable GPU. Module 17
also reserves one but is deprecated — don't start it. None of 22–23 or 24–27
do — all are CPU-only.

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
| Guardrail Service  | `8023` |
| ArcadeDB (HTTP)    | `2480` |
| Personal DB        | `8025` |
| Memory Service     | `8026` |
| MCP Server         | `8027` |

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
