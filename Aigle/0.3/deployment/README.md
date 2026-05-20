# Deployment Package

## Overview

21 modules under `modules/`, numbered `01`–`21` in dependency order. Each module is an independent Docker Compose stack. All services share the `raptor` external Docker bridge network.

---

## Module Reference

| ID | Directory              | GPU | Description                                                       |
| -- | ---------------------- | --- | ----------------------------------------------------------------- |
| 01 | 01-nfs-server          | No  | NFS Server — shared storage backing SeaweedFS volumes            |
| 02 | 02-redis-cluster       | No  | Redis 7-node cluster + standalone instance                        |
| 03 | 03-database            | No  | PostgreSQL (shared) + Qdrant vector DB                            |
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
| 15 | 15-chat-service        | No  | LangGraph RAG pipeline with hybrid search and Redis memory        |
| 16 | 16-training-service    | Yes | GPU training orchestration via FastAPI                            |
| 17 | 17-hybrid-search       | Yes | OpenSearch cluster (2 nodes) + Dashboards + Hybrid Search API     |
| 18 | 18-query-orchestrator  | No  | Query Orchestrator — intent classification (FAISS+BM25), signal extraction, multi-backend RAG routing |
| 19 | 19-graph-database      | No  | Neo4j graph database                                              |
| 20 | 20-graph-service       | No  | Graph Service — LLM-powered knowledge graph query and reasoning (`:8843`) |
| 21 | 21-agent-protocol      | No  | Agent Protocol — A2A discovery, orchestration and RAG pipeline (`:8030`) |

---

## Testing Status

Last tested on host **123** (2026-05-18) and remote host **165** (192.168.157.165, 2026-05-18).

| ID | Module              | Status                                                          |
| -- | ------------------- | --------------------------------------------------------------- |
| 01 | nfs-server          | Verified                                                        |
| 02 | redis-cluster       | Verified                                                        |
| 03 | database            | Verified                                                        |
| 04 | object-storage      | Verified                                                        |
| 05 | kafka-cluster       | Verified                                                        |
| 06 | authentication      | Verified                                                        |
| 07 | ai-ml-services      | Verified                                                        |
| 08 | media-worker        | Verified                                                        |
| 09 | audio-processing    | Verified                                                        |
| 10 | image-processing    | Verified                                                        |
| 11 | video-processing    | Verified (host 165); not tested on host 123 — insufficient VRAM |
| 12 | document-processing | Verified                                                        |
| 13 | api-services        | Verified                                                        |
| 14 | monitoring          | Incomplete                                                      |
| 15 | chat-service        | Verified                                                        |
| 16 | training-service    | Verified                                                        |
| 17 | hybrid-search       | Verified                                                        |
| 18 | query-orchestrator  | Verified                                                        |
| 19 | graph-database      | Verified                                                        |
| 20 | graph-service       | Verified                                                        |
| 21 | agent-protocol      | Verified                                                        |

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

### GPU modules

Modules 07, 08, 09, 10, 11, 12, 16, 17 require a CUDA-capable GPU.

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

| Service            | Port     |
| ------------------ | -------- |
| API Gateway        | `8012` |
| AI Lifecycle API   | `8010` |
| Grafana            | `3031` |
| AKHQ (Kafka UI)    | `8280` |
| Keycloak           | `8080` |
| MLflow             | `5000` |
| Neo4j (HTTP)       | `7474` |
| Query Orchestrator | `8843` |
| Graph Service      | `8843` |
| Agent Protocol     | `8030` |

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
docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi
```
