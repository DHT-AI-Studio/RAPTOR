# Raptor 0.3

Multimedia intelligence platform for processing video, audio, documents, and images through AI/ML pipelines. Provides hybrid search, RAG question-answering, knowledge graph reasoning, and A2A agent orchestration. Built entirely in Python with Docker microservices.

API Reference: [`API_REFERENCE.md`](API_REFERENCE.md) · Setup, Build & Configuration: [`BUILD.md`](BUILD.md) · Demo Frontend: [`raptor-demo-frontend/`](raptor-demo-frontend/)

---

## Architecture

21 modules under `deployment/modules/`, numbered `01`–`21` in dependency order. Each module is an independent Docker Compose stack sharing the `raptor` bridge network.

### Media Processing Flow

```
File Upload (API Gateway :8012)
  → Asset Management (LakeFS + SeaweedFS)
  → Kafka topic
  → Processing workers (audio / video / image / document)
    → Transcription (WhisperX) + OCR + Frame description (InternVL) + Summary
  → Embeddings → Qdrant (vector) + OpenSearch (BM25)
  → Knowledge Graph → Neo4j (entities + temporal facts)
  → Accessible via Search / RAG / A2A APIs
```

---

## Modules

### CPU-only

| ID | Module             | Description                                                              |
| -- | ------------------ | ------------------------------------------------------------------------ |
| 01 | nfs-server         | NFS Server — shared storage backing SeaweedFS volumes                   |
| 02 | redis-cluster      | Redis 7-node cluster + standalone instance                               |
| 03 | database           | PostgreSQL (shared) + Qdrant vector DB                                   |
| 04 | object-storage     | SeaweedFS distributed object storage + LakeFS + Asset Management API     |
| 05 | kafka-cluster      | Kafka KRaft cluster (3 controllers + 3 brokers) + AKHQ UI                |
| 06 | authentication     | Keycloak identity provider with embedded PostgreSQL                      |
| 13 | api-services       | API Gateway (`:8012`) + Asset Management + Search + A2A proxy APIs      |
| 14 | monitoring         | Prometheus + Grafana + Alertmanager + Node Exporter + Loki               |
| 15 | chat-service       | Chat service — LangGraph RAG pipeline with hybrid search and Redis memory |
| 18 | query-orchestrator | Query Orchestrator — intent classification (FAISS+BM25), signal extraction, multi-backend RAG routing |
| 19 | graph-database     | Neo4j graph database                                                     |
| 20 | graph-service      | Graph Service — LLM-powered knowledge graph query and reasoning (`:8843`) |
| 21 | agent-protocol     | Agent Protocol — A2A discovery, orchestration and RAG pipeline (`:8030`) |

### GPU required

| ID | Module              | Description                                                                   |
| -- | ------------------- | ----------------------------------------------------------------------------- |
| 07 | ai-ml-services      | MLflow tracking server + AI Lifecycle API                                     |
| 08 | media-worker        | Shared GPU base image (torch 2.7.1 cu128 sm_120/Blackwell + PaddlePaddle 3.3.0 + WhisperX + docling) |
| 09 | audio-processing    | WhisperX STT / diarization / PANNs / audio summary workers                    |
| 10 | image-processing    | InternVL image analysis + hybrid search workers                               |
| 11 | video-processing    | Video chunking / frame description / OCR / summary workers                    |
| 12 | document-processing | PDF/Office document analysis + LLM summary workers                            |
| 16 | training-service    | GPU training orchestration via FastAPI                                        |
| 17 | hybrid-search       | OpenSearch cluster (2 nodes) + Dashboards + Hybrid Search API                 |

---

## Testing Status

Last tested on host **123** (2026-05-18) and remote host **165** (2026-05-18).

| ID | Module              | Status                                           |
| -- | ------------------- | ------------------------------------------------ |
| 01 | nfs-server          | Verified                                         |
| 02 | redis-cluster       | Verified                                         |
| 03 | database            | Verified                                         |
| 04 | object-storage      | Verified                                         |
| 05 | kafka-cluster       | Verified                                         |
| 06 | authentication      | Verified                                         |
| 07 | ai-ml-services      | Verified                                         |
| 08 | media-worker        | Verified                                         |
| 09 | audio-processing    | Verified                                         |
| 10 | image-processing    | Verified                                         |
| 11 | video-processing    | Verified (host 165); not tested on host 123 — insufficient VRAM |
| 12 | document-processing | Verified                                         |
| 13 | api-services        | Verified                                         |
| 14 | monitoring          | Incomplete                                       |
| 15 | chat-service        | Verified                                         |
| 16 | training-service    | Verified                                         |
| 17 | hybrid-search       | Verified                                         |
| 18 | query-orchestrator  | Verified                                         |
| 19 | graph-database      | Verified                                         |
| 20 | graph-service       | Verified                                         |
| 21 | agent-protocol      | Verified                                         |

---

## Deployment

`deploy.sh` at the repo root is a thin wrapper around `deployment/modules/build.py` — every flag is identical.

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

# Info
bash deploy.sh --list                 # list all modules
bash deploy.sh --status               # show running / stopped status
bash deploy.sh -m 18 --logs           # follow docker compose logs

# Logging
bash deploy.sh -l build.log           # save output to file
```

Module 08 (`media-worker`) is a shared base image for modules 09–12. Build it first:

```bash
bash deploy.sh -m 08 --build
```

See `deployment/README.md` for the full dependency graph and module reference.

---

## Configuration

- Each module has its own `.env` file under `deployment/modules/<module>/` — create it from the committed `.env.example` template (see [`BUILD.md`](BUILD.md))
- All services share the `raptor` external Docker bridge network
- Shared PostgreSQL (`raptor-postgres`) is used by all services except Keycloak
- GPU allocation is managed by module 07 (`07-ai-ml-services`)

---

## Key Service Ports

| Service            | Port   |
| ------------------ | ------ |
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
