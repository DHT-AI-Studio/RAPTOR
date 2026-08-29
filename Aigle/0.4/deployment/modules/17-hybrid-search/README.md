# Hybrid Search API

OpenSearch + Qdrant hybrid search API, supporting vector search, BM25 search, hybrid search (RRF fusion), and dynamic payload schemas.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Supported Payload Schemas](#supported-payload-schemas)
- [Adding a Custom Payload Schema](#adding-a-custom-payload-schema)
- [Environment Configuration](#environment-configuration)
- [OpenSearch Prometheus Exporter Plugin](#opensearch-prometheus-exporter-plugin)
- [API Endpoints](#api-endpoints)
  - [Health Check](#health-check)
  - [Document Ingest](#document-ingest)
  - [Search](#search)
- [Configuration Reference](#configuration-reference)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Application                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Ingest API │  │   Search API │  │  Health API  │      │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘      │
│         │                │                                  │
│         └────────┬───────┘                                  │
│                  ▼                                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Service Container (dependency injection)  │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐            │  │
│  │  │Embedding │  │OpenSearch│  │ Qdrant   │            │  │
│  │  │ Manager  │  │ Service  │  │ Service  │            │  │
│  │  └──────────┘  └──────────┘  └──────────┘            │  │
│  │  ┌──────────┐  ┌──────────────────────┐              │  │
│  │  │ Reranker │  │  PayloadSchemaManager │              │  │
│  │  │ Manager  │  │  (unifies filter +    │              │  │
│  │  └──────────┘  │   BM25 + extractor)   │              │  │
│  │                └──────────────────────┘              │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Service Descriptions

| Service | Purpose |
|------|------|
| **Embedding Manager** | Text vectorization (BGE-M3 model) |
| **Reranker Manager** | Reranking (bge-reranker-v2-m3), pulling text via the schema's `content_extractor` |
| **PayloadSchemaManager** | Unified management of each schema's BM25 fields, filter behavior, and rerank text-extraction logic |
| **OpenSearch Service** | BM25 search, full-text retrieval (supports nested structures) |
| **Qdrant Service** | Vector search, similarity matching |
| **Task Manager** | Background task management |

---

## Supported Payload Schemas

`payload_schema` is passed in the **request body**, controlling search filter behavior, BM25 fields, and rerank text-extraction logic.

| Schema | BM25 fields | Filter behavior | Use case |
|--------|-----------|-------------|---------|
| `contextual` (default) | `payload.text`, `payload.summary` | Adds `status=active`, supports type/filename/speaker/source filtering | Standard documents, audio clips |
| `temporal` | `payload.events.description` (nested) | Only adds `payload_schema=temporal`, ignores other request filters | Video events, time series |

### PayloadSchema Field Reference

Each schema defines the following properties (all configured in `DEFAULT_EXTRACTORS` in `app/schemas.py`):

| Field | Description |
|------|------|
| `name` | Schema identifier |
| `description` | Description |
| `content_extractor` | A callable that extracts text from the payload dict for the reranker to use |
| `bm25_fields` | List of fields for OpenSearch BM25 search (supports nested paths) |
| `nested_paths` | Field prefixes that require a nested query |
| `skip_status_filter` | When `True`, doesn't add the `status=active` condition (default `False`) |
| `skip_req_filters` | When `True`, ignores the request's type/filename/speaker/source conditions (default `False`) |

---

## Adding a Custom Payload Schema

Just edit `app/schemas.py` and add a new entry to `DEFAULT_EXTRACTORS` — **no other files need to change**:

```python
def my_extractor(payload: Dict[str, Any]) -> str:
    return payload.get("my_field", "")

DEFAULT_EXTRACTORS["my_schema"] = PayloadSchema(
    name="my_schema",
    description="Description of the custom schema",
    content_extractor=my_extractor,
    bm25_fields=["payload.my_field"],
    nested_paths=[],
    skip_status_filter=False,
    skip_req_filters=False,
)
```

---

## Environment Configuration

### 1. Set environment variables

Create a `.env` file (Docker Compose uses the container name as the host):

```env
# OpenSearch Settings
OPENSEARCH_HOST=opensearch-node1       # Docker: container name; local testing: localhost
OPENSEARCH_PORT=9200
OPENSEARCH_INDEX=hybrid_index
OPENSEARCH_USER=admin
OPENSEARCH_PASSWORD=your_password
OPENSEARCH_DASHBOARDS_HOST=opensearch-dashboards
PORT_OPENSEARCH_DASHBOARDS=5601
VERIFY_CERTS=False                     # Set False for self-signed certs; OpenSearch has HTTPS enabled by default

# RRF Fusion Settings
RRF_K_FACTOR=60

# Qdrant Settings
QDRANT_HOST=qdrant                     # Docker: container name; local testing: localhost
PORT_QDRANT=6333
QDRANT_COLLECTION=raptor

# Embedding Model Settings
EMBEDDING_MODEL=BAAI/bge-m3
VECTOR_DIM=1024

# Reranker Model Settings
RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# HuggingFace Cache (model cache path; HF_HOME is fixed to /hf_cache inside the container, cannot be changed via .env)
HF_CACHE_PATH=/path/to/host/huggingface/cache  # host mount path

# App Settings
DEBUG=false
```

> **Note:** OpenSearch has HTTPS enabled by default (security plugin on) — the API connects with `use_ssl=True` + `verify_certs=False`; do not set `plugins.security.disabled: true`.

### 2. Start the service

```bash
docker compose up -d
```

Startup order: `opensearch-node1` → once its health check passes → the `api` container starts (controlled by `depends_on`).

---

## OpenSearch Prometheus Exporter Plugin

`opensearch-node1`/`opensearch-node2` don't use `opensearchproject/opensearch:3.4.0` directly —
instead they use a custom image built from `opensearch/Dockerfile`
(`raptor/opensearch-prometheus:3.4.0.0`), which installs the
[`opensearch-prometheus-exporter`](https://github.com/opensearch-project/opensearch-prometheus-exporter)
plugin at build time (its version must exactly match the OpenSearch version, currently `3.4.0.0`).

Both nodes must use the same image (the plugin set must match to form a cluster), so both point
at the same `build: ./opensearch`. Remember to update the plugin version in lockstep when
upgrading the OpenSearch version.

Metrics endpoint: `GET https://opensearch-node1:9200/_prometheus/metrics` (the security plugin
requires HTTPS + basic auth, using the same credentials as `OPENSEARCH_USER`/`OPENSEARCH_PASSWORD`).
See [14-monitoring/README.md](../14-monitoring/README.md) for the Prometheus scrape config — the
password goes through `basic_auth.password_file` (auto-generated from `.env` by build.py, not
committed to git), not written as plaintext in the config file.

Switching to this custom image for the first time requires `docker compose up -d --build` (or
`python build.py -m 17 --build`) — a plain `up -d` won't trigger a build.

---

## API Endpoints

### Health Check

```http
GET /api/v1/health/live
GET /api/v1/health/ready
```

`/ready` returns the ready status of the embedding model, reranker, OpenSearch, and Qdrant individually.

---

### Document Ingest

`payload_schema` is passed as a **query parameter**, determining the text-extraction logic used at ingest time.

#### Ingesting a JSON body

```http
POST /api/v1/ingest/json?payload_schema=contextual
Content-Type: application/json
```

**Temporal mode example:**

```http
POST /api/v1/ingest/json?payload_schema=temporal
```
```json
[
  {
    "id": "video1",
    "payload": {
      "video_id": "vid_001",
      "filename": "demo.mp4",
      "duration": 120.5,
      "events": [
        {
          "start_timestamp": 10.2,
          "end_timestamp": 15.8,
          "description": "A person walks into the room",
          "objects": ["person", "door"]
        }
      ]
    }
  }
]
```

#### Ingesting a JSON file

```http
POST /api/v1/ingest/file?payload_schema=contextual
POST /api/v1/ingest/file-background?payload_schema=contextual
```

Background ingest returns a `task_id`, whose progress can be checked via `GET /api/v1/ingest/task/{task_id}`.

---

### Search

`payload_schema` is passed in the **request body** (consistent across every search endpoint).

#### 1. Hybrid Search

```http
POST /api/v1/search/hybrid
Content-Type: application/json
```

Flow: BM25 (OpenSearch) + Vector (Qdrant) → RRF fusion → Reranker → Top-K results

**Request Body:**

| Field | Type | Required | Description |
|------|------|------|------|
| `query` | string | ✅ | Search query string |
| `top_k` | integer | ❌ | Number of results to return, default 10 |
| `payload_schema` | string | ❌ | Schema name, default `contextual` |
| `embedding_type` | string | ❌ | Field to vectorize, `text` or `summary` |
| `type` | string/list | ❌ | Document type filter |
| `filename` | list | ❌ | Filter by specific filename |
| `speaker` | list | ❌ | Speaker filter (audio/video) |
| `source` | string | ❌ | Source file type (e.g., `pdf`, `docx`) |

**Example:**

```json
{
  "query": "the budget issue raised in the meeting",
  "top_k": 10,
  "payload_schema": "contextual",
  "type": ["documents", "audios"]
}
```

**Temporal example:**

```json
{
  "query": "a person walks into the room",
  "top_k": 5,
  "payload_schema": "temporal"
}
```

#### 2. Vector Search

```http
POST /api/v1/search/vector
Content-Type: application/json
```

Goes through Qdrant only, no BM25 or rerank. Request body format is the same as hybrid.

#### 3. BM25 Search

```http
POST /api/v1/search/bm25
Content-Type: application/json
```

Goes through OpenSearch BM25 only, no vector or rerank. Request body format is the same as hybrid.

---

## Configuration Reference

| Variable | Default | Description |
|------|--------|------|
| `OPENSEARCH_HOST` | localhost | OpenSearch address (use the container name inside Docker) |
| `OPENSEARCH_PORT` | 9200 | OpenSearch port |
| `OPENSEARCH_INDEX` | hybrid_index | OpenSearch index name |
| `OPENSEARCH_USER` | admin | OpenSearch username |
| `OPENSEARCH_PASSWORD` | admin | OpenSearch password |
| `VERIFY_CERTS` | False | SSL certificate verification — set False for self-signed certs |
| `QDRANT_HOST` | localhost | Qdrant address (use the container name inside Docker) |
| `PORT_QDRANT` | 6333 | Qdrant port |
| `QDRANT_COLLECTION` | raptor | Qdrant collection name |
| `RRF_K_FACTOR` | 60 | RRF fusion constant (higher = smoother) |
| `RERANK_DEPTH` | 30 | Max documents passed to the reranker (lowering it reduces latency; must not go below top_k) |
| `EMBEDDING_MODEL` | BAAI/bge-m3 | Vectorization model |
| `VECTOR_DIM` | 1024 | Vector dimension |
| `RERANKER_MODEL` | BAAI/bge-reranker-v2-m3 | Reranker model |
| `HF_CACHE_PATH` | — | Host-side mount path for the HuggingFace cache (fixed at `/hf_cache` inside the container, cannot be changed via `.env`) |
| `DEBUG` | false | When enabled, error responses include a traceback |
