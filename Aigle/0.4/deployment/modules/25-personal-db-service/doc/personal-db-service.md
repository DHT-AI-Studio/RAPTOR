# Personal DB Service (Module 25) & ArcadeDB Server (Module 24) — PA-1 … PA-10

FastAPI app `Raptor 0.4 — Personal DB Service` (v0.4.0)
Branch: `personalDB-joe`
Module roots: `deployment/modules/25-personal-db-service` (service) · `deployment/modules/24-arcadedb-server` (ArcadeDB stack — PA-1, [§10](#10-module-24--arcadedb-server-pa-1))

---

## 1. Overview

A **per-user database service**: each user gets **one ArcadeDB database**, replacing the Raptor 0.3 per-user trio of Qdrant (dense vectors) + OpenSearch (BM25) + Neo4j (graph). A single unified **`Chunk`** vertex is the searchable unit for all four media types (`documents | videos | images | audios`), discriminated by `type` + `embedding_type`. One `Chunk` carries:

- a dense-vector **HNSW** index — replaces Qdrant,
- a **full-text** (Lucene/BM25) index — replaces OpenSearch,
- **graph edges** to entities and temporal facts — replaces Neo4j.

So one database does hybrid search, knowledge-graph traversal, temporal-knowledge-graph queries, and GraphRAG over the same physical records — no cross-store fan-out or score reconciliation.

### Tenancy

Every request carries the tenant in the **`X-Branch-ID`** header (the gateway derives it from the JWT). It maps to a database name via `db_name_for`:

```python
s = re.sub(r"[^A-Za-z0-9_-]", "_", branch_id.strip()).strip("_-") or "anon"
db = f"user_{s}"          # e.g. X-Branch-ID: alice → user_alice
```

Isolation is **physical** — separate ArcadeDB databases, not a shared table with a tenant column. (`/personal/publish/*` is the one exception: it carries `branch_id` in the body, since it simulates an upstream worker.)

### Responsibility split

Module 25 is primarily **storage + search**, and in production does not compute embeddings itself:

| Concern | Owner |
|---|---|
| Embeddings (1024-dim) | **Module 07** supplies pre-computed vectors; Module 25 embeds locally with BGE-M3 as a self-contained fallback |
| Entity / relationship / temporal-fact **extraction** — video | **Module 25 itself** (`app/services/graph_extractor.py`) — LLM-based (Ollama `/chat/completions`, same contract as Module 20), runs as a background task off the Kafka consumer, keyed off the video's own summary + moments. Ported from Module 20's `graph_builder.py`/`tkg.py`, not from Module 12. |
| Entity / relationship / temporal-fact **extraction** — audio / image / document | Not yet implemented — `entities`/`relationships`/`temporal_facts` stay empty for these media types (Workers 09-10, 12 still publish `[]`) |
| Publishing uploads to Kafka | **Workers 09-12** |
| Per-user storage, indexing, search | **Module 25 (this service)** |

---

## 2. ArcadeDB schema (`app/services/schema_init.py`)

`initialize_schema(client, db)` applies the DDL idempotently (`IF NOT EXISTS`), sized by `settings.vector_dim` (1024) and `settings.vector_similarity` (COSINE).

### Vertices

| Vertex | Key properties |
|---|---|
| **`Chunk`** | `chunk_id`, `type`, `embedding_type`, `text`, `summary`, `filename`, `source`, `asset_path`, `version_id`, `upload_time`, `status`, `embedding` (float[]), `sparse_indices`/`sparse_weights`; **video/audio segments**: `start_sec`, `end_sec`, `speaker`, `moment_index`; **video**: `contextual_text`, `asr_text`, `lvlm_desc`, `ocr_text`; **audio**: `audio_labels`; **document**: `chunk_index`, `page_numbers`, `section_heading`, `element_types`, `char_count` |
| **`Source`** | one per uploaded asset: `version_id`, `filename`, `asset_path`, `media_type`, `title`, `summary`, `chunk_count`, `processed_at`, `status` |
| **`Entity`** | cross-media (no source id): `entity_id`, `name`, `type`, `description`, `mention_count`, `created_at`, `updated_at` |
| **`TemporalFact`** | `fact_id`, `entity`, `entity_id`, `relation`, `value`, `time_start`, `time_end`, `confidence`, `created_at`, `source_version_id` |

### Edges

| Edge | Direction | Properties |
|---|---|---|
| `HAS_CHUNK` | Source → Chunk | — |
| `MENTIONS` | Chunk → Entity | `modality` (text / visual / asr) |
| `RELATION` | Entity → Entity | `relation`, `confidence`, `source_version_id` |
| `HAS_TEMPORAL_FACT` | Entity → TemporalFact | — |
| `OBSERVED_IN` | TemporalFact → Chunk | — |
| `CO_OCCURS_WITH` | Entity ↔ Entity | *declared but not written by any indexer* |

### Indexes

- **Dense vector (HNSW)**: `Chunk(embedding)` `LSM_VECTOR {dimensions:1024, similarity:COSINE}`
- **Full-text (BM25)**: `Chunk(text)`, `Chunk(summary)`, `Chunk(asr_text)`, `Chunk(lvlm_desc)`, `Chunk(contextual_text)`, `Entity(name)`
- **Sparse vector (future)**: `Chunk(sparse_indices, sparse_weights)` `LSM_SPARSE_VECTOR`
- **Keyword filters (NOTUNIQUE)**: `Chunk(type)`, `Chunk(status)`, `Chunk(version_id)`, `Chunk(embedding_type)`, `Entity(type)`
- **Unique (idempotent upsert)**: `Chunk(chunk_id)`, `Source(version_id)`, `Entity(entity_id)`, `TemporalFact(fact_id)`

---

## 3. API reference

All routes read `X-User-ID` (the JWT `sub`, injected by Module 13; `X-Branch-ID` accepted as a fallback) → **400** if missing. Read/index/search routes → **404** (`DatabaseNotInitializedError`) if the user DB doesn't exist; graph routes map `ValueError` → **400**; other errors → **500**.

### PA-2/3 — lifecycle (`/internal/db`, `database.py`)

| Method | Path | Response |
|---|---|---|
| POST | `/internal/db/init` | `InitResponse{user_id, database, created, status:"ready"}` — create DB (idempotent) + init schema |
| GET | `/internal/db/status` | `StatusResponse{user_id, db_exists, record_counts{chunks, entities, sources, temporal_facts, by_type}}` — 200 with `db_exists:false` when absent |
| DELETE | `/internal/db` | `{user_id, database, deleted:true}` — drop the whole DB; **503** if the deletion audit cannot be written |

The DELETE writes one `action='delete'` row to PostgreSQL `personal_db.personal_db_audit`
**before** ArcadeDB is touched, carrying the record counts as they stood at that moment —
once the database is dropped there is nothing left to count. If the audit write fails the
delete is refused with 503 rather than performed unrecorded (`PD_AUDIT_REQUIRED=0` opts out
for local runs with no Module 03). The DDL lives in Module 03's `init/postgresql/001_init.sql`.

### PA-4 — chunk indexing (`/personal/index`, `index.py`)

| Method | Path | Status | Response |
|---|---|---|---|
| POST | `/personal/index/chunk` | 201 | `IndexResponse{rid, status:"indexed"}` |
| DELETE | `/personal/index/{version_id}` | 200 | `{version_id, chunks, relationships, temporal_facts, sources, orphan_entities}` |

**`ChunkIndexRequest`** — required: `chunk_id`, `type` (`documents|videos|images|audios`), `embedding_type` (`text|summary`). `embedding` is **optional** `List[float]` (exactly 1024 when present); if omitted the service auto-embeds `text`/`summary` with local BGE-M3. All other fields optional (see [§2](#2-arcadedb-schema-appservicesschema_initpy) for the full media-specific set); `status` defaults `"active"`.

### PA-5 — graph indexing (`/personal/index`, `graph_index.py`)

All return `IndexResponse{rid, status:"indexed"}` (201).

| Method | Path | Body | Purpose |
|---|---|---|---|
| POST | `/personal/index/entity` | `EntityIndexRequest` | upsert Entity (idempotent on `entity_id`); `source_chunk_id` → `MENTIONS` + `mention_count` |
| POST | `/personal/index/relationship` | `RelationshipIndexRequest` | `RELATION` edge (deduped on from/to/relation) |
| POST | `/personal/index/temporal-fact` | `TemporalFactIndexRequest` | upsert `TemporalFact` (+ `HAS_TEMPORAL_FACT`, `OBSERVED_IN`) |

- `EntityIndexRequest`: `entity_id`, `name`, `type` (req); `description`, `source_chunk_id`, `modality` (opt).
- `RelationshipIndexRequest`: `from_entity_id`, `to_entity_id`, `relation` (req); `confidence`, `source_version_id` (opt).
- `TemporalFactIndexRequest`: `fact_id`, `entity`, `relation`, `value` (req); `entity_id`, `time_start`, `time_end`, `confidence`, `chunk_id`, `source_version_id` (opt).

> **Ordering matters**: index the endpoint **entities before** their relationships/temporal facts. If an endpoint entity doesn't exist yet, the edge's `SELECT` subquery is empty and ArcadeDB creates **no edge** — the call still returns 201 with an empty `rid`, no error. See [§8](#8-behaviors--caveats).

### PA-6 — search (`/personal/search`, `search.py`)

| Method | Path | Semantics |
|---|---|---|
| POST | `/personal/search/hybrid` | dense + BM25 fused with **RRF** (`vector.fuse`) |
| POST | `/personal/search/vector` | dense vector only |
| POST | `/personal/search/bm25` | BM25 full-text only |

**`SearchRequest`**: `query` (req), `top_k`=10, optional filters `type` (str or list), `embedding_type`, `status`, `version_id`, `filename`, `source`, `speaker`.
**`SearchResponse`**: `results: [{id, score, payload}]`, `timing: {...ms}`.

### PA-7 — graph / TKG / GraphRAG + raw query (`graph.py`)

| Method | Path | Body | Response |
|---|---|---|---|
| POST | `/personal/search/graph` | `GraphSearchRequest{entity_name, max_depth=2, query?}` | `{entities, edges, paths}` |
| POST | `/personal/search/tkg` | `TKGRequest{entity_name?, time_start?, time_end?, top_k=50}` | `{facts}` (conf. desc) |
| POST | `/personal/search/graphrag` | `GraphRAGRequest{query, top_k=10}` | `{results, timing}` |
| GET | `/personal/graph/entities` | query `type?, limit=50, offset=0` | `{entities, total, limit, offset}` |
| GET | `/personal/graph/entities/{name}` | — | `{entity, outgoing, incoming}` (404) |
| POST | `/personal/graph/query` | `RawGraphQueryRequest{query}` (**read-only SELECT**) | `{result}` |

`max_depth` is clamped 1..5. The raw-query endpoint and the `graph` query-override are validated by a read-only guard (see [§8](#8-behaviors--caveats)).

### PA-8 — Kafka publish (`/personal/publish`, `publish.py`)

| Method | Path | Body |
|---|---|---|
| POST | `/personal/publish/index-request` | `PublishIndexRequest{branch_id (req), version_id?, chunks[], entities[], relationships[], temporal_facts[]}` |

Wraps the flat lists into the worker envelope and publishes to Kafka; returns `{status:"published", topic, branch_id, counts:{...}}`. A **test/demo producer** that simulates workers 09-12 so the whole ingest pipeline can be driven from Swagger (see [§6](#6-kafka-ingest-pipeline-pa-8)).

---

## 4. Indexing internals

### `index_chunk` (`app/services/indexer.py`)

1. `_ensure_ready` — 404 if the DB doesn't exist (DB creation is PA-2's job).
2. **Auto-embed** when `embedding is None`: pick `text` (if `embedding_type=="text"`) or `summary`, falling back to whichever is present; `ValueError` if neither; embed with local BGE-M3.
3. `UPDATE Chunk SET <provided fields> UPSERT RETURN AFTER @rid WHERE chunk_id = :chunk_id` — idempotent via the unique index; only non-null fields are written.
4. If `version_id` present → `_link_source`: upsert the `Source` vertex, then create `HAS_CHUNK(Source→Chunk)` exactly once (check-then-create).

### `delete_by_version` (`DELETE /personal/index/{version_id}`)

Cascade for one uploaded asset: collect affected entities → delete the asset's `Chunk`s (cascading `MENTIONS`/`OBSERVED_IN`/`HAS_CHUNK`) → delete `RELATION`/`TemporalFact`/`Source` tagged with the version → **recompute `mention_count`** on affected entities from surviving edges → **prune orphan entities** (`mention_count==0 AND both().size()==0`). Idempotent; returns per-type delete counts.

### Graph indexers (`app/services/graph_indexer.py`)

- **`index_entity`** — upsert on `entity_id`; if `source_chunk_id` given and no `MENTIONS` edge exists yet, create it (+ `modality`) and recompute `mention_count`. Re-indexing an existing mention is a no-op and **skips** the count refresh.
- **`index_relationship`** — dedup on `(relation, from, to)`; creates the `RELATION` edge only if both endpoint entities exist (else silent no-op — see [§8](#8-behaviors--caveats)).
- **`index_temporal_fact`** — upsert on `fact_id`; add `HAS_TEMPORAL_FACT` (if `entity_id`) and `OBSERVED_IN` (if `chunk_id`), each once.

---

## 5. Search internals (`app/services/searcher.py`)

Query embeddings come from `app/services/embedder.py` — a local `SentenceTransformer("BAAI/bge-m3")`, loaded once lazily (thread executor, `asyncio.Lock`), `normalize_embeddings=True` → 1024-dim vectors. Device from `PD_EMBEDDING_DEVICE` (`auto` ⇒ let sentence-transformers pick cuda→cpu). **No HTTP dependency on Module 07.**

- **`vector_search`** — embed query → `expand(vectorNeighbors('Chunk[embedding]', :qvec, top_k))` + filters; score = `1 - distance`.
- **`bm25_search`** — `SEARCH_INDEX('Chunk[text]') OR SEARCH_INDEX('Chunk[summary]')` (chunk text **and** whole-asset summary) + filters; rank-based scores.
- **`hybrid_search`** — `vector.fuse(vectorNeighbors(...), (SELECT ... BM25 ...), {fusion:'RRF', limit:top_k})`; combines dense + BM25 inside the database. *(0.3's cross-encoder rerank after RRF is not yet ported — TODO.)*
- **`graph_search`** — `TRAVERSE both('RELATION') ... MAXDEPTH <depth>` from the named entity → entities + `RELATION` edges + `shortestPath` paths. An optional raw `query` override must pass the read-only guard.
- **`tkg_search`** — `TemporalFact` filtered by entity/time window, ordered by `confidence DESC`.
- **`graphrag_search`** — vector-rank chunks, then a **second query** backfills each chunk's mentioned entities (`out('MENTIONS')` does not resolve on the temporary records produced by `expand(vectorNeighbors(...))`, hence two queries).
- **`list_entities` / `get_entity`** — entity listing (by `mention_count DESC`) and detail with incoming/outgoing relations.

**Read-only SQL guard** (`is_read_only_select`): single statement, must start with `SELECT`, rejects `insert|update|delete|create|drop|alter|truncate|import|grant|revoke|rebuild|move|traverse into` and statement chaining.

---

## 6. Kafka ingest pipeline (PA-8)

Keeps each personal DB in sync with uploads automatically. Workers 09-12 publish the same index payload they send to the global DBs onto the `personal-index-requests` topic; this service's consumer routes each message to the right per-user DB.

### Consumer (`app/services/kafka_consumer.py`)

- Topic `personal-index-requests`, group `personal-db-service`, `auto_offset_reset=earliest`, **manual commit after successful handling** (no message lost on crash).
- **Dedup**: Redis key `personal:indexed:{chunk_id}` with 7-day TTL; `chunk_id` is the sole dedup key.
- **Auto-creates** the user DB + schema on first message for a new branch (consumer-side equivalent of `/internal/db/init`).
- **Embeds** content if the message carries no vector (same BGE-M3 fallback as PA-4).
- Routes graph-layer entries to the PA-5 indexers; a missing required field skips that entry (logged) instead of failing the whole message.
- **Video graph extraction**: after a video's chunks are indexed, spawns `graph_extractor.extract_and_index_video()` as a background `asyncio` task (bounded by `PD_GRAPH_EXTRACTION_MAX_CONCURRENCY`, default 2) rather than awaiting it inline — `personal-index-requests` is one topic shared by every media type and every user, and the LLM calls this makes (up to `PD_GRAPH_EXTRACTION_MAX_MOMENTS`+1, 90s timeout each) would otherwise block every other queued message behind a slow video. Only anchors MENTIONS/OBSERVED_IN edges to `chunk_id`s actually written as `Chunk` rows in the same message. On shutdown, in-flight extractions are cancelled (`drain_graph_extraction_tasks()`), not waited out — graph data here is best-effort, not core search.

**Envelope shape** (also produced by `build_envelope`):

```jsonc
{ "payload": {
    "branch_id": "<branch>",
    "parameters": {
      "version_id": "<v>",
      "chunks":  [ { "id": "c1", "type": "documents", "text": "…" } ],
      "moments": [ … ],
      "entities":      [ { "entity_id": "e1", "name": "…", "type": "ORG", "source_chunk_id": "c1" } ],
      "relationships": [ { "from_entity_id": "e1", "to_entity_id": "e2", "relation": "…" } ],
      "temporal_facts":[ { "fact_id": "t1", "relation": "…", "value": "…", "entity_id": "e1" } ]
    } } }
```

Entries may be flat (as above) or wrapped in a nested `payload`; chunk entries carry `id`. `PD_KAFKA_ENABLED=0` disables the consumer (the HTTP API still runs).

---

## 7. Configuration

`app/core/config.py` — pydantic-settings, env prefix `PD_`:

| Env var | Default | |
|---|---|---|
| `PD_ARCADEDB_URL` | `http://raptor-arcadedb:2480` | ArcadeDB host |
| `PD_ARCADEDB_USER` / `PD_ARCADEDB_PASSWORD` | `root` / _(no default — required)_ | must match Module 24 |
| `PD_POSTGRES_DSN` | `postgresql://raptor:raptor@raptor-postgres:5432/personal_db` | deletion audit (Module 03) |
| `PD_AUDIT_REQUIRED` | `1` | `1` refuses DELETE when the audit DB is down |
| `PD_VECTOR_DIM` / `PD_VECTOR_SIMILARITY` | `1024` / `COSINE` | HNSW index |
| `PD_KAFKA_ENABLED` | `True` | start the consumer |
| `PD_KAFKA_BOOTSTRAP` | `kafka:9092` | broker |
| `PD_KAFKA_TOPIC` / `PD_KAFKA_GROUP_ID` | `personal-index-requests` / `personal-db-service` | |
| `PD_REDIS_URL` | `redis://raptor-redis:6379` | dedup |
| `PD_REDIS_DEDUP_TTL` | `604800` | 7 days |
| `PD_EMBEDDING_MODEL` / `PD_EMBEDDING_DEVICE` | `BAAI/bge-m3` / `auto` | local embedder |
| `PD_HTTP_TIMEOUT` | `60.0` | ArcadeDB client |
| `PD_LLM_BASE_URL` / `PD_CHAT_MODEL_NAME` | `http://host.docker.internal:11434/v1` / `qwen2.5:7b` | Ollama endpoint for personal-graph entity extraction (video only); same values as Module 20's `LLM_BASE_URL`/`CHAT_MODEL_NAME` if sharing one Ollama instance |
| `PD_GRAPH_EXTRACTION_ENABLED` | `True` | `False` disables video graph extraction entirely |
| `PD_GRAPH_EXTRACTION_MAX_MOMENTS` | `10` | cap on per-moment temporal-fact LLM calls per video |
| `PD_GRAPH_EXTRACTION_MAX_CONCURRENCY` | `2` | max videos extracting concurrently — runs as background tasks so a burst of uploads can't flood the LLM endpoint |

Deps (`requirements.txt`): fastapi, uvicorn[standard], httpx, pydantic(-settings), sentence-transformers, aiokafka, redis.

### Run

```bash
conda activate llm
cd deployment/modules/25-personal-db-service

# API only (Kafka consumer off)
PD_ARCADEDB_URL=http://localhost:2480 PD_ARCADEDB_PASSWORD=$ARCADEDB_ROOT_PASSWORD PD_KAFKA_ENABLED=0 \
  PYTHONPATH=. python -m uvicorn app.main:app --host 127.0.0.1 --port 8027

# with the Kafka consumer (needs a broker + redis)
PD_ARCADEDB_URL=http://localhost:2480 PD_ARCADEDB_PASSWORD=$ARCADEDB_ROOT_PASSWORD \
  PD_KAFKA_ENABLED=1 PD_KAFKA_BOOTSTRAP=localhost:9092 PD_REDIS_URL=redis://localhost:6381 \
  PYTHONPATH=. python -m uvicorn app.main:app --host 127.0.0.1 --port 8027
```

Swagger `GET /docs` · health `GET /health`.

---

## 8. Behaviors & caveats

1. **Physical isolation** — one ArcadeDB DB per branch; there is no way for branch B to read branch A's data (verified by `test_isolation`).
2. **Relationship / temporal-fact edges are a silent no-op** when their endpoint entities/chunks don't exist yet (empty `SELECT` subquery → no edge, still HTTP 201). Index entities first.
3. **GraphRAG uses a two-query workaround** — `out('MENTIONS')` can't resolve on `expand(vectorNeighbors(...))` temporary records.
4. **`mention_count` is refreshed only when a new `MENTIONS` edge is created** (idempotent re-index skips the recount).
5. **`CO_OCCURS_WITH`** is declared in the schema but never written.
6. **Hybrid search has no cross-encoder rerank yet** (present in 0.3; marked TODO).
7. **Embedding is optional at the API** — a pre-computed 1024-dim vector (Module 07 / pipeline) is used as-is; otherwise the service embeds locally.

---

## 9. Tests (PA-10)

`pytest.ini`: `asyncio_mode=auto`, `testpaths=tests`. Integration tests hit a **real ArcadeDB** and self-skip if it's unreachable:

```bash
PD_ARCADEDB_URL=http://localhost:2480 PD_ARCADEDB_PASSWORD=$ARCADEDB_ROOT_PASSWORD python -m pytest tests/ -q
```

Fixtures (`conftest.py`): `fake_vector` (deterministic pseudo-embedding), `make_db` (fresh `user_<branch>` per test, auto-dropped), `mock_embed` (monkeypatch the embedder).

| File | Coverage |
|---|---|
| `test_lifecycle.py` | create/schema → empty counts → drop; schema init idempotent |
| `test_index.py` | chunk upsert idempotent; Source + HAS_CHUNK linking; delete-by-version cascade + orphan prune; entity MENTIONS + `mention_count`; relationship dedup; temporal-fact edges |
| `test_search.py` | vector / bm25 / hybrid (ordering + timing); type filter |
| `test_graph.py` | traversal returns entities + edges + paths over a `knows` chain |
| `test_graph_search.py` | read-only SQL guard (allow SELECT, reject DML/chaining); dedup/edges/paths; depth clamp; TKG filters + ordering (pure unit, `FakeClient`) |
| `test_isolation.py` | branch A indexes; A sees 1, B sees 0 |
| `test_kafka.py` | consumer indexes + dedupes; auto-creates DB for a new user; routes entities to the graph indexer |
| `test_embedder.py` | real BGE-M3: 1024-dim normalized vectors; relevant-first ranking |

---

## 10. Module 24 — ArcadeDB server (PA-1)

The infrastructure ticket underneath the Personal DB Service: ArcadeDB deployed as its own Docker Compose stack (`deployment/modules/24-arcadedb-server/`) so Module 25 can create and access one database per user. ArcadeDB is a multi-model engine (graph + vector + BM25 in one process), which is what lets a single per-user database replace the 0.3 Qdrant + OpenSearch + Neo4j trio.

### Compose stack (`24-arcadedb-server/docker-compose.yml`)

| Aspect | Value |
|---|---|
| Image | `arcadedata/arcadedb:latest` (verified against 26.6.x) |
| Container / hostname / network alias | `raptor-arcadedb` |
| HTTP API | `http://raptor-arcadedb:2480` (in-network) — Module 25 connects here |
| Binary protocol | `2424` (optional) |
| Data volume | `arcadedb_data` → `/home/arcadedb/databases` (one directory per user DB) |
| Network | external `${DOCKER_NETWORK}` (= `raptor`) |
| Restart | `unless-stopped` |
| Resources | limits 4 GB / 4 CPU |
| Mode | development (ArcadeDB Studio UI enabled at `:2480`) |

**Root password** is set via `JAVA_OPTS=-Darcadedb.server.rootPassword=${ARCADEDB_ROOT_PASSWORD}`. The `ARCADEDB_SERVER_ROOTPASSWORD` env var is **not** honoured by this image and its use makes startup hang — hence the `JAVA_OPTS` form.

**Health check** uses `wget` (the image ships `wget`, not `curl`):

```yaml
healthcheck:
  test: ["CMD-SHELL", "wget -q -O /dev/null http://localhost:2480/api/v1/ready || exit 1"]
  interval: 30s
  timeout: 10s
  retries: 5
  start_period: 60s
```

`GET /api/v1/ready` returns **HTTP 204** on readiness (`wget -q`/`curl -f` treat any 2xx as success).

### `.env` (`deployment/modules/.env`)

```dotenv
PORT_ARCADEDB_HTTP=2480
PORT_ARCADEDB_BINARY=2424
DOCKER_NETWORK=raptor
TIMEZONE=Asia/Taipei
ARCADEDB_ROOT_PASSWORD=<your password>        # under the "24 — arcadedb-server" section
```

### Registration (`deployment/modules/build.py`)

```python
Module(
    id="24",
    name="24-arcadedb-server",
    description="ArcadeDB multi-model server — per-user graph + vector + BM25 databases",
    deps=[],                                  # foundational — no dependencies
    health_containers=["raptor-arcadedb"],
    steps=[_compose_up("24-arcadedb-server/docker-compose.yml")],
)
```

Module 25 declares `deps=[..., "24"]`, so `raptor-arcadedb` is brought up (and health-gated) before the Personal DB Service starts.

### Deploy

```bash
cd deployment
bash deploy.sh -m 24            # brings up raptor-arcadedb, waits for health
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:2480/api/v1/ready   # → 204
```

Studio UI: `http://localhost:2480` (login `root` / `${ARCADEDB_ROOT_PASSWORD}`).

### Implementation notes (where reality differs from the original AC)

The image behaves differently from the ticket's assumptions; the compose file reflects the **verified** behaviour:

| Original AC | Actual (implemented) |
|---|---|
| Data at `/arcadedb-ce/databases` | `/home/arcadedb/databases` |
| `GET /api/v1/ready` returns 200 | returns **204** |
| root password via env var | via `JAVA_OPTS -Darcadedb.server.rootPassword` (env var hangs startup) |
| "raptor network" | external network name is `raptor` (`DOCKER_NETWORK`) |
| healthcheck with curl | image has no `curl`; uses `wget` |

### How Module 25 consumes it

Module 25 talks to this stack over the HTTP API only (`PD_ARCADEDB_URL`, default `http://raptor-arcadedb:2480` in-network, `http://localhost:2480` from the host). It issues server commands (`create/drop database`) and per-database `command`/`query` calls; each user maps to database `user_<branch_id>` inside the shared `arcadedb_data` volume. See [§1](#1-overview) and [§7](#7-configuration).
