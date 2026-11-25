# Module 25 — personal-db-service

Per-user isolated ArcadeDB database: hybrid (BM25 + vector) search, GraphRAG, a
temporal knowledge graph, and People Search (face identity) — one database per
user (`user_<branch_id>`), physically isolated rather than a shared table with a
tenant column. Backed by [24-arcadedb-server](../24-arcadedb-server/). Reached
through Module 13's gateway; this module never talks to clients directly.

Full architecture/API design doc: [`doc/personal-db-service.md`](doc/personal-db-service.md).
Temporal grammar and interval primitives: [`doc/temporal.md`](doc/temporal.md) — run
their pure tests with `python -m pytest tests/unit -q`.

**Key dependencies:** 02, 03, 05, 07, 13, 24

## Routers

| File | Covers |
|---|---|
| `database.py` | Per-user database lifecycle |
| `index.py` | Chunk/entity indexing, cloning (Module 04's content-dedup fast path) |
| `search.py` | Hybrid search, GraphRAG, people-in-videos (scoped lookup) |
| `identity.py` | People Search identity CRUD, bind/unbind/merge, seed photos |
| `face_index.py` | Face detection/group indexing |
| `graph.py` / `graph_index.py` | Graph entity/relation queries and indexing |
| `temporal.py` | Temporal fact ingest/extraction/reasoning/verification |
| `publish.py` | `/personal/publish/*` — the one endpoint family taking `branch_id` in the body instead of `X-Branch-ID`, since it simulates an upstream worker |
| `gdpr.py` | Erasure/export cascades |

## Reranker

Local cross-encoder (`PD_RERANKER_MODEL`, default `BAAI/bge-reranker-v2-m3`),
`PD_RERANKER_MAX_LENGTH` (default 512) caps tokenized length per pair so one long
outlier doesn't pad an entire batch.

## Deploy

```bash
python build.py -m 25
```
