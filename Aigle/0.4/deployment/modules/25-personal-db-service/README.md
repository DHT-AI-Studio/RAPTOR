# Module 25 — personal-db-service

Per-user isolated database lifecycle, hybrid/graph/temporal search, and the Kafka consumer that indexes events from modules 09–12. This is the platform's single content index — it replaced the old per-user Qdrant + OpenSearch + Neo4j trio (modules 17/19/20) with one ArcadeDB database per user. See [`doc/personal-db-service.md`](doc/personal-db-service.md) for the full architectural writeup (tenancy model, schema, responsibility split with module 24).

**Key dependencies:** 02, 03, 05, 07, 13, 24

## Quick start

```bash
cd deployment/modules/25-personal-db-service
cp .env.example .env
docker compose up -d
curl http://localhost:8025/health
open http://localhost:8025/docs
```

Published on `PORT_PERSONAL_DB` (default `8025`; container listens on `8000`).

## Tenancy

Every request carries the tenant in the **`X-Branch-ID`** header — there is no Bearer-JWT auth at this layer (that happens upstream, at the API Gateway / module 13, which derives `X-Branch-ID` from the caller's JWT before proxying in). `X-Branch-ID` maps to a physically separate ArcadeDB database via `db_name_for()` (`X-Branch-ID: alice` → database `user_alice`). This service is not meant to be called directly by end users — it's reached from modules 09–12 (indexing), 13/15/21 (search/RAG), and Kafka (index events).

## Endpoint groups

| Prefix | Router | Purpose |
| --- | --- | --- |
| `/internal/db/*` | `database.py` | Database lifecycle (create/drop/exists per user) |
| `/personal/index/*` | `index.py` (PA-4) | Document/moment chunk indexing |
| `/personal/index/*` | `graph_index.py` (PA-5) | Entity/relationship/temporal-fact indexing (same prefix as PA-4, separate router) |
| `/personal/search/*` | `search.py` (PA-6) | Hybrid / vector / BM25 search over the user's chunks |
| `/personal/search/*` | `graph.py` — `search_router` (PA-7) | Graph / TKG / GraphRAG search (same prefix as PA-6, separate router) |
| `/personal/graph/*` | `graph.py` — `graph_router` (PA-7) | Entity list/detail + raw graph query |
| `/personal/publish/*` | `publish.py` (PA-8) | Publish an index-request straight to Kafka (test/demo path — simulates an upstream worker; the only endpoint here that carries `branch_id` in the body instead of the header) |

Full request/response schemas: `GET /docs` on the running service, or `doc/personal-db-service.md`.
