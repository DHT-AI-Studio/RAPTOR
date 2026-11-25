# Personal DB Service (Module 25) & ArcadeDB Server (Module 24) — PA-1 … PA-10

FastAPI app `Raptor 0.5 — Personal DB Service` (v0.5.0)
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
| **`TemporalFact`** | `fact_id`, `entity`, `entity_id`, `relation`, `value`, `time_start`, `time_end`, `confidence`, `created_at`, `source_version_id`; canonical dates (TMP-3): `s_state`/`e_state` (`known`/`unknown`), `s_lo`/`s_hi`/`e_lo`/`e_hi` (LONG, UTC epoch ms), `precision`, `timezone`, `time_start_raw`/`time_end_raw` (as written), `timeline_id` (TMP-4: `world` for calendar time), `quarantine_reason`, `requires_reextraction`, `reextraction_reason` (why a parsed row is doubtful), `normalizer_version`/`migration_version`; v2 (TMP-8): `axis` (`world`), `time_unit` (`ms`), and — declared, NULL until a writer can establish them — `extent`, `anchor`, `date_text`, `source_sentence`, `evidence_offsets`, `confirmed_active_at`, `derived`, `support_ids`, `network_version`, `media_start_ms`/`media_end_ms`. `time_start`/`time_end` are kept as lossy `YYYY-MM-DD` projections for v1 readers (see *TemporalFact v2* below) |

**People layer** (dev plan v1.4 §4.2–4.5; `_face_ddl()`), written from Module 11's face detection and the Identity API:

| Vertex | Key properties |
|---|---|
| **`FaceTrace`** | one continuous appearance in one video: `trace_id`, `version_id`, `start_sec`, `end_sec`, `frame_count`, `avg_confidence`, `embedding` (512-d ArcFace, mean of the trace's best faces), `frames_json` |
| **`Face`** | `kind` = `representative` (a trace's best frame: `version_id`, `frame_no`, `timestamp_sec`, `bbox`, `quality`, `min_side`) or `seed` (a user's photo of a person: `identity_id`, `embedding`); both: `face_id`, `image_path`, `image_version_id` (a Module 04 file) |
| **`FaceGroup`** | the same person within one video (Module 11's clustering): `group_id`, `version_id`, `embedding` (frame-weighted mean of its traces), `frame_count` |
| **`Identity`** | the same person across videos: `identity_id`, `name`, `aliases_json` ({locale: name}), `identity_type`, `source` (`auto` / `manual`), `identity_status` (`pending` / `active` / `merged` / `deleted` -- deliberately not `status`, see below), `origin_version_id` (pending-person only), `created_at`, `updated_at` |
| **`IdentityHistory`** | one merge or delete: `history_id`, `operation`, `before_snapshot_json` / `after_snapshot_json`, `history_state` (`applied` / `reverted`), `undo_deadline` (+24 h) |

### Edges

| Edge | Direction | Properties |
|---|---|---|
| `HAS_CHUNK` | Source → Chunk | — |
| `MENTIONS` | Chunk → Entity | `modality` (text / visual / asr) |
| `RELATION` | Entity → Entity | `relation`, `confidence`, `source_version_id` |
| `HAS_TEMPORAL_FACT` | Entity → TemporalFact | — |
| `OBSERVED_IN` | TemporalFact → Chunk | — |
| `CO_OCCURS_WITH` | Entity ↔ Entity | *declared but not written by any indexer* |
| `TRACE_IN_SOURCE` | FaceTrace → Source | — (the only link from face data into the content graph) |
| `HAS_KEY_FACE` | FaceTrace → Face | — (its representative frame) |
| `TRACE_IN_GROUP` | FaceTrace → FaceGroup | — |
| `GROUP_OF_IDENTITY` | FaceGroup → Identity | `score` (match similarity), `auto` (made by matching, not a user), `created_at` |
| `HAS_PROFILE_FACE` | Identity → Face | — (the seed photo) |

Face data is kept out of content search by construction: no people-layer vertex has a `status` property (graph_query's subgraph filter drops paths through non-active `status`) and no people-layer edge type is in `graph_query._SUBGRAPH_EDGE_TYPES`, so GraphRAG / TKG / hybrid search never return a face, trace or identity. Takedown needs no separate sync: People Search reads the video's status from the shared `Source` node (`TRACE_IN_SOURCE`).

### Indexes

- **Dense vector (HNSW)**: `Chunk(embedding)` `LSM_VECTOR {dimensions:1024, similarity:COSINE}`
- **Full-text (BM25)**: `Chunk(text)`, `Chunk(summary)`, `Chunk(asr_text)`, `Chunk(lvlm_desc)`, `Chunk(contextual_text)`, `Entity(name)`
- **Sparse vector (future)**: `Chunk(sparse_indices, sparse_weights)` `LSM_SPARSE_VECTOR`
- **Keyword filters (NOTUNIQUE)**: `Chunk(type)`, `Chunk(status)`, `Chunk(version_id)`, `Chunk(embedding_type)`, `Entity(type)`
- **Unique (idempotent upsert)**: `Chunk(chunk_id)`, `Source(version_id)`, `Entity(entity_id)`, `TemporalFact(fact_id)`
- **People layer**: `LSM_VECTOR` on `FaceTrace(embedding)`, `FaceGroup(embedding)`, `Face(embedding)` (`face_vector_dim` 512, cosine); full-text `Identity(name)`; NOTUNIQUE `FaceTrace/FaceGroup/Face(version_id)`, `Identity(identity_status)`, `Identity(origin_version_id)`; UNIQUE `FaceTrace(trace_id)`, `Face(face_id)`, `FaceGroup(group_id)`, `Identity(identity_id)`, `IdentityHistory(history_id)`

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
| POST | `/personal/search/tkg` | `TKGRequest{query, time_start?, time_end?, entity?, keywords?, match?, relations?, certainty=possible, max_depth=2, limit=50, score_threshold=0.5}` | `{query, matched_entities, subgraph_nodes, subgraph_edges, temporal_facts, moment_ids, retrieval, complete}` (`temporal_facts` by `time_start` asc; each with `when` and `precision`; with a window also `allen`) |
| POST | `/personal/search/graphrag` | `GraphRAGRequest{query, top_k=10}` | `{results, timing}` |
| GET | `/personal/graph/entities` | query `type?, limit=50, offset=0` | `{entities, total, limit, offset}` |
| GET | `/personal/graph/entities/{name}` | — | `{entity, outgoing, incoming}` (404) |
| POST | `/personal/graph/query` | `RawGraphQueryRequest{query}` (**read-only SELECT**) | `{result}` |

**TKG retrieval (TMP-4).** A question that names no entity is no longer answered with nothing: an entity is named
(`entity`, else the query text fulltext-matches one) → the facts of the matched entities, plus facts of other entities in the
window whose evidence mentions the name — `keywords` when given, else `entity` — so "Samsung 2010-2016" also finds "Google
announced a special version of the Samsung Galaxy S4" (`entity` given but matching nothing: no facts, never widened); else
`keywords` given →
facts in the window whose evidence (entity, relation, value and the `OBSERVED_IN` passage) contains **every** keyword,
case-insensitive, an English plural also matching its singular; else → every active fact in the window (temporal-only).
Facts added by a mention, and the last two kinds, return at most `limit` facts after filtering, with `complete=false` when there were more; `retrieval` says which
mode ran. The subgraph / `moment_ids` context comes from the same entity match.
Temporal-only reads are served by the `(timeline_id, s_lo)` index (p95 under the 500 ms target on 10k facts: 80–230 ms measured on a
loaded host). `TemporalFact` also has `(timeline_id, e_hi)` and `(entity_id, s_lo)` (TMP-8). ArcadeDB 26.10 caveat: a NOTUNIQUE
index range scan with an inclusive lower bound (`>=`, `BETWEEN`) drops rows that share a key, so `s_lo`/`e_hi` lower bounds
are always written `> bound - 1`.

**TKG time window.** The candidate facts are filtered by the window.
`time_start` is a lower bound on `fact.time_start` and `time_end` an upper bound on `fact.time_end` — the fact must lie *inside* the window
(a fact that only overlaps it is not returned). Dates are compared as instants, not strings (TMP-3): each window bound and each fact endpoint
is parsed into a range of UTC epoch ms at the precision it was given — `2011` is all of 2011, `2011-05` all of May,
`2011-05-03T10:00:00+08:00` that instant — so the lower bound is the start of `time_start`'s range and the upper bound the end of
`time_end`'s. Parsing is the EN/ZH grammar of `app.temporal.normalize` (see [temporal.md](temporal.md)): ISO `YYYY`, `YYYY-MM`,
`YYYY-MM-DD` and date-times with or without a zone, plus forms such as `2011年`, `民國100年`, `2011年上半年`, `2011年底`, `Q3 2011`,
`late 2011`, `the summer of 2011`, `May 2011`, `2010-2012`, `between 2008 and 2013` (grammar 1.1.1, see temporal.md; `2010 年 6 月` with spaces as `2010年6月`); `2011/05/03`-style slashes are not accepted (quarantined);
calendar dates and zone-less times are read in `PD_TEMPORAL_DEFAULT_TIMEZONE` (default `Asia/Taipei`). An unparseable bound is a 400.
Facts not yet migrated by the backfill (no `s_state`) still compare their strings against the window's calendar dates. A NULL / unknown endpoint (open-ended fact, unknown start) imposes no constraint on its own side, but never lets a fact escape the
opposite bound: an open-ended fact that starts after `time_end`, or a fact without `time_start` that ended before `time_start`, is not returned.
A fact with both endpoints NULL always passes.

**TKG Allen match modes (P07).** With a window, every returned fact carries
`allen: {vs_window, certainty}`: the Allen relations (point-inclusive, `app.temporal.allen`) the fact can have to the
window given its endpoint ranges, and whether it certainly satisfies the request. `match` / `relations` select facts
by those relations instead of the containment above; leaving both unset keeps the containment semantics, so existing
callers see the same facts.

| `match` | Accepted relations | Use |
|---|---|---|
| unset | containment, as above | Today's behaviour; `allen` is judged against `inside` |
| `inside` | starts, during, finishes, equals | |
| `overlaps` | all but before / after: inclusive intersection, boundary points included | "What was Hugo Barra doing in 2010" (served 2008–2013) |
| `contains` | contains, started-by, finished-by, equals | |
| `any` | all 13 | |
| `relations: [...]` | any of the 13 names; overrides `match` | Strict "after 2011" is `relations: ["after"]` against 2011 |

`certainty: possible` (default) keeps a fact if *some* possible relation is accepted; `certainty: certain` only if *every* one is,
and `allen.certainty` says which applies. A year-only fact is `certain`ly inside its own year although four relations remain
possible; a fact with an unknown end that started inside the window is only `possible`ly inside it.
The SQL read is only a necessary prefilter — inclusive intersection, with an endpoint that isn't known passing, or no window
predicate at all when `before`/`after` are accepted — and the exact filter runs before `limit`. A row not normalised yet is
classified from its `time_*` strings; a row whose dates can't be read has `allen: null` and is kept only by `match: any`.
An open window side ("from 2012 on") lies beyond every instant, so an unknown end never passes it: a fact that started in
2013 with no known end is `certain`ly inside "from 2012 on".
The fields reach Module 25 from every caller: the gateway's `POST /api/0.4/search/tkg` and the MCP tool `raptor_tkg_query`
forward them, and Module 18's `/classify` plan carries `match` (`overlaps` for a window closed on both sides — "in 2010",
"2010-2016"; `inside` for one open side — "after 2011", 「2011年以後」, TMP-2), which Module 21's TKG agent passes on. `allen` comes back unchanged.

**Backfilling dates on existing facts (TMP-3).** Facts written before TMP-3 are migrated in place by
`python -m app.services.temporal_backfill --all` (or `--branch <sub>`, `--dry-run` to count) inside the container. Raw values are kept in
`*_raw`; an unparseable value makes its endpoint `unknown` with a `quarantine_reason` (the row is kept). Every quarantined row gets
`requires_reextraction=true`, and `precision=unknown` when an endpoint could not be parsed. Each row is stamped with `migration_version`,
which is the checkpoint: an interrupted run resumes where it stopped, and a run over a migrated database changes nothing. It does not read
`PD_TEMPORAL_V2` and runs the same with the flag off. Back up ArcadeDB first.

**TemporalFact v2 (TMP-8).** The schema is additive and is applied when a database is created, on every `/internal/db/init` call
(existing databases included), and by the backfill; it does not depend on `PD_TEMPORAL_V2` (default `false`), because v1 reads need the canonical dates too. The backfill is the
migration: run it once per existing database after deploying (a new normaliser version makes it redo rows migrated earlier, from their
raw values). It writes only what a stored row can establish — the ranges, states, `precision`, `timezone`, `timeline_id`, `axis`,
`time_unit` — and leaves `extent`, `anchor`, the evidence fields, `confirmed_active_at`, `derived`, `support_ids`, `network_version` and
`media_*_ms` NULL. A NULL endpoint becomes `unknown`, never `ongoing`.

`time_start` / `time_end` stay for v1 readers, as projections of the ranges. They are **lossy**; read the ranges, states and `precision`
when the difference matters:

| Stored as written (`*_raw`) | `time_start` / `time_end` projection | What the projection loses |
|---|---|---|
| `2011` (start) | `2011-01-01` | That it is the whole year: the range is `s_lo`..`s_hi` = all of 2011, `precision=year` |
| `2013-06` (end) | `2013-06-30` | That it is the whole month; an end projects to the *last* day of its range, a start to the *first* |
| `2011-05-03T10:00:00+08:00` | `2011-05-03` | The time of day and the zone (the date is the one in `timezone`) |
| `Q3 2013`, `2011年上半年`, `2010-2012` | first / last day of the period | The period itself; `precision` is `quarter`, `half_year`, `year` |
| missing, `null`, or unparseable | NULL | Why: `s_state`/`e_state` is `unknown`, and an unparseable value has a `quarantine_reason` |

**Padded dates (TMP-8).** A writer that only knew "2012" or "August 2013" often sends full dates or midnight timestamps
(`2012-01-01T00:00:00Z` .. `2012-12-31T00:00:00Z`; the index API has no precision field). Two rules apply on write, in the
backfill, and to query window bounds:
- A timestamp at exactly 00:00 is read as that **whole calendar day** in its own offset (the interpreting timezone when it
  has none), never as one millisecond: `2012-12-31T00:00:00Z` as an end covers all of 31 December. A timestamp with a time
  of day stays a point.
- A **padded-looking period** — from the 1st of a month to the last day of a month, or a lone start on 1 January — cannot
  establish its own precision. Its range is kept as written; `precision` becomes `unknown`, `requires_reextraction=true`,
  and `reextraction_reason` says why (`quarantine_reason` stays empty: the row is usable, only doubtful). A lone 1st of a
  month is not marked, since real events on the 1st are too common.

Values that already carry their precision (`2012`, `2013-08`, `2012年`) are never marked. Writers should send those rather
than padded dates.

**Dates for answers (D8).** Each TKG fact also carries `when`: its dates at the precision they have, built from the ranges —
`2024-07 (no end date stated)`, `2008 to 2013`, `2010-06-01`, `2011-01..2011-06` (some time in those months), or the instants of a
timestamp — read in the fact's own `timezone`; a row not backfilled yet is read from its `time_*` strings. And `precision` as stored.
An answer should state dates from `when`, never from the padded `time_start` / `time_end` (a model given `2024-07-01` writes
"1 July"). Module 21's TKG agent does.

`max_depth` is clamped 1..5. The raw-query endpoint and the `graph` query-override are validated by a read-only guard (see [§8](#8-behaviors--caveats)).

### PA-8 — Kafka publish (`/personal/publish`, `publish.py`)

| Method | Path | Body |
|---|---|---|
| POST | `/personal/publish/index-request` | `PublishIndexRequest{branch_id (req), version_id?, chunks[], entities[], relationships[], temporal_facts[]}` |

Wraps the flat lists into the worker envelope and publishes to Kafka; returns `{status:"published", topic, branch_id, counts:{...}}`. A **test/demo producer** that simulates workers 09-12 so the whole ingest pipeline can be driven from Swagger (see [§6](#6-kafka-ingest-pipeline-pa-8)).

### People layer — face traces, Identity API, People Search (dev plan v1.4 §4.2–4.6)

Module 13 exposes these as `/api/{v}/identity/*` and `/api/{v}/search/people` (adding presigned `image_url` / `asset_url`); its Swagger has the user-facing descriptions. People Search there requires the caller's **`face_detection` consent** (off by default — biometric data): without it Module 13 answers 403 `PERMISSION_DENIED` and never asks Module 25; if consent can't be read, 503 `CONSENT_CHECK_UNAVAILABLE`. The seed-photo upload (`POST /api/{v}/identity/{id}/profile-image`) needs the same consent, and Module 11 detects faces only for users who granted it. Withdrawing it makes Module 33 call `DELETE /internal/gdpr/faces/user/{user_id}` here, which deletes the whole face layer -- every Face, FaceTrace, FaceGroup, Identity and IdentityHistory vertex -- and leaves the rest of the user database. Module 21's PeopleAgent uses the same two gateway routes, as the end user: it lists the user's people, keeps the ones the question names, and searches each. Errors: **404** unknown person/history or no user DB, **409** a change to a merged/deleted person, merging into itself, or undo/redo in the wrong state or past the 24-h deadline, **400** missing `X-User-ID`.

**Face write** (Module 11 → here, once per analysed video):

| Method | Path | Status | Body → Response |
|---|---|---|---|
| POST | `/personal/index/face-traces` | 201 | `FaceTracesRequest{version_id, asset_path?, filename?, traces[{trace_id, start_sec, end_sec, embedding(512), frames[], representative{…, image_path, image_version_id}}], groups[{group_id, embedding(512), trace_ids[]}]}` → `{version_id, traces_written, groups_written, replaced{…}, matches[{group_id, identity_id, created_identity, score, skipped_small_face}]}` |

Replaces any earlier write of the same `version_id` (a re-analysis never leaves stale traces). Each group is then **matched across videos** (1.4 §4.5 step 5): the best weighted similarity among other videos' groups (× `face_group_match_weight` 0.95) and the user's seed photos (× `face_seed_match_weight` 1.0), `face_match_candidates` (10) neighbours per index.
- below `face_match_threshold` (0.50), or a representative face smaller than `face_match_min_face_px` (60 px): the group stays **unassigned** — it appears on the video's pending list (`identity_id: null`);
- a hit on a group or seed photo that has an Identity: linked to it;
- a hit on another video's still-unassigned group: both become one new `pending` Identity (`source: auto`, `created_identity: true`) — someone seen in several videos becomes one person before anyone names them.

Chain assembly (tracking) and within-video clustering happen in Module 11, not here.

**Identity API** (`/personal/identity`, `identity.py`):

| Method | Path | Body | Response |
|---|---|---|---|
| GET | `/identities?limit&offset&status` | — | `IdentityList{total, items[IdentityOut]}`; merged/deleted only when asked for by `status` |
| POST | `/upload` | `{identity_id?, name, aliases?, identity_type?}` | `IdentityOut` — 201 created (`active`), 200 updated; with `identity_id` a retry never creates twice |
| GET | `/{identity_id}` | — | `IdentityOut{identity_id, name, aliases, identity_type, source, identity_status, origin_version_id, avatar, video_count, group_count, …}` — `avatar`: the seed photo if there is one; otherwise the best-quality frame of the person's largest group, re-chosen whenever its groups change (match, bind, unbind, merge, video deleted). Avatars picked before this rule: `python -m app.services.avatar_backfill --all --dry-run`, then without `--dry-run` (seed photos and merged/deleted people are left as they are) |
| PATCH | `/{identity_id}` | `{name?, aliases? (null value removes a locale), identity_status?: "active"}` | `IdentityOut` — `pending → active` is the only status change here |
| DELETE | `/{identity_id}` | — | `{identity_id, history_id, undo_deadline, groups_unassigned[]}` — groups go back to their videos' pending lists; traces, frames and seed photo kept; person marked `deleted` (out of listings and People Search, seed photo stops matching) |
| GET | `/file/{version_id}/pending-persons` | — | `{version_id, unassigned_groups[GroupOut], pending_identities[IdentityOut + groups in this video]}` |
| POST | `/file/{version_id}/pending-person` | `{name?, trace_id? \| trace_ids? \| group_id?}` (at most one) | 201 `{identity, bind?, binds[], bind_group?}` — a `pending` person (`source: manual`) started from this video, optionally binding one of its traces, a list of them (`binds`, one per trace), or a whole group (`bind_group`, as `bind/group`); every ID is checked first, so a bad one creates nothing |
| POST | `/{identity_id}/bind/trace` | `{version_id, trace_id}` | `BindResult{identity_id, version_id, trace_id, group_id, previous_group_id, removed_groups[], removed_identities[]}` — moves the trace into this person's group for that video (created if none); the group it left is recomputed or removed if empty |
| POST | `/{identity_id}/bind/group` | `{version_id, group_id}` | `BindGroupResult{identity_id, version_id, group_id, previous_identity_id, removed_identities[]}` — puts a whole group (every trace Module 11 clustered together) under this person in one call; the group itself is kept and only its `GROUP_OF_IDENTITY` edge moves, so a group another person held moves here; a system-created pending person left with no groups is removed; 404 if the group isn't in that video |
| POST | `/{identity_id}/unbind` | `{version_id, trace_id}` | `BindResult` — trace left in no group; 409 if it isn't this person's |
| GET | `/{identity_id}/traces` | — | `{identity_id, traces[TraceOut{trace_id, version_id, group_id, start_sec, end_sec, avg_confidence, representative, source_status}]}` — all videos, whatever their status |
| POST | `/{identity_id}/mergeinto` | `{target_id, keep_source=true}` | `MergeResult{merge_id, undo_deadline, source_id, source_kept, groups_moved[], target}` — groups and seed photos move to the target, aliases fill only missing locales, target's avatar is re-chosen (see `avatar` below); source kept as `merged` (or removed) |
| POST | `/merge/{merge_id}/undo` · `/merge/{merge_id}/redo` | — | `HistoryReplay{history_id, operation, history_state, undo_deadline, applied[], skipped[]}` — for merges and deletes (`merge_id` = `history_id`), within 24 h; items changed again since are `skipped` |
| POST | `/{identity_id}/profile-image` | `{image_path, image_version_id, embedding(512)}` | `IdentityOut` — seed photo (Module 13 stores it in Module 04 and gets the embedding from Module 11 first); becomes the avatar and is matched against later videos; a new one replaces the old |
| GET | `/{identity_id}/profile-image` | — | `FaceImage{face_id, kind, image_path, image_version_id}` — seed photo, else a representative frame |

**People Search** (`/personal/search`, `search.py`):

| Method | Path | Body | Response |
|---|---|---|---|
| POST | `/personal/search/people` | `{query (name or alias, whole or part), limit=50 (≤500)}` | `PeopleSearchResponse{query, identities[{identity_id, name, matched, exact}], total, results[PeopleSegment{identity_id, identity_name, version_id, trace_id, first_sec, last_sec, confidence, keyframe, keyframe_path, asset_path, filename}]}` — one segment per trace (`first_sec`–`last_sec` is the clip to play; `keyframe_path` = `keyframe.image_path`); merged/deleted people never match, and only videos whose `Source` is active are returned — unregistered, archived or not-yet-indexed videos are left out, the same rule as content search. A user with no database yet gets an empty result, not 404 |
| POST | `/personal/search/people/in-videos` | `PeopleInVideosRequest{identity_ids[], version_ids[]}` (both required, non-empty) | `PeopleInVideosResponse{results[PeopleSegment{...same shape as above}]}` |

Name matching: names go through the `Identity(name)` full-text index, then each hit is checked again in Python, because the index's bigram match is over-inclusive (`金門豬肉` must not find `金門`); a single CJK character, which the index can't match, is a substring scan. Aliases have no index and are always checked in Python. Case-insensitive; `exact` means the whole name or alias matched.

**`people/in-videos` vs. `people`:** the scoped counterpart to `people` above —
every trace for *these specific identities*, restricted to *these specific videos*,
with **no `limit`** at all: the result is already bounded by `len(identity_ids) ×
len(version_ids)`, not by how many times an identity appears platform-wide. Added
because Module 13's `video_search` route used to attach `people` to a content-matched
video via a plain `/personal/search/people` call sized to `top_k × candidate_multiplier`
— unrelated to how many segments a popular identity actually has, so it silently
truncated people data for arbitrary videos once an identity was popular enough. Module
13 now calls this endpoint instead, scoped to the union of RRF content-candidates and
name-matched videos. No consent/active-`Source` filtering happens here (unlike `people`
above) — the caller (Module 13) already has that context from its own query.

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
- **`tkg_search`** — `TemporalFact` by named entity, evidence keywords, or the time window alone (TMP-4), filtered by the window (UTC ms ranges, TMP-3) or by Allen relations to it (`match` / `relations`, P07), ordered by `time_start ASC`.
- **`graphrag_search`** — vector-rank chunks, then a **second query** backfills each chunk's mentioned entities (`out('MENTIONS')` does not resolve on the temporary records produced by `expand(vectorNeighbors(...))`, hence two queries).
- **`list_entities` / `get_entity`** — entity listing (by `mention_count DESC`) and detail with incoming/outgoing relations.

**Read-only SQL guard** (`is_read_only_select`): single statement, must start with `SELECT`, rejects `insert|update|delete|create|drop|alter|truncate|import|grant|revoke|rebuild|move|traverse into` and statement chaining.

---

## 6. Kafka ingest pipeline (PA-8)

Keeps each personal DB in sync with uploads automatically. Workers 09-12 publish the same index payload they send to the global DBs onto the `personal-index-requests` topic; this service's consumer routes each message to the right per-user DB.

### Consumer (`app/services/kafka_consumer.py`)

- Topic `personal-index-requests`, group `personal-db-service`, `auto_offset_reset=earliest`, **manual commit after successful handling** (no message lost on crash).
- **Dedup**: Redis key `personal:indexed:{branch_id}:{chunk_id}` with 7-day TTL, per tenant. A hit is trusted only while the tenant's `Chunk` row (or, for a summary, the `Source` summary) still exists -- the key outlives a delete-by-version, and a re-upload must not be skipped.
- **Auto-creates** the user DB + schema on first message for a new branch (consumer-side equivalent of `/internal/db/init`).
- **Embeds** content if the message carries no vector (same BGE-M3 fallback as PA-4).
- Routes graph-layer entries to the PA-5 indexers; a missing required field skips that entry (logged) instead of failing the whole message.
- **Capture / publish time (P13)**: a chunk entry may carry `capture_time` / `capture_time_source` (the file's own creation date, read by the worker); stored on `Chunk` and `Source` as read, validated by `app/temporal/anchor.py`, and used by the temporal job to date deictic phrases. See doc/temporal.md.
- **Temporal job (P12)**: for every source, any medium, records a durable temporal job (`temporal_jobs.enqueue()`) right after its chunks are indexed, before the offset commits; the job runs in the background worker (`TemporalJobRunner`). See doc/temporal.md, "Durable temporal jobs (P12)".
- **Video graph extraction** (entities/relations only): after a video's chunks are indexed, spawns `graph_extractor.extract_and_index_video()` as a background `asyncio` task (bounded by `PD_GRAPH_EXTRACTION_MAX_CONCURRENCY`, default 1) rather than awaiting it inline — `personal-index-requests` is one topic shared by every media type and every user, and the LLM calls this makes (90s timeout each) would otherwise block every other queued message behind a slow video. Only anchors MENTIONS/OBSERVED_IN edges to `chunk_id`s actually written as `Chunk` rows in the same message. On shutdown, in-flight extractions are cancelled (`drain_graph_extraction_tasks()`), not waited out — graph data here is best-effort, not core search.

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
| `PD_LLM_BASE_URL` / `PD_CHAT_MODEL_NAME` | `http://host.docker.internal:11434/v1` / `gemma4:12b` | Ollama endpoint for personal-graph entity extraction (video only); same values as Module 20's `LLM_BASE_URL`/`CHAT_MODEL_NAME` if sharing one Ollama instance |
| `PD_GRAPH_EXTRACTION_ENABLED` | `True` | `False` disables video graph extraction entirely |
| `PD_TEMPORAL_JOBS_ENABLED` | `True` | P12 durable temporal job per indexed source (every medium) |
| `PD_TEMPORAL_LLM_ENABLED` | `True` | `False`: rules-only temporal extraction, no Module 07 calls |
| `PD_TEMPORAL_LLM_CONCURRENCY` | `1` | concurrent Module 07 calls for temporal extraction |
| `PD_TEMPORAL_JOB_WORKERS` | `2` | temporal jobs run side by side; their model calls still take turns |
| `PD_TEMPORAL_LLM_TIMEOUT` / `PD_TEMPORAL_LLM_RETRIES` | `180` / `3` | per call; retried on connection errors, 502/503/504 and unreadable JSON |
| `PD_TEMPORAL_SEGMENT_MAX_ATTEMPTS` | `3` | job runs after which a failing segment stays failed |
| `PD_GRAPH_EXTRACTION_MAX_CONCURRENCY` | `1` | max videos extracting concurrently — runs as background tasks so a burst of uploads can't flood the LLM endpoint |

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
| `test_tkg_allen_match.py` (unit + integration) | P07 match modes, certainty, strict after, one-sided windows, legacy rows; the intersection prefilter checked exhaustively never to drop an accepted fact |
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
