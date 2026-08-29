"""
Per-user ArcadeDB schema initialization (PA-3) — UNIFIED `Chunk` model (0.4).

Design: one `Chunk` vertex is the single searchable content unit for ALL four
media types (documents|videos|images|audios), discriminated by `type` +
`embedding_type`. A single Chunk carries a dense-vector index + full-text index
(= Qdrant + OpenSearch effect) AND graph edges (= Neo4j effect) — so ArcadeDB
replaces all three 0.3 stores without the Document/Moment duplication.

Grounded in the live 0.3 data (see repo `personal-db-chunk-schema.md`):
  - retrieval layer is flat (type + embedding_type, content in text|summary), DENSE-ONLY.
  - graph search traverses via EDGES, never via a source_id scalar → Entity has NO source_id.
  - sparse is OPTIONAL (nothing produces it yet; required would block all real ingest).

DDL verified against arcadedata/arcadedb:latest (26.6.1). Corrections vs plan:
LSM_VECTOR METADATA{dimensions,similarity}; ARRAY_OF_FLOATS; sparse = two props
(sparse_indices + sparse_weights) indexed together as LSM_SPARSE_VECTOR.
"""
from __future__ import annotations

import logging
from typing import List

from app.core.config import settings
from app.services.arcadedb_client import ArcadeDBClient

logger = logging.getLogger("personal_db.schema")


def _ddl(dim: int, sim: str) -> List[str]:
    return [
        # ================= vertex: Chunk (unified searchable content) =================
        "CREATE VERTEX TYPE Chunk IF NOT EXISTS",
        # -- common (all media) --
        "CREATE PROPERTY Chunk.chunk_id IF NOT EXISTS STRING",
        "CREATE PROPERTY Chunk.type IF NOT EXISTS STRING",            # documents|videos|images|audios
        "CREATE PROPERTY Chunk.embedding_type IF NOT EXISTS STRING",  # text | summary
        "CREATE PROPERTY Chunk.text IF NOT EXISTS STRING",
        "CREATE PROPERTY Chunk.summary IF NOT EXISTS STRING",
        "CREATE PROPERTY Chunk.filename IF NOT EXISTS STRING",
        "CREATE PROPERTY Chunk.source IF NOT EXISTS STRING",
        "CREATE PROPERTY Chunk.asset_path IF NOT EXISTS STRING",
        "CREATE PROPERTY Chunk.version_id IF NOT EXISTS STRING",
        "CREATE PROPERTY Chunk.upload_time IF NOT EXISTS STRING",
        "CREATE PROPERTY Chunk.status IF NOT EXISTS STRING",
        "CREATE PROPERTY Chunk.embedding IF NOT EXISTS ARRAY_OF_FLOATS",
        "CREATE PROPERTY Chunk.sparse_indices IF NOT EXISTS ARRAY_OF_INTEGERS",
        "CREATE PROPERTY Chunk.sparse_weights IF NOT EXISTS ARRAY_OF_FLOATS",
        # -- temporal (video / audio segments) --
        "CREATE PROPERTY Chunk.start_sec IF NOT EXISTS FLOAT",
        "CREATE PROPERTY Chunk.end_sec IF NOT EXISTS FLOAT",
        "CREATE PROPERTY Chunk.speaker IF NOT EXISTS STRING",
        # -- video-specific --
        "CREATE PROPERTY Chunk.contextual_text IF NOT EXISTS STRING",
        "CREATE PROPERTY Chunk.asr_text IF NOT EXISTS STRING",
        "CREATE PROPERTY Chunk.lvlm_desc IF NOT EXISTS STRING",
        "CREATE PROPERTY Chunk.ocr_text IF NOT EXISTS STRING",
        # -- audio-specific --
        "CREATE PROPERTY Chunk.audio_labels IF NOT EXISTS LIST",
        # -- chunk_index: unified sequence index within the source asset, one
        # property for every media type (video/audio events upstream call it
        # moment_index, document events call it chunk_index -- kafka_consumer.py
        # normalizes both into this single property; no separate "Moment" node
        # type exists here to justify a video-specific name) --
        "CREATE PROPERTY Chunk.chunk_index IF NOT EXISTS INTEGER",
        # -- document-specific --
        "CREATE PROPERTY Chunk.page_numbers IF NOT EXISTS LIST",
        "CREATE PROPERTY Chunk.section_heading IF NOT EXISTS STRING",
        "CREATE PROPERTY Chunk.element_types IF NOT EXISTS LIST",
        "CREATE PROPERTY Chunk.char_count IF NOT EXISTS INTEGER",

        # ================= vertex: Source (per uploaded asset) =================
        "CREATE VERTEX TYPE Source IF NOT EXISTS",
        "CREATE PROPERTY Source.version_id IF NOT EXISTS STRING",
        "CREATE PROPERTY Source.filename IF NOT EXISTS STRING",
        "CREATE PROPERTY Source.asset_path IF NOT EXISTS STRING",
        "CREATE PROPERTY Source.media_type IF NOT EXISTS STRING",
        "CREATE PROPERTY Source.title IF NOT EXISTS STRING",
        "CREATE PROPERTY Source.summary IF NOT EXISTS STRING",
        # Whole-asset summary embedding -- was previously a fake "summary Chunk"
        # (embedding_type="summary", no start_sec/end_sec) so summary content
        # stayed searchable via the Chunk-only hybrid/bm25/vector queries; now
        # lives directly on Source (matching Module 20's Source.summary,
        # neo4j_writer.py:61) with its own FULL_TEXT + vector index below, so
        # hybrid_search()/bm25_search()/vector_search() can query it directly
        # instead of faking it as a Chunk.
        "CREATE PROPERTY Source.embedding IF NOT EXISTS ARRAY_OF_FLOATS",
        "CREATE PROPERTY Source.chunk_count IF NOT EXISTS INTEGER",
        "CREATE PROPERTY Source.processed_at IF NOT EXISTS STRING",
        "CREATE PROPERTY Source.status IF NOT EXISTS STRING",

        # ================= vertex: Entity (cross-media, NO source_id) =================
        "CREATE VERTEX TYPE Entity IF NOT EXISTS",
        "CREATE PROPERTY Entity.entity_id IF NOT EXISTS STRING",
        "CREATE PROPERTY Entity.name IF NOT EXISTS STRING",
        "CREATE PROPERTY Entity.type IF NOT EXISTS STRING",
        "CREATE PROPERTY Entity.description IF NOT EXISTS STRING",
        "CREATE PROPERTY Entity.mention_count IF NOT EXISTS INTEGER",
        "CREATE PROPERTY Entity.created_at IF NOT EXISTS STRING",
        "CREATE PROPERTY Entity.updated_at IF NOT EXISTS STRING",

        # ================= vertex: TemporalFact =================
        "CREATE VERTEX TYPE TemporalFact IF NOT EXISTS",
        "CREATE PROPERTY TemporalFact.fact_id IF NOT EXISTS STRING",
        "CREATE PROPERTY TemporalFact.entity IF NOT EXISTS STRING",
        "CREATE PROPERTY TemporalFact.entity_id IF NOT EXISTS STRING",
        "CREATE PROPERTY TemporalFact.relation IF NOT EXISTS STRING",
        "CREATE PROPERTY TemporalFact.value IF NOT EXISTS STRING",
        "CREATE PROPERTY TemporalFact.time_start IF NOT EXISTS STRING",
        "CREATE PROPERTY TemporalFact.time_end IF NOT EXISTS STRING",
        "CREATE PROPERTY TemporalFact.confidence IF NOT EXISTS FLOAT",
        "CREATE PROPERTY TemporalFact.created_at IF NOT EXISTS STRING",
        "CREATE PROPERTY TemporalFact.source_version_id IF NOT EXISTS STRING",  # delete-by-source only
        # status: mirrors Chunk.status/Source.status -- set at creation and by
        # set_status_by_version(). Existing per-user databases created before
        # this property existed won't have it backfilled (same "no automatic
        # upgrade for already-created DBs" gap as CJKAnalyzer -- see
        # schema_init.py's own module docstring / the plan this was tracked
        # in); tkg_search()'s query treats a null status as active, not as
        # "exclude", to avoid hiding pre-migration facts entirely.
        "CREATE PROPERTY TemporalFact.status IF NOT EXISTS STRING",

        # ================= edges (all reference Chunk; media-agnostic) =================
        "CREATE EDGE TYPE HAS_CHUNK IF NOT EXISTS",           # Source → Chunk
        "CREATE EDGE TYPE MENTIONS IF NOT EXISTS",            # Chunk → Entity   {modality}
        "CREATE EDGE TYPE RELATION IF NOT EXISTS",            # Entity → Entity  {relation, confidence}
        "CREATE EDGE TYPE CO_OCCURS_WITH IF NOT EXISTS",      # Entity ↔ Entity
        "CREATE EDGE TYPE HAS_TEMPORAL_FACT IF NOT EXISTS",   # Entity → TemporalFact
        "CREATE EDGE TYPE OBSERVED_IN IF NOT EXISTS",         # TemporalFact → Chunk
        "CREATE EDGE TYPE MENTIONED_IN IF NOT EXISTS",        # Entity → Source (document-level,
                                                               # from summary-level extraction)

        # ================= full-text (BM25 / Lucene) =================
        # Chunk(text) powers the already-shipped BM25/hybrid search
        # (searcher.py). Was left on the default StandardAnalyzer deliberately
        # through the PA-7 graph/TKG/GraphRAG work (reconfiguring it would
        # change ranking for a feature already in production use, out of
        # scope then) -- moved to CJKAnalyzer now that Source(summary) is
        # CJK-analyzed too (the Source-summary migration), so the two indexes
        # merged into one bm25_search() result list score comparably instead
        # of one being bigram-tokenized and the other char-by-char. Confirmed
        # live this doesn't hurt English/Latin content: modern CJKAnalyzer
        # (3.1+) tokenizes non-CJK text the same whole-word, case-folded way
        # StandardAnalyzer does, and only bigrams actual CJK characters.
        "CREATE INDEX IF NOT EXISTS ON Chunk (text) FULL_TEXT "
        "METADATA {\"analyzer\": \"org.apache.lucene.analysis.cjk.CJKAnalyzer\"}",
        # Chunk(summary) is dead schema now (nothing writes or queries it any
        # more -- summary lives on Source), left on StandardAnalyzer since
        # reconfiguring it has no observable effect either way.
        "CREATE INDEX IF NOT EXISTS ON Chunk (summary) FULL_TEXT",
        # asr_text/lvlm_desc/contextual_text and both Entity fields are CJK-
        # analyzed (bigram tokenization) -- confirmed live that the default
        # StandardAnalyzer's per-character tokenization produces real false
        # positives (e.g. "以色列" matching unrelated text via the bare
        # character "色"). Nothing else in the codebase queries these five
        # indexes yet (only graph_query.py, added alongside this), so
        # reconfiguring them has no effect on any other feature.
        "CREATE INDEX IF NOT EXISTS ON Chunk (asr_text) FULL_TEXT "
        "METADATA {\"analyzer\": \"org.apache.lucene.analysis.cjk.CJKAnalyzer\"}",
        "CREATE INDEX IF NOT EXISTS ON Chunk (lvlm_desc) FULL_TEXT "
        "METADATA {\"analyzer\": \"org.apache.lucene.analysis.cjk.CJKAnalyzer\"}",
        "CREATE INDEX IF NOT EXISTS ON Chunk (contextual_text) FULL_TEXT "
        "METADATA {\"analyzer\": \"org.apache.lucene.analysis.cjk.CJKAnalyzer\"}",
        "CREATE INDEX IF NOT EXISTS ON Entity (name) FULL_TEXT "
        "METADATA {\"analyzer\": \"org.apache.lucene.analysis.cjk.CJKAnalyzer\"}",
        "CREATE INDEX IF NOT EXISTS ON Entity (description) FULL_TEXT "
        "METADATA {\"analyzer\": \"org.apache.lucene.analysis.cjk.CJKAnalyzer\"}",
        # Source(summary) is a brand-new index (not an existing production
        # feature being reconfigured, unlike Chunk(text)/Chunk(summary) above)
        # so it goes straight to CJKAnalyzer, no StandardAnalyzer legacy to
        # preserve.
        "CREATE INDEX IF NOT EXISTS ON Source (summary) FULL_TEXT "
        "METADATA {\"analyzer\": \"org.apache.lucene.analysis.cjk.CJKAnalyzer\"}",

        # ================= dense vector (HNSW) =================
        f"CREATE INDEX IF NOT EXISTS ON Chunk (embedding) LSM_VECTOR METADATA {{'dimensions':{dim},'similarity':'{sim}'}}",
        f"CREATE INDEX IF NOT EXISTS ON Source (embedding) LSM_VECTOR METADATA {{'dimensions':{dim},'similarity':'{sim}'}}",

        # ================= sparse vector (optional; kept for future hybrid) =================
        "CREATE INDEX IF NOT EXISTS ON Chunk (sparse_indices, sparse_weights) LSM_SPARSE_VECTOR",

        # ================= keyword (fast filter) =================
        "CREATE INDEX IF NOT EXISTS ON Chunk (type) NOTUNIQUE",
        "CREATE INDEX IF NOT EXISTS ON Chunk (status) NOTUNIQUE",
        "CREATE INDEX IF NOT EXISTS ON Chunk (version_id) NOTUNIQUE",
        "CREATE INDEX IF NOT EXISTS ON Chunk (embedding_type) NOTUNIQUE",
        "CREATE INDEX IF NOT EXISTS ON Entity (type) NOTUNIQUE",

        # ================= unique keys (idempotent upsert) =================
        "CREATE INDEX IF NOT EXISTS ON Chunk (chunk_id) UNIQUE",
        "CREATE INDEX IF NOT EXISTS ON Source (version_id) UNIQUE",
        "CREATE INDEX IF NOT EXISTS ON Entity (entity_id) UNIQUE",
        "CREATE INDEX IF NOT EXISTS ON TemporalFact (fact_id) UNIQUE",
    ]


async def initialize_schema(client: ArcadeDBClient, database_name: str) -> None:
    """Apply the full per-user unified-Chunk schema. Idempotent."""
    statements = _ddl(settings.vector_dim, settings.vector_similarity)
    for stmt in statements:
        await client.command(database_name, stmt, ignore_exists=True)
    logger.info(f"[schema] initialized {len(statements)} DDL statements on {database_name}")
