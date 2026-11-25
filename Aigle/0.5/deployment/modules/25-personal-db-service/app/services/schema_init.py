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

from app.core.config import settings
from app.services.arcadedb_client import ArcadeDBClient

logger = logging.getLogger("personal_db.schema")


def _ddl(dim: int, sim: str) -> list[str]:
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
        "CREATE PROPERTY Chunk.capture_time IF NOT EXISTS STRING",         # P13: file metadata, as read
        "CREATE PROPERTY Chunk.capture_time_source IF NOT EXISTS STRING",
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
        # -- GD-6 / VIE01-251: true when Module 12's ingestion scan found PII
        # in this chunk (Kafka pii_detected); absent otherwise. Surfaced as
        # pii_warning in search results and counted in the DSAR. --
        "CREATE PROPERTY Chunk.pii_flag IF NOT EXISTS BOOLEAN",
        # CM-3 / VIE01-406: a video moment's keyframe moderation verdict (Module 11
        # -> Module 23 image guard); the video's rollup is Source.moderation_flags.
        "CREATE PROPERTY Chunk.moderation_flags IF NOT EXISTS MAP",
        # CM-5 / VIE01-408: the *current* admin-review status (flagged | cleared |
        # removed) of this chunk's row(s) in Module 23's moderation_flags
        # table; null = never flagged. Deliberately separate from moderation_flags
        # above: that property is a frozen ingestion-time verdict kafka_consumer.py
        # overwrites wholesale on every re-index, while this one is written only by
        # this module's own moderation_flag_consumer.py reacting to Module 23's
        # moderation.flagged Kafka events, so a re-index can never clobber an
        # admin's review decision. Search excludes 'flagged' and 'removed' by
        # default (see searcher.py's _filters()/_source_filters()).
        "CREATE PROPERTY Chunk.review_status IF NOT EXISTS STRING",

        # ================= vertex: Source (per uploaded asset) =================
        "CREATE VERTEX TYPE Source IF NOT EXISTS",
        "CREATE PROPERTY Source.generation IF NOT EXISTS INTEGER",         # P14: advanced by archive / reactivation
        "CREATE PROPERTY Source.capture_time IF NOT EXISTS STRING",        # P13: file metadata, as read
        "CREATE PROPERTY Source.capture_time_source IF NOT EXISTS STRING",
        "CREATE PROPERTY Source.version_id IF NOT EXISTS STRING",
        "CREATE PROPERTY Source.filename IF NOT EXISTS STRING",
        "CREATE PROPERTY Source.asset_path IF NOT EXISTS STRING",
        "CREATE PROPERTY Source.media_type IF NOT EXISTS STRING",
        "CREATE PROPERTY Source.title IF NOT EXISTS STRING",
        # CM-2: Module 10's image moderation verdict (Module 23 image guard), as sent
        # with the image's summary entry; absent when the asset wasn't moderated.
        "CREATE PROPERTY Source.moderation_flags IF NOT EXISTS MAP",
        # CM-5 / VIE01-408: same meaning as Chunk.review_status above, for assets
        # flagged at the Source (whole-asset) level.
        "CREATE PROPERTY Source.review_status IF NOT EXISTS STRING",
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
        # TMP-3: canonical dates (temporal_normalize.py). Each endpoint is a range of
        # UTC epoch ms -- "2011" is all of 2011 -- compared as numbers, never as the
        # raw strings. time_start / time_end above become YYYY-MM-DD projections for
        # v1 readers; the values as written are kept in *_raw.
        "CREATE PROPERTY TemporalFact.time_start_raw IF NOT EXISTS STRING",
        "CREATE PROPERTY TemporalFact.time_end_raw IF NOT EXISTS STRING",
        "CREATE PROPERTY TemporalFact.timeline_id IF NOT EXISTS STRING",     # TMP-4: 'world' (calendar time)
        "CREATE PROPERTY TemporalFact.s_state IF NOT EXISTS STRING",         # known | unknown
        "CREATE PROPERTY TemporalFact.e_state IF NOT EXISTS STRING",
        "CREATE PROPERTY TemporalFact.s_lo IF NOT EXISTS LONG",
        "CREATE PROPERTY TemporalFact.s_hi IF NOT EXISTS LONG",
        "CREATE PROPERTY TemporalFact.e_lo IF NOT EXISTS LONG",
        "CREATE PROPERTY TemporalFact.e_hi IF NOT EXISTS LONG",
        "CREATE PROPERTY TemporalFact.precision IF NOT EXISTS STRING",
        "CREATE PROPERTY TemporalFact.timezone IF NOT EXISTS STRING",
        "CREATE PROPERTY TemporalFact.normalizer_version IF NOT EXISTS STRING",
        "CREATE PROPERTY TemporalFact.migration_version IF NOT EXISTS STRING",  # backfill skips rows at the target
        "CREATE PROPERTY TemporalFact.quarantine_reason IF NOT EXISTS STRING",  # set when a date could not be used
        "CREATE PROPERTY TemporalFact.requires_reextraction IF NOT EXISTS BOOLEAN",  # true for every quarantined row (P03)
        "CREATE PROPERTY TemporalFact.reextraction_reason IF NOT EXISTS STRING",  # why a parsed row is doubtful (padded dates)
        # TMP-8: the rest of TemporalFact v2 (temporal plan §5.8). Additive: the properties
        # above keep their meaning. axis / time_unit are derived from timeline_id by the
        # normaliser; the others stay NULL until a writer that can establish them supplies
        # them -- the migration never guesses them for a legacy row.
        "CREATE PROPERTY TemporalFact.axis IF NOT EXISTS STRING",            # world | media
        "CREATE PROPERTY TemporalFact.time_unit IF NOT EXISTS STRING",       # unit of s_lo..e_hi: ms
        "CREATE PROPERTY TemporalFact.extent IF NOT EXISTS STRING",          # point | span
        "CREATE PROPERTY TemporalFact.anchor IF NOT EXISTS STRING",          # how a relative date was resolved
        "CREATE PROPERTY TemporalFact.date_text IF NOT EXISTS STRING",       # the date as the source wrote it
        "CREATE PROPERTY TemporalFact.source_sentence IF NOT EXISTS STRING",
        "CREATE PROPERTY TemporalFact.evidence_offsets IF NOT EXISTS STRING",
        "CREATE PROPERTY TemporalFact.confirmed_active_at IF NOT EXISTS LONG",   # ms; only when e_state = ongoing
        "CREATE PROPERTY TemporalFact.derived IF NOT EXISTS BOOLEAN",        # inferred by reasoning, not stated
        "CREATE PROPERTY TemporalFact.support_ids IF NOT EXISTS LIST",       # fact_ids a derived fact rests on
        "CREATE PROPERTY TemporalFact.network_version IF NOT EXISTS STRING",
        "CREATE PROPERTY TemporalFact.media_start_ms IF NOT EXISTS LONG",    # where in a recording a world-time
        "CREATE PROPERTY TemporalFact.media_end_ms IF NOT EXISTS LONG",      # claim was said
        "CREATE PROPERTY TemporalFact.extractor_version IF NOT EXISTS STRING",   # P12: which extractor wrote it
        "CREATE PROPERTY TemporalFact.links IF NOT EXISTS STRING",       # P15: JSON qualitative links, for reasoning

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
        ("CREATE INDEX IF NOT EXISTS ON Chunk (text) FULL_TEXT "
        "METADATA {\"analyzer\": \"org.apache.lucene.analysis.cjk.CJKAnalyzer\"}"),
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
        ("CREATE INDEX IF NOT EXISTS ON Chunk (asr_text) FULL_TEXT "
        "METADATA {\"analyzer\": \"org.apache.lucene.analysis.cjk.CJKAnalyzer\"}"),
        ("CREATE INDEX IF NOT EXISTS ON Chunk (lvlm_desc) FULL_TEXT "
        "METADATA {\"analyzer\": \"org.apache.lucene.analysis.cjk.CJKAnalyzer\"}"),
        ("CREATE INDEX IF NOT EXISTS ON Chunk (contextual_text) FULL_TEXT "
        "METADATA {\"analyzer\": \"org.apache.lucene.analysis.cjk.CJKAnalyzer\"}"),
        ("CREATE INDEX IF NOT EXISTS ON Entity (name) FULL_TEXT "
        "METADATA {\"analyzer\": \"org.apache.lucene.analysis.cjk.CJKAnalyzer\"}"),
        ("CREATE INDEX IF NOT EXISTS ON Entity (description) FULL_TEXT "
        "METADATA {\"analyzer\": \"org.apache.lucene.analysis.cjk.CJKAnalyzer\"}"),
        # Source(summary) is a brand-new index (not an existing production
        # feature being reconfigured, unlike Chunk(text)/Chunk(summary) above)
        # so it goes straight to CJKAnalyzer, no StandardAnalyzer legacy to
        # preserve.
        ("CREATE INDEX IF NOT EXISTS ON Source (summary) FULL_TEXT "
        "METADATA {\"analyzer\": \"org.apache.lucene.analysis.cjk.CJKAnalyzer\"}"),

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
        "CREATE INDEX IF NOT EXISTS ON Chunk (review_status) NOTUNIQUE",
        "CREATE INDEX IF NOT EXISTS ON Source (review_status) NOTUNIQUE",
        "CREATE INDEX IF NOT EXISTS ON Entity (type) NOTUNIQUE",

        # ================= unique keys (idempotent upsert) =================
        "CREATE INDEX IF NOT EXISTS ON Chunk (chunk_id) UNIQUE",
        "CREATE INDEX IF NOT EXISTS ON Source (version_id) UNIQUE",
        "CREATE INDEX IF NOT EXISTS ON Entity (entity_id) UNIQUE",
        "CREATE INDEX IF NOT EXISTS ON TemporalFact (fact_id) UNIQUE",
        # TMP-4: temporal-only retrieval scans one timeline by start. ArcadeDB 26.10 bug: a
        # NOTUNIQUE index range scan with an INCLUSIVE lower bound (>=, BETWEEN) drops rows
        # when keys repeat -- and every fact of one year shares its s_lo. Exclusive (>),
        # <=, <, = are correct. So s_lo / e_hi lower bounds are always written as
        # "> bound - 1" (searcher._ENDPOINT_CHECK); test_index_range_scan_bug guards it.
        "CREATE INDEX IF NOT EXISTS ON TemporalFact (timeline_id, s_lo) NOTUNIQUE",
        # TMP-8: the other two v2 indexes (§5.8). The same lower-bound rule applies to both.
        "CREATE INDEX IF NOT EXISTS ON TemporalFact (timeline_id, e_hi) NOTUNIQUE",
        "CREATE INDEX IF NOT EXISTS ON TemporalFact (entity_id, s_lo) NOTUNIQUE",
        # temporal-only retrieval reads facts with an unknown start by equality on s_state
        "CREATE INDEX IF NOT EXISTS ON TemporalFact (s_state) NOTUNIQUE",
        # migration_version is only ever compared with <> (backfill), never by range.
        "CREATE INDEX IF NOT EXISTS ON TemporalFact (migration_version) NOTUNIQUE",
    ]


def _job_ddl() -> list[str]:
    """Durable temporal jobs (P12, temporal plan §5.7). Document types, not vertices: job state is
    bookkeeping, never part of the content graph. One TemporalJob per (source version, extractor
    version, normaliser version); one TemporalJobSegment per chunk, holding its own result so a
    restarted job only redoes the segments that are not done."""
    return [
        "CREATE DOCUMENT TYPE TemporalJob IF NOT EXISTS",
        "CREATE PROPERTY TemporalJob.job_key IF NOT EXISTS STRING",
        "CREATE PROPERTY TemporalJob.branch_id IF NOT EXISTS STRING",      # recovery after a restart needs the tenant
        "CREATE PROPERTY TemporalJob.source_version_id IF NOT EXISTS STRING",
        "CREATE PROPERTY TemporalJob.media_type IF NOT EXISTS STRING",
        "CREATE PROPERTY TemporalJob.extractor_version IF NOT EXISTS STRING",
        "CREATE PROPERTY TemporalJob.normalizer_version IF NOT EXISTS STRING",
        "CREATE PROPERTY TemporalJob.model IF NOT EXISTS STRING",          # null: rules only
        "CREATE PROPERTY TemporalJob.status IF NOT EXISTS STRING",         # pending|running|complete|partial|failed|cancelled
        "CREATE PROPERTY TemporalJob.reason IF NOT EXISTS STRING",
        "CREATE PROPERTY TemporalJob.segments_total IF NOT EXISTS INTEGER",
        "CREATE PROPERTY TemporalJob.segments_done IF NOT EXISTS INTEGER",
        "CREATE PROPERTY TemporalJob.segments_failed IF NOT EXISTS INTEGER",
        "CREATE PROPERTY TemporalJob.segments_unsupported IF NOT EXISTS INTEGER",
        "CREATE PROPERTY TemporalJob.facts IF NOT EXISTS INTEGER",
        "CREATE PROPERTY TemporalJob.unresolved IF NOT EXISTS INTEGER",
        "CREATE PROPERTY TemporalJob.anchor_time IF NOT EXISTS STRING",       # P13: validated capture / publish instant
        "CREATE PROPERTY TemporalJob.anchor_source IF NOT EXISTS STRING",     # where it came from (anchor.SOURCES)
        "CREATE PROPERTY TemporalJob.anchor_reason IF NOT EXISTS STRING",     # why a supplied one was not used
        "CREATE PROPERTY TemporalJob.generation IF NOT EXISTS INTEGER",      # P14: the source's generation at start
        "CREATE PROPERTY TemporalJob.created_at IF NOT EXISTS LONG",
        "CREATE PROPERTY TemporalJob.updated_at IF NOT EXISTS LONG",
        "CREATE INDEX IF NOT EXISTS ON TemporalJob (job_key) UNIQUE",
        "CREATE INDEX IF NOT EXISTS ON TemporalJob (source_version_id) NOTUNIQUE",
        "CREATE INDEX IF NOT EXISTS ON TemporalJob (status) NOTUNIQUE",

        "CREATE DOCUMENT TYPE TemporalJobSegment IF NOT EXISTS",
        "CREATE PROPERTY TemporalJobSegment.segment_key IF NOT EXISTS STRING",   # job_key + chunk_id
        "CREATE PROPERTY TemporalJobSegment.job_key IF NOT EXISTS STRING",
        "CREATE PROPERTY TemporalJobSegment.chunk_id IF NOT EXISTS STRING",
        "CREATE PROPERTY TemporalJobSegment.seq IF NOT EXISTS INTEGER",          # document order
        "CREATE PROPERTY TemporalJobSegment.text IF NOT EXISTS STRING",          # the evidence part of the Chunk's text
        "CREATE PROPERTY TemporalJobSegment.text_offset IF NOT EXISTS INTEGER",  # where it starts in Chunk.text
        "CREATE PROPERTY TemporalJobSegment.heading IF NOT EXISTS STRING",
        "CREATE PROPERTY TemporalJobSegment.page IF NOT EXISTS STRING",
        "CREATE PROPERTY TemporalJobSegment.media_start_ms IF NOT EXISTS LONG",
        "CREATE PROPERTY TemporalJobSegment.media_end_ms IF NOT EXISTS LONG",
        "CREATE PROPERTY TemporalJobSegment.status IF NOT EXISTS STRING",        # pending|done|failed|unsupported
        "CREATE PROPERTY TemporalJobSegment.reason IF NOT EXISTS STRING",
        "CREATE PROPERTY TemporalJobSegment.reply IF NOT EXISTS STRING",         # the model's JSON reply, as received
        "CREATE PROPERTY TemporalJobSegment.attempts IF NOT EXISTS INTEGER",
        "CREATE INDEX IF NOT EXISTS ON TemporalJobSegment (segment_key) UNIQUE",
        "CREATE INDEX IF NOT EXISTS ON TemporalJobSegment (job_key) NOTUNIQUE",

        # Unresolved candidates (§5.6): a supported assertion the extractor found but could not date ("later",
        # "at the time", "after the acquisition"), kept with its evidence and qualitative links for reasoning
        # (P15). A document type of its own, so no TKG or temporal query ever returns an undated row as a fact.
        # P15: the stored ingestion-time reasoning result of one source (§5.5): status, network version,
        # conflict witness, links used / unresolved. Derived facts are TemporalFact rows (derived = true).
        "CREATE DOCUMENT TYPE TemporalReasoning IF NOT EXISTS",
        "CREATE PROPERTY TemporalReasoning.source_version_id IF NOT EXISTS STRING",
        "CREATE PROPERTY TemporalReasoning.status IF NOT EXISTS STRING",      # consistent | inconsistent | no_links | failed
        "CREATE PROPERTY TemporalReasoning.network_version IF NOT EXISTS STRING",
        "CREATE PROPERTY TemporalReasoning.derived IF NOT EXISTS INTEGER",
        "CREATE PROPERTY TemporalReasoning.witness IF NOT EXISTS STRING",     # JSON
        "CREATE PROPERTY TemporalReasoning.links_used IF NOT EXISTS STRING",  # JSON
        "CREATE PROPERTY TemporalReasoning.links_unresolved IF NOT EXISTS STRING",
        "CREATE PROPERTY TemporalReasoning.computed_at IF NOT EXISTS LONG",
        "CREATE INDEX IF NOT EXISTS ON TemporalReasoning (source_version_id) UNIQUE",

        "CREATE DOCUMENT TYPE TemporalCandidate IF NOT EXISTS",
        "CREATE PROPERTY TemporalCandidate.candidate_id IF NOT EXISTS STRING",
        "CREATE PROPERTY TemporalCandidate.source_version_id IF NOT EXISTS STRING",
        "CREATE PROPERTY TemporalCandidate.chunk_id IF NOT EXISTS STRING",
        "CREATE PROPERTY TemporalCandidate.entity IF NOT EXISTS STRING",
        "CREATE PROPERTY TemporalCandidate.relation IF NOT EXISTS STRING",
        "CREATE PROPERTY TemporalCandidate.value IF NOT EXISTS STRING",
        "CREATE PROPERTY TemporalCandidate.date_text IF NOT EXISTS STRING",       # "" when the text gives no time
        "CREATE PROPERTY TemporalCandidate.reason IF NOT EXISTS STRING",          # why it has no interval
        "CREATE PROPERTY TemporalCandidate.links IF NOT EXISTS STRING",           # JSON [{relation, other}]
        "CREATE PROPERTY TemporalCandidate.source_sentence IF NOT EXISTS STRING",
        "CREATE PROPERTY TemporalCandidate.evidence_offsets IF NOT EXISTS STRING",
        "CREATE PROPERTY TemporalCandidate.extent IF NOT EXISTS STRING",
        "CREATE PROPERTY TemporalCandidate.origin IF NOT EXISTS STRING",          # rules | llm | rules+llm
        "CREATE PROPERTY TemporalCandidate.confidence IF NOT EXISTS FLOAT",
        "CREATE PROPERTY TemporalCandidate.extractor_version IF NOT EXISTS STRING",
        "CREATE PROPERTY TemporalCandidate.status IF NOT EXISTS STRING",       # P14: follows its source's status
        "CREATE INDEX IF NOT EXISTS ON TemporalCandidate (candidate_id) UNIQUE",
        "CREATE INDEX IF NOT EXISTS ON TemporalCandidate (source_version_id) NOTUNIQUE",
    ]


def _face_ddl(dim: int, sim: str) -> list[str]:
    """People / identity layer (dev plan v1.4 §4.2, §4.4).

    Kept apart from the content graph on purpose:
      - no vertex type here has a `status` property -- graph_query.py's subgraph
        filter drops any path through a node whose `status` isn't null/active, so
        Identity's lifecycle lives in `identity_status` instead;
      - none of these edge types appear in graph_query._SUBGRAPH_EDGE_TYPES, so face
        data never surfaces in GraphRAG/TKG results. The only link into the content
        graph is TRACE_IN_SOURCE -> Source, the same node content indexing uses, so
        takedown/archive status (Source.status) needs no separate sync.
    """
    return [
        # ================= vertex: FaceTrace (one continuous appearance in one video) =========
        "CREATE VERTEX TYPE FaceTrace IF NOT EXISTS",
        "CREATE PROPERTY FaceTrace.trace_id IF NOT EXISTS STRING",
        "CREATE PROPERTY FaceTrace.version_id IF NOT EXISTS STRING",
        "CREATE PROPERTY FaceTrace.start_sec IF NOT EXISTS FLOAT",
        "CREATE PROPERTY FaceTrace.end_sec IF NOT EXISTS FLOAT",
        "CREATE PROPERTY FaceTrace.frame_count IF NOT EXISTS INTEGER",
        "CREATE PROPERTY FaceTrace.avg_confidence IF NOT EXISTS FLOAT",
        "CREATE PROPERTY FaceTrace.embedding IF NOT EXISTS ARRAY_OF_FLOATS",
        # per-frame detections [{frame_no, timestamp_sec, bbox, confidence}, ...] as JSON:
        # one property per trace rather than one node per frame (1.4 §4.2 scale decision)
        "CREATE PROPERTY FaceTrace.frames_json IF NOT EXISTS STRING",

        # ================= vertex: Face (a trace's representative frame, or a seed photo) ======
        "CREATE VERTEX TYPE Face IF NOT EXISTS",
        "CREATE PROPERTY Face.face_id IF NOT EXISTS STRING",
        "CREATE PROPERTY Face.kind IF NOT EXISTS STRING",              # representative | seed
        "CREATE PROPERTY Face.version_id IF NOT EXISTS STRING",        # representative only
        "CREATE PROPERTY Face.identity_id IF NOT EXISTS STRING",       # seed only: whose photo it is
        "CREATE PROPERTY Face.image_path IF NOT EXISTS STRING",
        "CREATE PROPERTY Face.image_version_id IF NOT EXISTS STRING",
        "CREATE PROPERTY Face.frame_no IF NOT EXISTS INTEGER",
        "CREATE PROPERTY Face.timestamp_sec IF NOT EXISTS FLOAT",
        "CREATE PROPERTY Face.bbox IF NOT EXISTS ARRAY_OF_FLOATS",
        "CREATE PROPERTY Face.quality IF NOT EXISTS FLOAT",
        "CREATE PROPERTY Face.min_side IF NOT EXISTS FLOAT",
        "CREATE PROPERTY Face.embedding IF NOT EXISTS ARRAY_OF_FLOATS",  # seed only

        # ================= vertex: FaceGroup (same person within one video) =====================
        "CREATE VERTEX TYPE FaceGroup IF NOT EXISTS",
        "CREATE PROPERTY FaceGroup.group_id IF NOT EXISTS STRING",
        "CREATE PROPERTY FaceGroup.version_id IF NOT EXISTS STRING",
        "CREATE PROPERTY FaceGroup.embedding IF NOT EXISTS ARRAY_OF_FLOATS",
        "CREATE PROPERTY FaceGroup.frame_count IF NOT EXISTS INTEGER",
        "CREATE PROPERTY FaceGroup.created_at IF NOT EXISTS STRING",

        # ================= vertex: Identity (same person across videos) ========================
        "CREATE VERTEX TYPE Identity IF NOT EXISTS",
        "CREATE PROPERTY Identity.identity_id IF NOT EXISTS STRING",
        "CREATE PROPERTY Identity.name IF NOT EXISTS STRING",
        "CREATE PROPERTY Identity.aliases_json IF NOT EXISTS STRING",
        "CREATE PROPERTY Identity.identity_type IF NOT EXISTS STRING",
        "CREATE PROPERTY Identity.source IF NOT EXISTS STRING",        # auto | manual | ...
        "CREATE PROPERTY Identity.identity_status IF NOT EXISTS STRING",  # pending | active | merged | deleted
        "CREATE PROPERTY Identity.created_at IF NOT EXISTS STRING",
        "CREATE PROPERTY Identity.updated_at IF NOT EXISTS STRING",
        # pending-person only: the video it was created from, so it shows on that video's
        # pending list before any trace is bound to it
        "CREATE PROPERTY Identity.origin_version_id IF NOT EXISTS STRING",

        # ================= vertex: IdentityHistory (merge/delete snapshots, 24h undo) ==========
        "CREATE VERTEX TYPE IdentityHistory IF NOT EXISTS",
        "CREATE PROPERTY IdentityHistory.history_id IF NOT EXISTS STRING",
        "CREATE PROPERTY IdentityHistory.operation IF NOT EXISTS STRING",
        "CREATE PROPERTY IdentityHistory.before_snapshot_json IF NOT EXISTS STRING",
        "CREATE PROPERTY IdentityHistory.after_snapshot_json IF NOT EXISTS STRING",
        "CREATE PROPERTY IdentityHistory.history_state IF NOT EXISTS STRING",
        "CREATE PROPERTY IdentityHistory.undo_deadline IF NOT EXISTS STRING",
        "CREATE PROPERTY IdentityHistory.created_at IF NOT EXISTS STRING",

        # ================= edges =================
        "CREATE EDGE TYPE TRACE_IN_SOURCE IF NOT EXISTS",      # FaceTrace → Source
        "CREATE EDGE TYPE HAS_KEY_FACE IF NOT EXISTS",         # FaceTrace → Face
        "CREATE EDGE TYPE TRACE_IN_GROUP IF NOT EXISTS",       # FaceTrace → FaceGroup
        "CREATE EDGE TYPE GROUP_OF_IDENTITY IF NOT EXISTS",    # FaceGroup → Identity {score, auto}
        "CREATE PROPERTY GROUP_OF_IDENTITY.score IF NOT EXISTS FLOAT",
        "CREATE PROPERTY GROUP_OF_IDENTITY.auto IF NOT EXISTS BOOLEAN",
        "CREATE PROPERTY GROUP_OF_IDENTITY.created_at IF NOT EXISTS STRING",
        "CREATE EDGE TYPE HAS_PROFILE_FACE IF NOT EXISTS",     # Identity → Face

        # ================= vector indexes (ArcFace) =================
        f"CREATE INDEX IF NOT EXISTS ON FaceTrace (embedding) LSM_VECTOR METADATA {{'dimensions':{dim},'similarity':'{sim}'}}",
        f"CREATE INDEX IF NOT EXISTS ON FaceGroup (embedding) LSM_VECTOR METADATA {{'dimensions':{dim},'similarity':'{sim}'}}",
        f"CREATE INDEX IF NOT EXISTS ON Face (embedding) LSM_VECTOR METADATA {{'dimensions':{dim},'similarity':'{sim}'}}",

        # ================= full-text / keyword =================
        ("CREATE INDEX IF NOT EXISTS ON Identity (name) FULL_TEXT "
        "METADATA {\"analyzer\": \"org.apache.lucene.analysis.cjk.CJKAnalyzer\"}"),
        "CREATE INDEX IF NOT EXISTS ON FaceTrace (version_id) NOTUNIQUE",
        "CREATE INDEX IF NOT EXISTS ON FaceGroup (version_id) NOTUNIQUE",
        "CREATE INDEX IF NOT EXISTS ON Face (version_id) NOTUNIQUE",
        "CREATE INDEX IF NOT EXISTS ON Identity (identity_status) NOTUNIQUE",
        "CREATE INDEX IF NOT EXISTS ON Identity (origin_version_id) NOTUNIQUE",

        # ================= unique keys (idempotent upsert) =================
        "CREATE INDEX IF NOT EXISTS ON FaceTrace (trace_id) UNIQUE",
        "CREATE INDEX IF NOT EXISTS ON Face (face_id) UNIQUE",
        "CREATE INDEX IF NOT EXISTS ON FaceGroup (group_id) UNIQUE",
        "CREATE INDEX IF NOT EXISTS ON Identity (identity_id) UNIQUE",
        "CREATE INDEX IF NOT EXISTS ON IdentityHistory (history_id) UNIQUE",
    ]


async def initialize_face_schema(client: ArcadeDBClient, database_name: str) -> None:
    """Apply only the people / identity layer. Idempotent. Databases created before
    this layer existed get it on their first face write (identity_store.py) rather
    than through a one-off migration of every user's database."""
    statements = _face_ddl(settings.face_vector_dim, settings.vector_similarity)
    for stmt in statements:
        await client.command(database_name, stmt, ignore_exists=True)
    logger.info(f"[schema] initialized {len(statements)} face DDL statements on {database_name}")


async def initialize_job_schema(client: ArcadeDBClient, database_name: str) -> None:
    """Apply only the temporal job types. Idempotent. Databases created before P12 get them on
    their first job (temporal_jobs.py), like the people layer on its first face write."""
    for stmt in _job_ddl():
        await client.command(database_name, stmt, ignore_exists=True)


async def initialize_schema(client: ArcadeDBClient, database_name: str) -> None:
    """Apply the full per-user unified-Chunk schema plus the people layer. Idempotent."""
    statements = _ddl(settings.vector_dim, settings.vector_similarity)
    for stmt in statements:
        await client.command(database_name, stmt, ignore_exists=True)
    logger.info(f"[schema] initialized {len(statements)} DDL statements on {database_name}")
    await initialize_face_schema(client, database_name)
    await initialize_job_schema(client, database_name)
