"""Personal DB Service configuration (pydantic-settings, env prefix PD_)."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PD_", case_sensitive=False)

    # ArcadeDB (Module 24) connection
    arcadedb_url: str = "http://raptor-arcadedb:2480"
    arcadedb_user: str = "root"
    # No default: startup fails loudly rather than retrying against a wrong password.
    arcadedb_password: str
    http_timeout: float = 60.0

    # Temporal dates (TMP-3): the zone in which calendar dates ("2011", "2011-05-03")
    # and zone-less times are read before conversion to UTC ms. Same zone as
    # Module 04's TIMEZONE default.
    temporal_default_timezone: str = "Asia/Taipei"
    # TemporalFact v2 (TMP-8, temporal plan §5.8 / §8), per deployment. The v2 schema and
    # the backfill (temporal_backfill.py) do not depend on it: both apply with it off,
    # and the v1 time_start / time_end projections are always written.
    temporal_v2: bool = False
    # P12 (temporal plan §5.7): durable per-source temporal jobs. Every indexed source -- any
    # medium -- gets one job; it runs the rules + LLM extractor (app/temporal/extract.py) segment
    # by segment, storing each segment's result, so a restart resumes where it stopped.
    temporal_jobs_enabled: bool = True
    temporal_llm_enabled: bool = True          # off: rules only, no Module 07 calls
    # Module 07 calls are serialised by default: 4 concurrent calls made the gateway drop
    # connections in testing (§5.6 "Throttling").
    temporal_llm_concurrency: int = 1
    # Jobs run side by side (each still waits its turn for Module 07), so a short document is not queued
    # behind a long video's whole extraction.
    temporal_job_workers: int = 2
    temporal_llm_timeout: float = 180.0
    temporal_expiry_sweep_seconds: float = 60.0  # P21: how often expired inline sources are archived (0: never)
    temporal_llm_retries: int = 3              # extra attempts on connection errors / 502 / 503 / 504 / bad JSON
    temporal_llm_retry_delay: float = 5.0      # seconds, doubled per attempt
    temporal_segment_max_attempts: int = 3     # a segment failing this many job runs stays failed

    # Per-user schema vector index settings
    vector_dim: int = 1024
    vector_similarity: str = "COSINE"

    # People / identity (dev plan v1.4 §4.2, §4.5). Face embeddings are ArcFace
    # (512-dim) -- a separate index dimension from the 1024-dim text embeddings above.
    face_vector_dim: int = 512
    # Cross-video matching: a group is linked to an existing Identity when its best
    # weighted similarity reaches this; otherwise it gets its own pending Identity.
    # 1.4 starts at 0.45 and calls it a starting point to tune; defaulting higher
    # because a wrong cross-video link pollutes the people list across every video,
    # while a missed one only shows up as a duplicate pending person to merge.
    # The value depends on the recognition model Module 11 runs: 0.50 is for AuraFace
    # with mirror-averaged embeddings (it was 0.55 for InsightFace buffalo_l).
    face_match_threshold: float = 0.50
    face_group_match_weight: float = 0.95   # hit on another video's FaceGroup
    face_seed_match_weight: float = 1.0     # hit on a user-uploaded seed photo
    face_match_candidates: int = 10         # nearest neighbours fetched per index
    # Groups whose best face is smaller than this are not auto-linked to anyone:
    # small-face embeddings are unreliable and were the main source of wrong merges
    # in testing. They still get their own pending Identity for the user to resolve.
    face_match_min_face_px: int = 60

    # Kafka (Module 05)
    kafka_enabled: bool = True                 # start the consumer in the app lifespan
    kafka_bootstrap: str = "kafka:9092"
    kafka_topic: str = "personal-index-requests"
    kafka_group_id: str = "personal-db-service"
    kafka_dlq_topic: str = "personal-index-requests-dlq"
    # Published once _handle_message() actually finishes a real indexing pass --
    # 09-12's orchestrators wait on this instead of flipping status the instant
    # they've merely enqueued the request (see VIE01-255 update_status() race).
    kafka_topic_completed: str = "personal-index-completed"
    kafka_max_attempts: int = 3                # then park in the DLQ and move on

    # CM-5 / VIE01-408: Module 23's flag-lifecycle events (created + every status
    # transition) -- consumed to keep Chunk/Source.review_status in sync so search
    # can exclude flagged content without calling Module 23 per result.
    moderation_flag_consumer_enabled: bool = True
    moderation_flag_topic: str = "moderation.flagged"
    moderation_flag_group_id: str = "personal-db-service-moderation"

    # Redis (Module 02) — version_id deduplication
    redis_url: str = "redis://raptor-redis:6379"
    redis_dedup_ttl: int = 604800  # 7 days

    # PostgreSQL (Module 03, database `personal_db`) — deletion audit trail
    postgres_dsn: str = "postgresql://raptor:raptor@raptor-postgres:5432/personal_db"
    # Deletion is irreversible. When the audit cannot be written the delete is
    # refused (503) rather than performed unrecorded — an audit trail that can be
    # skipped proves nothing. Set to False only for local runs without Module 03.
    audit_required: bool = True

    # Local embedding (sentence-transformers, no Module 07 dependency)
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "auto"  # 'cpu', 'cuda', or 'auto'

    # Local cross-encoder reranker (sentence-transformers, no Module 17 dependency —
    # hybrid_search must keep working if module 17 is ever retired). Same model and
    # sigmoid temperature scaling as module 17's own reranker, for ranking parity.
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "auto"  # 'cpu', 'cuda', or 'auto'
    reranker_temperature: float = 0.25
    reranker_max_length: int = 512  # caps tokenized length; one long outlier otherwise pads the whole batch
    rerank_depth: int = 15  # widen the fusion pool to this before rerank, then truncate to top_k

    # LLM-based entity/relationship/temporal-fact extraction for the personal
    # graph (video only for now). Same Ollama chat-completions contract as
    # Module 20's graph-service (app/graph_builder.py + tkg.py) -- deliberately
    # NOT Module 12's document_graph_service, which posts through the module 07
    # inference gateway with a task name inconsistent with every other working
    # call site, and calls a TEMPORAL_MODEL_URL service that doesn't exist
    # anywhere in the repo.
    graph_extraction_enabled: bool = True
    llm_base_url: str = "http://host.docker.internal:11434/v1"
    chat_model_name: str = "gemma4:12b"
    # Extraction calls go through module 07 (think=false by default there),
    # not llm_base_url's OpenAI-compat endpoint directly -- that layer has no
    # think control, and gemma4-family models "think" by default, measured
    # live at 10-20x the latency for an identical final answer (4.8s vs 59s
    # on the same extraction prompt). llm_base_url/chat_model_name kept for
    # the commented-out fallback path in graph_extractor.py's _call_llm().
    inference_url: str = "http://raptor-ai-lifecycle-api:8010"
    inference_think: bool = False  # gemma4-family "thinking" mode for _call_llm_raw's calls via module 07
    # Extraction runs as a background task, not awaited inline by the Kafka
    # consumer loop (see kafka_consumer.py's _run_graph_extraction) -- this
    # bounds how many videos' worth of LLM calls run at once, so a burst of
    # uploads can't slam the LLM endpoint with unbounded concurrent requests.
    graph_extraction_max_concurrency: int = 1  # was 2; see root .env.example for why
    # Bounds concurrent LLM calls WITHIN a single video's extraction (moment
    # batches in extract_moment_entities_batched(), and the per-moment
    # temporal-fact candidates) -- separate axis from
    # graph_extraction_max_concurrency above, which bounds how many *videos*
    # extract at once. Previously both loops were a plain sequential
    # `for ... await ...` (ported verbatim from Module 20, which is the same
    # way). Bounded concurrency here means one video finishes faster, which
    # shrinks the wall-clock window during which its LLM calls could overlap
    # with the next video's -- unbounded (asyncio.gather with no semaphore)
    # would fire a large video's ~85 batches all at once, which is exactly
    # the kind of burst this is trying to avoid, not help with.
    graph_extraction_batch_concurrency: int = 2

    # GD-1 / VIE01-246: shared secret Module 33 (the GDPR erasure
    # orchestrator) presents as X-Internal-Token on DELETE /internal/gdpr/
    # user/{user_id}. Empty by default so that endpoint 401s until an
    # operator deliberately sets it, rather than silently accepting
    # requests with no token at all.
    gdpr_internal_token: str = ""


settings = Settings()
