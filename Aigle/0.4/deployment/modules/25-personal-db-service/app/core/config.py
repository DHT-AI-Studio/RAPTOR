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

    # Per-user schema vector index settings
    vector_dim: int = 1024
    vector_similarity: str = "COSINE"

    # Kafka (Module 05)
    kafka_enabled: bool = True                 # start the consumer in the app lifespan
    kafka_bootstrap: str = "kafka:9092"
    kafka_topic: str = "personal-index-requests"
    kafka_group_id: str = "personal-db-service"
    kafka_dlq_topic: str = "personal-index-requests-dlq"
    kafka_max_attempts: int = 3                # then park in the DLQ and move on

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
    chat_model_name: str = "qwen2.5:7b"
    # Extraction calls go through module 07 (think=false by default there),
    # not llm_base_url's OpenAI-compat endpoint directly -- that layer has no
    # think control, and qwen3.x-family models "think" by default, measured
    # live at 10-20x the latency for an identical final answer (4.8s vs 59s
    # on the same extraction prompt). llm_base_url/chat_model_name kept for
    # the commented-out fallback path in graph_extractor.py's _call_llm().
    inference_url: str = "http://raptor-ai-lifecycle-api:8010"
    inference_think: bool = False  # qwen3.x-family "thinking" mode for _call_llm_raw's calls via module 07
    graph_extraction_max_moments: int = 10  # cap on per-moment temporal-fact LLM calls
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


settings = Settings()
