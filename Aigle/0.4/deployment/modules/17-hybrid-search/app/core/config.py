from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Qdrant Settings
    QDRANT_HOST: str = "localhost"
    PORT_QDRANT: int = 6333
    QDRANT_COLLECTION: str = "raptor"

    # OpenSearch Settings
    OPENSEARCH_HOST: str = "localhost"
    OPENSEARCH_PORT: int = 9200
    OPENSEARCH_INDEX: str = "hybrid_index"
    OPENSEARCH_USER: str = "admin"
    OPENSEARCH_PASSWORD: str = "admin"
    OPENSEARCH_DASHBOARDS_HOST: str = "localhost"
    OPENSEARCH_VERIFY_CERTS: bool = False

    # rrf-fusion Settings
    RRF_K_FACTOR: int = 60
    RERANK_DEPTH: int = 15           # reranker 吃幾份文件（top_k * 3 後再截）
    RERANK_MAX_DOC_CHARS: int = 512  # reserved — truncation currently disabled; tokenizer handles max_length

    # Embedding Model Settings
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    VECTOR_DIM: int = 1024

    # Reranker Model Settings
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANKER_TEMPERATURE: float = 0.25  # sigmoid temperature scaling; <1 sharpens score distribution

    # App settings
    DEBUG: bool = False

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()