from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_DATA_DIR = Path(__file__).parent.parent / "data"


class Settings(BaseSettings):
    """Query Orchestrator 統一設定。

    所有環境變數預設加上 QO_ 前綴（例如 QO_LOG_LEVEL、QO_INFERENCE_URL）。
    可在 .env 檔案中覆蓋。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="QO_",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),
    )

    # ── Logging ───────────────────────────────────────────────────────
    log_level: str = Field("INFO", description="Log level (DEBUG, INFO, WARNING, ERROR)")

    # ── LLM / Inference ──────────────────────────────────────────────
    inference_url: str = Field(
        "http://raptor-ai-lifecycle-api:8010",
        description="Inference API base URL",
    )
    inference_model: str = Field(
        "qwen3.5:0.8b",
        description="Default LLM model for re-ranking",
    )
    inference_timeout: int = Field(
        15,
        description="Inference API timeout in seconds",
    )
    inference_think: bool = Field(
        False,
        description="Ollama 'thinking' mode for _call_llm_rerank's LLM re-rank call",
    )

    # ── Embedding model & index paths ────────────────────────────────
    model_name: str = Field(
        "BAAI/bge-m3",
        description="Sentence-transformer embedding model name",
    )
    faiss_index_path: str = Field(
        str(_DATA_DIR / "faiss_index.pkl"),
        description="Path to FAISS index file",
    )
    bm25_index_path: str = Field(
        str(_DATA_DIR / "bm25_index.pkl"),
        description="Path to BM25 index file",
    )
    intent_taxonomy_path: str = Field(
        str(_DATA_DIR / "intent_taxonomy.yaml"),
        description="Path to intent taxonomy YAML",
    )

    # ── Classification thresholds ────────────────────────────────────
    default_confidence_threshold: float = Field(
        0.35,
        description="Minimum confidence to accept a classification",
    )
    gap_threshold: float = Field(
        0.1,
        description="Score gap threshold for single vs. multi pipeline",
    )
    bm25_threshold: float = Field(
        1.5,
        description="BM25 gate threshold; below this the query is unknown",
    )
    bm25_weight: float = Field(
        0.2,
        description="Fixed BM25 weight (fallback when LightGBM unavailable)",
    )
    vector_weight: float = Field(
        0.8,
        description="Fixed FAISS/VECTOR weight (fallback when LightGBM unavailable)",
    )

    # ── LightGBM dynamic fusion ──────────────────────────────────────
    use_lightgbm_fusion: bool = Field(
        True,
        description="Enable LightGBM dynamic weight fusion (paper core improvement: +1.4 pp)",
    )
    lightgbm_model_path: str = Field(
        str(_DATA_DIR / "lightgbm_fusion_model.pkl"),
        description="Path to LightGBM model file",
    )
    lightgbm_features: list = Field(
        default_factory=list,  # lazy init below
        description="Feature list for LightGBM ranker",
    )
    lightgbm_rerank_threshold: float = Field(
        0.15,
        description="Ranker top-1/top-2 gap threshold; below this triggers LLM re-rank",
    )
    use_llm_rerank: bool = Field(
        False,
        description="Set True to enable LLM re-rank tier",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
         # Lazy init for list default value
        if not self.lightgbm_features:
            self.lightgbm_features = [
                 "bm25_max", "bm25_mean", "faiss_max", "faiss_mean",
                 "query_length", "bm25_top1", "faiss_top1", "bm25_faiss_gap",
             ]

    # ── Backward-compatible uppercase aliases ────────────────────────
    # These properties exist solely to support existing code that uses
    # settings.BM25_THRESHOLD, settings.MODEL_NAME, etc.

    @property
    def QO_LOG_LEVEL(self) -> str:
        return self.log_level

    @property
    def INFERENCE_URL(self) -> str:
        return self.inference_url

    @property
    def INFERENCE_MODEL(self) -> str:
        return self.inference_model

    @property
    def INFERENCE_TIMEOUT(self) -> int:
        return self.inference_timeout

    @property
    def INFERENCE_THINK(self) -> bool:
        return self.inference_think

    @property
    def MODEL_NAME(self) -> str:
        return self.model_name

    @property
    def FAISS_INDEX_PATH(self) -> str:
        return self.faiss_index_path

    @property
    def BM25_INDEX_PATH(self) -> str:
        return self.bm25_index_path

    @property
    def INTENT_TAXONOMY_PATH(self) -> str:
        return self.intent_taxonomy_path

    @property
    def DEFAULT_CONFIDENCE_THRESHOLD(self) -> float:
        return self.default_confidence_threshold

    @property
    def GAP_THRESHOLD(self) -> float:
        return self.gap_threshold

    @property
    def BM25_THRESHOLD(self) -> float:
        return self.bm25_threshold

    @property
    def BM25_WEIGHT(self) -> float:
        return self.bm25_weight

    @property
    def VECTOR_WEIGHT(self) -> float:
        return self.vector_weight

    @property
    def USE_LIGHTGBM_FUSION(self) -> bool:
        return self.use_lightgbm_fusion

    @property
    def LIGHTGBM_MODEL_PATH(self) -> str:
        return self.lightgbm_model_path

    @property
    def LIGHTGBM_FEATURES(self) -> list:
        return self.lightgbm_features

    @property
    def LIGHTGBM_RERANK_THRESHOLD(self) -> float:
        return self.lightgbm_rerank_threshold

    @property
    def USE_LLM_RERANK(self) -> bool:
        return self.use_llm_rerank


@lru_cache()
def get_settings() -> Settings:
    """Return a cached Settings instance.

    Using lru_cache ensures that all modules share the same settings object
    throughout the application lifecycle.
    """
    return Settings()
