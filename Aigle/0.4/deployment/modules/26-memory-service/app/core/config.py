from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    storage_root: str = "/mnt/memvid-storage"
    embedding_model: str = "BAAI/bge-m3"

    # Redis (shared cluster from Module 02)
    redis_host: str = "raptor-redis-standalone"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None

    # Extraction LLM (Ollama chat model for preference/entity/fact extraction)
    extraction_model: str = "qwen2.5:7b"

    # Module 07 (ASR + inference)
    module07_url: str = "http://raptor-ai-lifecycle-api:8010"
    inference_think: bool = False  # qwen3.x "thinking" mode for text-generation calls via module 07

    # Compact thresholds (MV-12)
    # threshold = context_window - reserved_output_tokens - buffer_tokens
    # 當 session tokens >= threshold 時觸發壓縮
    compact_buffer_tokens: int = 13000           # 安全緩衝，預留給 system prompt / token 計算誤差
    compact_reserved_output_tokens: int = 20000  # 保留給 LLM 輸出回覆的空間下限；請求帶 max_tokens 時取 max(該值, 此設定)
    compact_min_tail_tokens: int = 10000         # tail 最少保留的 token 數（不足時向前擴展）
    compact_min_text_messages: int = 5           # tail 最少保留的對話輪數
    compact_max_tail_tokens: int = 40000         # tail 最多保留的 token 數（超出部分才被摘要）
    compact_summary_max_tokens: int = 12000      # LLM 產生摘要的最大 token 數
    compact_max_section_tokens: int = 2000       # 既有 summary 每個 "# heading" 段落的上限，超出才截斷
    compact_llm_timeout_seconds: float = 120.0   # 呼叫 Module 07 做摘要的 HTTP timeout（秒）
    compact_keep_turns: int = 10                 # 無論 token 預算為何，一定保留的最近完整輪數
    compact_tool_result_retention_hours: float = 24.0  # 一定保留的 tool/result pair 時間窗（小時）

    model_config = {"env_prefix": "MEM_", "case_sensitive": False}


settings = Settings()

