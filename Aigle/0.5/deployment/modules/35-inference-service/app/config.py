"""Module 35 settings, read once from the environment.

Every value has a documented default so the service starts with only the
platform-wide .env; see .env.example for the full list.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from urllib.parse import quote

# Task keys used by GET/POST /models/default (INF-3) and by the model type
# each task needs.  Text tasks (translate/summarize/complete/transform) all
# resolve through the "text" default.
DEFAULT_TASKS = ("text", "stt", "tts", "clone", "diarize")
TASK_MODEL_TYPE = {"text": "llm", "stt": "stt", "tts": "tts", "clone": "clone", "diarize": "diarize"}


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _redis_url() -> str:
    explicit = _env("REDIS_URL")
    if explicit:
        return explicit
    host = _env("REDIS_HOST", "raptor-redis-standalone")
    port = _env("REDIS_PORT", "6379")
    db = _env("REDIS_DB", "0")
    password = _env("REDIS_PASSWORD")
    auth = f":{quote(password, safe='')}@" if password else ""
    return f"redis://{auth}{host}:{port}/{db}"


@dataclass(frozen=True)
class Settings:
    # Upstreams
    module07_api_url: str = "http://raptor-ai-lifecycle-api:8010"
    module07_ollama_host: str = "http://raptor-ollama:11434"
    module07_timeout_s: float = 120.0
    registry_timeout_s: float = 5.0
    model_cache_ttl_s: float = 60.0
    # VIE01-316/317: STT/TTS forward to Module 34's already-loaded Whisper/TTS models
    # instead of this service loading its own second copy -- see module34_client.py.
    module34_api_url: str = "http://raptor-realtime-audio:8034"
    module34_timeout_s: float = 60.0

    # Module 02 Redis (INF-3 defaults)
    redis_url: str = "redis://raptor-redis-standalone:6379/0"

    # Factory defaults per task (INF-3); empty = no default configured
    default_models: dict[str, str | None] = field(default_factory=dict)

    # Prompt templates (INF-5/6/8); empty = built-in template
    translate_system_prompt: str = ""
    summarize_system_prompt: str = ""
    transform_system_prompt: str = ""

    # INF-6 long-input handling
    summarize_chunk_chars: int = 12000
    summarize_max_chunks: int = 8

    log_level: str = "INFO"
    guardrail_enabled: bool = False
    guardrail_check_output: bool = False
    module23_url: str = "http://raptor-guardrail-service:8026"
    guardrail_timeout_s: float = 20.0
    guardrail_input_path: str = "/guardrail/policy/decide"
    guardrail_output_path: str = "/guardrail/policy/decide"
    voice_enabled: bool = False
    voice_model_dir: str = "/opt/voice-models"
    # VIE01-419 / INF-20: XTTS-v2 weights are Coqui Public Model License
    # (non-commercial) -- /voice/clone stays off even with voice_enabled,
    # until this is explicitly opted into.
    clone_enabled: bool = False
    # Which clone backends to actually load; "xtts-v2" preserves today's default
    # behavior exactly. A build that never bakes a listed backend's weights fails
    # loudly at image-build time (Dockerfile.voice's bundle-completeness check),
    # not silently at first request.
    clone_backends_runtime: frozenset[str] = frozenset({"xtts-v2"})
    # Eagerly load these at startup instead of on first request; default empty
    # means every configured backend loads lazily, on its first real use.
    clone_backends_warm: frozenset[str] = frozenset()
    speaker_cache_ttl_s: int = 86400
    speaker_cache_max: int = 100
    # GD-1 / VIE01-246: shared secret Module 33 presents as X-Internal-Token
    # on /internal/gdpr/*; empty rejects every call. GD-11 also sends it to
    # Module 33's GET /consent.
    gdpr_internal_token: str = ""
    # GD-11: Module 33, asked per /voice/clone request whether the caller
    # granted voice_clone consent (GDPR Art. 9 biometric data).
    module33_url: str = "http://raptor-privacy-compliance-service:8033"

    @classmethod
    def from_env(cls) -> Settings:
        clone_backends_runtime = frozenset(
            n.strip() for n in _env("CLONE_BACKENDS_RUNTIME", "xtts-v2").split(",") if n.strip()
        )
        clone_backends_warm = frozenset(
            n.strip() for n in _env("CLONE_BACKENDS_WARM").split(",") if n.strip()
        )
        default_models = {task: (_env(f"DEFAULT_{task.upper()}_MODEL") or None) for task in DEFAULT_TASKS}
        if default_models["clone"] is None:
            # Single-backend deployments need zero new config. A "both" deployment
            # with no explicit default implicitly prefers Chatterbox (no license
            # restriction, unlike XTTS-v2's Coqui Public Model License) -- still
            # overridable via DEFAULT_CLONE_MODEL or POST /models/default.
            if len(clone_backends_runtime) == 1:
                default_models["clone"] = next(iter(clone_backends_runtime))
            elif "chatterbox-multilingual-v3" in clone_backends_runtime:
                default_models["clone"] = "chatterbox-multilingual-v3"
        return cls(
            module07_api_url=_env("MODULE07_API_URL", cls.module07_api_url).rstrip("/"),
            module07_ollama_host=_env("MODULE07_OLLAMA_HOST", cls.module07_ollama_host).rstrip("/"),
            module07_timeout_s=_env_float("MODULE07_TIMEOUT_S", cls.module07_timeout_s),
            registry_timeout_s=_env_float("MODULE07_REGISTRY_TIMEOUT_S", cls.registry_timeout_s),
            module34_api_url=_env("MODULE34_API_URL", cls.module34_api_url).rstrip("/"),
            module34_timeout_s=_env_float("MODULE34_TIMEOUT_S", cls.module34_timeout_s),
            model_cache_ttl_s=_env_float("MODEL_CACHE_TTL_S", cls.model_cache_ttl_s),
            redis_url=_redis_url(),
            default_models=default_models,
            translate_system_prompt=_env("TRANSLATE_SYSTEM_PROMPT"),
            summarize_system_prompt=_env("SUMMARIZE_SYSTEM_PROMPT"),
            transform_system_prompt=_env("TRANSFORM_SYSTEM_PROMPT"),
            summarize_chunk_chars=max(1000, _env_int("SUMMARIZE_CHUNK_CHARS", cls.summarize_chunk_chars)),
            summarize_max_chunks=max(1, _env_int("SUMMARIZE_MAX_CHUNKS", cls.summarize_max_chunks)),
            log_level=_env("LOG_LEVEL", cls.log_level),
            guardrail_enabled=_env("GUARDRAIL_ENABLED", "false").lower() == "true",
            guardrail_check_output=_env("GUARDRAIL_CHECK_OUTPUT", "false").lower() == "true",
            module23_url=_env("MODULE23_URL", cls.module23_url).rstrip("/"),
            guardrail_timeout_s=max(0.1, _env_float("GUARDRAIL_TIMEOUT_S", 20.0)),
            guardrail_input_path=_env("GUARDRAIL_INPUT_PATH", cls.guardrail_input_path),
            guardrail_output_path=_env("GUARDRAIL_OUTPUT_PATH", cls.guardrail_output_path),
            voice_enabled=_env("VOICE_ENABLED", "false").lower() == "true",
            voice_model_dir=_env("VOICE_MODEL_DIR", cls.voice_model_dir),
            clone_enabled=_env("CLONE_ENABLED", "false").lower() == "true",
            clone_backends_runtime=clone_backends_runtime,
            clone_backends_warm=clone_backends_warm,
            speaker_cache_ttl_s=max(1, _env_int("SPEAKER_CACHE_TTL_S", cls.speaker_cache_ttl_s)),
            speaker_cache_max=max(1, _env_int("SPEAKER_CACHE_MAX", cls.speaker_cache_max)),
            gdpr_internal_token=_env("GD_INTERNAL_TOKEN"),
            module33_url=_env("MODULE33_URL", cls.module33_url).rstrip("/"),
        )


@lru_cache
def get_settings() -> Settings:
    return Settings.from_env()
