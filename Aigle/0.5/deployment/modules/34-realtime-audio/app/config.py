"""Module 34 settings, read once from the environment (see README for the full list)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from urllib.parse import quote


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    value = _env(name).lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    return default


def _redis_url() -> str:
    explicit = _env("REDIS_URL")
    if explicit:
        return explicit
    password = _env("REDIS_PASSWORD")
    auth = f":{quote(password, safe='')}@" if password else ""
    return (
        f"redis://{auth}{_env('REDIS_HOST', 'raptor-redis-standalone')}:"
        f"{_env('REDIS_PORT', '6379')}/{_env('REDIS_DB', '0')}"
    )


def _postgres_dsn() -> str:
    """RT_POSTGRES_DSN, else the platform POSTGRES_* variables with RT_POSTGRES_DB (realtime_audio)."""
    explicit = _env("RT_POSTGRES_DSN")
    if explicit:
        return explicit
    host = _env("POSTGRES_HOST")
    if not host:
        return ""
    user = quote(_env("POSTGRES_USER", "raptor"), safe="")
    password = quote(_env("POSTGRES_PASSWORD"), safe="")
    auth = f"{user}:{password}@" if password else f"{user}@"
    return f"postgresql://{auth}{host}:{_env('POSTGRES_PORT', '5432')}/{_env('RT_POSTGRES_DB', 'realtime_audio')}"


@dataclass(frozen=True)
class Settings:
    redis_url: str = "redis://raptor-redis-standalone:6379/0"

    # RA-1: tickets and sessions
    ticket_ttl_s: int = 30
    session_ttl_s: int = 1800
    max_sessions: int = 20
    # VIE01-365 AC: "enforces per-user quotas" -- max_sessions alone is service-wide, so one
    # user could otherwise take every slot. 3 is a starting default (a phone + a browser tab +
    # a bit of slack for a slow reconnect); adjust per deployment.
    max_sessions_per_user: int = 3
    config_timeout_s: float = 10.0
    heartbeat_s: float = 10.0

    # Auth (POST /audio/session/token): JWKS from Keycloak, or a fixed PEM public key
    keycloak_url: str = "http://raptor-keycloak:8080"
    jwt_public_key: str = ""
    jwt_audience: str = ""

    # RA-6: AudioService.AudioChat (gRPC bidirectional streaming), same JWT check as REST
    grpc_enabled: bool = True
    grpc_port: int = 8044

    # RA-2: VAD
    vad_chunk_ms: int = 30
    vad_threshold: float = 0.5
    vad_min_silence_ms: int = 300
    vad_min_speech_ms: int = 200
    max_utterance_s: float = 10.0
    vad_speech_pad_ms: int = 96
    vad_pool_size: int = 4
    # RTA-X4: bounded queues.  A full audio queue drops its oldest chunk (QUEUE_OVERFLOW); a full
    # STT queue drops the new utterance (STT_OVERLOADED).
    audio_queue_max: int = 200          # queued chunks per session (200 x 30ms = 6s)
    stt_queue_max: int = 10             # finished utterances waiting for their final transcript

    # RTA-X5: barge-in while an answer is playing
    barge_in_threshold: float = 0.7
    barge_in_min_ms: int = 300
    min_utterance_chars: int = 2        # a final transcript needs this many letters to start a turn

    # RTA-X3: framing, resume, keep-alive
    resume_window_s: float = 60.0       # 0 = a dropped connection ends the session
    resume_buffer_kb: int = 1024        # sent frames kept per session for replay after RESUME
    ping_interval_s: float = 30.0       # 0 = no PING
    pong_timeout_s: float = 10.0

    # Output: answer audio leaves at playback speed, at most RT_OUTPUT_LEAD_MS ahead of the
    # client's playback position; RT_OUTPUT_QUEUE_MS of synthesized audio may wait unsent.
    output_pace: bool = True
    output_lead_ms: int = 500
    output_queue_ms: int = 10000
    output_stall_s: float = 10.0        # a connected client that reads nothing this long: QUEUE_OVERFLOW

    # RA-8: false = accept 16 kHz PCM only (no resampling, no Opus / G.711)
    codec_resample: bool = True

    # RA-3 / RTA-X2: speech-to-text (faster-whisper)
    stt_enabled: bool = True
    stt_model: str = "large-v3"
    stt_model_dir: str = "/models/stt"
    stt_provisional_model: str = ""
    stt_device: str = "auto"
    stt_compute_type: str = ""
    stt_cpu_model: str = "tiny"
    stt_language: str = "zh-TW"          # default when CONFIG omits "language"; "auto" = detect
    stt_beam_size: int = 5
    stt_max_concurrent: int = 1
    stt_zh_convert: str = "s2tw"
    stt_streaming: bool = True
    stt_stride_s: float = 0.5
    stt_first_decode_ms: int = 300

    # RA-4: text-to-speech (app/tts_backends/)
    tts_enabled: bool = True
    tts_model: str = "kokoro-82m"
    tts_voice: str = ""
    tts_speed: float = 1.0
    tts_device: str = "auto"
    tts_kokoro_dir: str = "/models/tts/kokoro"
    tts_max_concurrent: int = 2
    tts_queue_max: int = 5              # answer sentences waiting for synthesis, per session
    tts_silence_pad_ms: int = 200
    tts_breeze_enabled: bool = False      # research / non-commercial licence: opt-in only
    tts_breeze_url: str = "http://raptor-breeze-tts:7860"
    tts_breeze_voices: str = ""
    tts_breeze_timeout_s: float = 60.0
    tts_breeze_busy_retries: int = 25

    # RA-5: voice turns through Module 15
    llm_enabled: bool = True
    chat_service_url: str = "http://raptor-chat-service:8021/api/v1"
    chat_stream: bool = True
    llm_timeout_s: float = 30.0
    llm_first_token_timeout_s: float = 30.0   # Module 15 often needs 8-29 s for its first sentence
    llm_fallback_text: str = "系統暫時無法回應，請稍後再試。"

    # RTA-X6 / RTA-08: Module 23 screens transcripts before Module 15 and answers before TTS
    guardrail_enabled: bool = True
    guardrail_url: str = "http://raptor-guardrail-service:8026"
    guardrail_output: bool = True
    guardrail_timeout_s: float = 5.0
    guardrail_fail_open: bool = True     # Module 23 unreachable: go on (logged), like Module 15
    guardrail_block_text: str = "抱歉，這個問題我無法協助處理。"

    # RTA-X9: Module 23 in external-brain sessions: off | input | output | both
    external_guardrail: str = "output"

    # RTA-X6 / RTA-07: transcripts -> Module 25 (Kafka); turns -> Module 26 when the brain does not
    index_transcripts: bool = True
    kafka_bootstrap: str = "kafka-broker1:9092,kafka-broker2:9092,kafka-broker3:9092"
    personal_index_topic: str = "personal-index-requests"
    memory_enabled: bool = True
    memory_url: str = "http://raptor-memory-service:8026"

    # RTA-X7 / RA-9: consent-aware session recording
    recording_enabled: bool = False
    module33_url: str = "http://raptor-privacy-compliance-service:8033"
    gd_internal_token: str = ""
    recording_topic: str = "audio-recording-finalize"
    recording_chunk_s: float = 10.0
    recording_retention_days: int = 90
    recording_collection: str = ""       # SeaweedFS collection; empty = the default one
    seaweedfs_master: str = "http://raptor-seaweedfs-master1:9333"
    postgres_dsn: str = ""

    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if self.external_guardrail not in {"off", "input", "output", "both"}:
            raise ValueError("RT_EXTERNAL_GUARDRAIL must be off, input, output or both")

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            redis_url=_redis_url(),
            ticket_ttl_s=_int("RT_TICKET_TTL_S", cls.ticket_ttl_s),
            session_ttl_s=_int("RT_SESSION_TTL_S", cls.session_ttl_s),
            max_sessions=_int("RT_MAX_SESSIONS", cls.max_sessions),
            max_sessions_per_user=_int("RT_MAX_SESSIONS_PER_USER", cls.max_sessions_per_user),
            config_timeout_s=_float("RT_CONFIG_TIMEOUT_S", cls.config_timeout_s),
            heartbeat_s=_float("RT_HEARTBEAT_S", cls.heartbeat_s),
            keycloak_url=_env("KEYCLOAK_URL", cls.keycloak_url).rstrip("/"),
            jwt_public_key=_env("RT_JWT_PUBLIC_KEY").replace("\\n", "\n"),
            jwt_audience=_env("RT_JWT_AUDIENCE"),
            grpc_enabled=_bool("RT_GRPC_ENABLED", cls.grpc_enabled),
            grpc_port=_int("RT_GRPC_PORT", cls.grpc_port),
            vad_chunk_ms=_int("RT_VAD_CHUNK_MS", cls.vad_chunk_ms),
            vad_threshold=_float("RT_VAD_THRESHOLD", cls.vad_threshold),
            vad_min_silence_ms=_int("RT_VAD_MIN_SILENCE_MS", cls.vad_min_silence_ms),
            vad_min_speech_ms=_int("RT_VAD_MIN_SPEECH_MS", cls.vad_min_speech_ms),
            max_utterance_s=_float("RT_MAX_UTTERANCE_S", cls.max_utterance_s),
            vad_speech_pad_ms=_int("RT_VAD_SPEECH_PAD_MS", cls.vad_speech_pad_ms),
            vad_pool_size=_int("RT_VAD_POOL_SIZE", cls.vad_pool_size),
            audio_queue_max=_int("RT_AUDIO_QUEUE_MAX", cls.audio_queue_max),
            # RT_UTTERANCE_QUEUE_MAX is the pre-RTA-X4 name of the same queue
            stt_queue_max=_int("RT_STT_QUEUE_MAX", _int("RT_UTTERANCE_QUEUE_MAX", cls.stt_queue_max)),
            barge_in_threshold=_float("RT_BARGE_IN_THRESHOLD", cls.barge_in_threshold),
            barge_in_min_ms=_int("RT_BARGE_IN_MIN_MS", cls.barge_in_min_ms),
            min_utterance_chars=_int("RT_MIN_UTTERANCE_CHARS", cls.min_utterance_chars),
            resume_window_s=_float("RT_SESSION_RESUME_WINDOW_S", cls.resume_window_s),
            resume_buffer_kb=_int("RT_RESUME_BUFFER_KB", cls.resume_buffer_kb),
            ping_interval_s=_float("RT_PING_INTERVAL_S", cls.ping_interval_s),
            pong_timeout_s=_float("RT_PONG_TIMEOUT_S", cls.pong_timeout_s),
            output_pace=_bool("RT_OUTPUT_PACE", cls.output_pace),
            output_lead_ms=_int("RT_OUTPUT_LEAD_MS", cls.output_lead_ms),
            output_queue_ms=_int("RT_OUTPUT_QUEUE_MS", cls.output_queue_ms),
            output_stall_s=_float("RT_OUTPUT_STALL_S", cls.output_stall_s),
            codec_resample=_bool("RT_CODEC_RESAMPLE", cls.codec_resample),
            stt_enabled=_bool("RT_STT_ENABLED", cls.stt_enabled),
            stt_model=_env("RT_STT_MODEL", cls.stt_model),
            stt_model_dir=_env("RT_STT_MODEL_DIR", cls.stt_model_dir),
            stt_provisional_model=_env("RT_STT_PROVISIONAL_MODEL"),
            stt_device=_env("RT_STT_DEVICE", cls.stt_device),
            stt_compute_type=_env("RT_STT_COMPUTE_TYPE"),
            stt_cpu_model=_env("RT_STT_CPU_MODEL", cls.stt_cpu_model),
            stt_language=_env("RT_STT_LANGUAGE", cls.stt_language),
            stt_beam_size=_int("RT_STT_BEAM_SIZE", cls.stt_beam_size),
            stt_max_concurrent=_int("RT_STT_MAX_CONCURRENT", cls.stt_max_concurrent),
            stt_zh_convert=os.environ.get("RT_STT_ZH_CONVERT", cls.stt_zh_convert),
            stt_streaming=_bool("RT_STT_STREAMING", cls.stt_streaming),
            stt_stride_s=_float("RT_STT_STRIDE_S", cls.stt_stride_s),
            stt_first_decode_ms=_int("RT_STT_FIRST_DECODE_MS", cls.stt_first_decode_ms),
            tts_enabled=_bool("RT_TTS_ENABLED", cls.tts_enabled),
            tts_model=_env("RT_TTS_MODEL", cls.tts_model),
            tts_voice=_env("RT_TTS_VOICE"),
            tts_speed=_float("RT_TTS_SPEED", cls.tts_speed),
            tts_device=_env("RT_TTS_DEVICE", cls.tts_device),
            tts_kokoro_dir=_env("RT_TTS_KOKORO_DIR", cls.tts_kokoro_dir),
            tts_max_concurrent=_int("RT_TTS_MAX_CONCURRENT", cls.tts_max_concurrent),
            tts_queue_max=_int("RT_TTS_QUEUE_MAX", cls.tts_queue_max),
            tts_silence_pad_ms=_int("RT_TTS_SILENCE_PAD_MS", cls.tts_silence_pad_ms),
            tts_breeze_enabled=_bool("RT_TTS_BREEZE_ENABLED", cls.tts_breeze_enabled),
            tts_breeze_url=_env("RT_TTS_BREEZE_URL", cls.tts_breeze_url),
            tts_breeze_voices=_env("RT_TTS_BREEZE_VOICES"),
            tts_breeze_timeout_s=_float("RT_TTS_BREEZE_TIMEOUT_S", cls.tts_breeze_timeout_s),
            tts_breeze_busy_retries=_int("RT_TTS_BREEZE_BUSY_RETRIES", cls.tts_breeze_busy_retries),
            llm_enabled=_bool("RT_LLM_ENABLED", cls.llm_enabled),
            chat_service_url=_env("RT_CHAT_SERVICE_URL", cls.chat_service_url),
            chat_stream=_bool("RT_CHAT_STREAM", cls.chat_stream),
            llm_timeout_s=_float("RT_LLM_TIMEOUT_S", cls.llm_timeout_s),
            llm_first_token_timeout_s=_float("RT_LLM_FIRST_TOKEN_TIMEOUT_S", cls.llm_first_token_timeout_s),
            llm_fallback_text=_env("RT_LLM_FALLBACK_TEXT", cls.llm_fallback_text),
            guardrail_enabled=_bool("RT_GUARDRAIL_ENABLED", cls.guardrail_enabled),
            guardrail_url=_env("RT_GUARDRAIL_URL", cls.guardrail_url).rstrip("/"),
            guardrail_output=_bool("RT_GUARDRAIL_OUTPUT", cls.guardrail_output),
            guardrail_timeout_s=_float("RT_GUARDRAIL_TIMEOUT_S", cls.guardrail_timeout_s),
            guardrail_fail_open=_bool("RT_GUARDRAIL_FAIL_OPEN", cls.guardrail_fail_open),
            guardrail_block_text=_env("RT_GUARDRAIL_BLOCK_TEXT", cls.guardrail_block_text),
            external_guardrail=_env("RT_EXTERNAL_GUARDRAIL", cls.external_guardrail).lower(),
            index_transcripts=_bool("RT_INDEX_TRANSCRIPTS", cls.index_transcripts),
            kafka_bootstrap=_env("RT_KAFKA_BOOTSTRAP", _env("KAFKA_BOOTSTRAP_SERVERS", cls.kafka_bootstrap)),
            personal_index_topic=_env("RT_KAFKA_PERSONAL_INDEX_TOPIC", cls.personal_index_topic),
            memory_enabled=_bool("RT_MEMORY_ENABLED", cls.memory_enabled),
            memory_url=_env("RT_MEMORY_SERVICE_URL", cls.memory_url).rstrip("/"),
            recording_enabled=_bool("RT_RECORDING_ENABLED", cls.recording_enabled),
            module33_url=_env("RT_MODULE33_URL", cls.module33_url).rstrip("/"),
            gd_internal_token=_env("GD_INTERNAL_TOKEN"),
            recording_topic=_env("RT_KAFKA_RECORDING_TOPIC", cls.recording_topic),
            recording_chunk_s=_float("RT_RECORDING_CHUNK_S", cls.recording_chunk_s),
            recording_retention_days=_int("RT_RECORDING_RETENTION_DAYS", cls.recording_retention_days),
            recording_collection=_env("RT_RECORDING_COLLECTION"),
            seaweedfs_master=_env("RT_SEAWEEDFS_MASTER", cls.seaweedfs_master).rstrip("/"),
            postgres_dsn=_postgres_dsn(),
            log_level=_env("LOG_LEVEL", cls.log_level),
        )


@lru_cache
def get_settings() -> Settings:
    return Settings.from_env()
