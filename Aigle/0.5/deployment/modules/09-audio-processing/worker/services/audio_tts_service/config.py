# services/audio_tts_service/config.py

SERVICE_NAME = "audio_tts_service"

KAFKA_GROUP_ID = "audio-tts-service-group"
KAFKA_TOPIC_TTS_REQUEST = "audio-tts-requests"
KAFKA_TOPIC_TTS_RESULT = "audio-tts-results"
KAFKA_TOPIC_DLQ = "audio-tts-dlq"

# TTS inference
TTS_TASK = "tts"
TTS_ENGINE = "vibevoice"
TTS_MODEL = "vibevoice-tts"
DEFAULT_VOICE = "default"
DEFAULT_SPEED = 1.0
DEFAULT_FORMAT = "wav"

LOG_LEVEL = "INFO"
LOG_FILE = "audio_tts_service.log"
