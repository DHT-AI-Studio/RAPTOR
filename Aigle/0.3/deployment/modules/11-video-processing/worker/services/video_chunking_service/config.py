# services/video_chunking_service/config.py

SERVICE_NAME = "video_chunking_service"

KAFKA_GROUP_ID = "video-chunking-service-group"
KAFKA_TOPIC_CHUNKING_REQUEST = "video-chunking-requests"
KAFKA_TOPIC_CHUNKING_RESULT = "video-chunking-results"
KAFKA_TOPIC_DLQ = "video-chunking-dlq"

# Chunking parameters
MOMENT_DURATION_SEC = 10
FRAMES_PER_MOMENT = 2          # top-N keyframes selected per moment
FRAME_SAMPLE_FPS = 1           # extract 1 frame/sec before scoring
BLUR_THRESHOLD = 100.0         # Laplacian variance; below = blurry
DIVERSITY_THRESHOLD = 0.85     # HSV histogram correlation; above = too similar, skip

# Normalization parameters
NORMALIZE_HEIGHT = 720         # target height (width auto-scaled, divisible by 2)
NORMALIZE_VIDEO_CODEC = "libx264"
NORMALIZE_CRF = 23             # H.264 quality (18=high, 28=low)
NORMALIZE_PRESET = "fast"
NORMALIZE_AUDIO_CODEC = "aac"
NORMALIZE_AUDIO_BITRATE = "128k"
NORMALIZE_AUDIO_SAMPLERATE = 44100
NORMALIZE_PIX_FMT = "yuv420p"

# Temp files
TEMP_FILE_DIR = "/tmp/media_processing/video_chunking"

# Redis
REDIS_DB = 0
REDIS_KEY_PREFIX = "video_chunking:"
REDIS_KEY_TTL = 86400

LOG_LEVEL = "INFO"
LOG_FILE = "video_chunking_service.log"
