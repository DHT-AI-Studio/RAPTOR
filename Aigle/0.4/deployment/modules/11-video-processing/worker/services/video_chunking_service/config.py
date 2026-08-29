# services/video_chunking_service/config.py

import os

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
# subprocess.run(..., timeout=...) for each of the 3 ffmpeg fallback tiers
# (full-GPU CUDA, CPU decode + nvenc, full CPU libx264) -- was a bare literal
# 600 repeated 3x directly in chunker.py, not pulled in here like the other
# normalize params, and not configurable at all. Confirmed live this can
# actually be hit: a real upload's 3rd (full-CPU, slowest) tier timed out at
# exactly 600s after the first two GPU tiers had already failed.
NORMALIZE_TIMEOUT_SEC = int(os.environ.get("VIDEO_NORMALIZE_TIMEOUT_SEC", 600))
# GPU tiers (full-GPU CUDA, CPU-decode+nvenc) retry this many times before
# chunker.py gives up and falls through to the next tier -- found live that
# a real upload hit CUDA_ERROR_NO_DEVICE/OpenEncodeSessionEx on both GPU
# tiers, fell all the way to the slowest full-CPU tier, which then itself
# timed out; nvidia-smi confirmed the GPU was healthy again shortly after,
# suggesting the failure was transient rather than a real absence of GPU.
# Retrying cheaply here avoids paying the CPU tier's much higher timeout
# cost for what may just be a momentary driver/toolkit hiccup. Only applies
# to the two GPU tiers -- the final full-CPU tier is not retried, since by
# the time we're there both GPU tiers have already had their own retries.
NORMALIZE_GPU_RETRIES = int(os.environ.get("VIDEO_NORMALIZE_GPU_RETRIES", 3))
NORMALIZE_GPU_RETRY_DELAY_SEC = float(os.environ.get("VIDEO_NORMALIZE_GPU_RETRY_DELAY_SEC", 5))

# Temp files
TEMP_FILE_DIR = "/tmp/media_processing/video_chunking"

# Redis
REDIS_DB = 0
REDIS_KEY_PREFIX = "video_chunking:"
REDIS_KEY_TTL = 86400

LOG_LEVEL = "INFO"
LOG_FILE = "video_chunking_service.log"
