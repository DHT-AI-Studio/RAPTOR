# services/audio_recognizer_service/config.py

import os
from pathlib import Path

# Kafka 配置
#KAFKA_BOOTSTRAP_SERVERS = '192.168.157.165:19002,192.168.157.165:19003,192.168.157.165:19004'
KAFKA_TOPIC_RECOGNIZER_REQUEST = "audio-recognizer-requests"
KAFKA_TOPIC_RECOGNIZER_RESULT = "audio-recognizer-results"
KAFKA_TOPIC_DLQ = "audio-recognizer-dlq"

KAFKA_CONSUMER_CONFIG = {
    "session_timeout_ms": 120000,        # 2 分鐘
    "heartbeat_interval_ms": 20000,      # 20 秒
    "max_poll_interval_ms": 600000,      # 10 分鐘
    "request_timeout_ms": 130000,        
    "enable_auto_commit": True,
    "auto_commit_interval_ms": 5000,     # 5 秒自動提交
    "auto_offset_reset": "latest",       # 從最新消息開始
}

# 異步處理
ASYNC_PROCESSING_CONFIG = {
    "max_workers": 4,                    # 線程池最大工作線程數
    "max_concurrent_tasks": 8,           # 最大並發任務數
    "task_timeout": 1800,                # 單個任務超時時間（30分鐘）
    "enable_task_monitoring": True,      # 啟用任務監控
}

# 服務配置
SERVICE_NAME = "audio_recognizer_service"
LOG_LEVEL = "INFO"

# 臨時檔案配置
TEMP_FILE_DIR = "/tmp/audio_processing"
RECOGNITION_RESULTS_DIR = os.path.join(TEMP_FILE_DIR, "recognition_results")

# 確保目錄存在
Path(TEMP_FILE_DIR).mkdir(parents=True, exist_ok=True)
Path(RECOGNITION_RESULTS_DIR).mkdir(parents=True, exist_ok=True)

# WhisperX 配置
WHISPER_MODEL = os.getenv('WHISPER_MODEL', 'large-v3')
WHISPER_BATCH_SIZE = int(os.getenv('WHISPER_BATCH_SIZE', '16'))

# LangSmith 配置
LANGSMITH_PROJECT = os.getenv('LANGSMITH_PROJECT', 'audioprocess')

# 支援的音頻檔案類型
SUPPORTED_AUDIO_TYPES = ['wav', 'mp3', 'mp4', 'flac', 'ogg', 'm4a', 'wma']
