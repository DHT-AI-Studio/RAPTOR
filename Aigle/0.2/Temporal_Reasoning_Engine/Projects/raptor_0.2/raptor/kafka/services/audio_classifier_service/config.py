# services/audio_classifier_service/config.py

import os
from pathlib import Path

# Kafka 配置
KAFKA_BOOTSTRAP_SERVERS = '192.168.157.165:19002,192.168.157.165:19003,192.168.157.165:19004'
KAFKA_TOPIC_CLASSIFIER_REQUEST = "audio-classifier-requests"
KAFKA_TOPIC_CLASSIFIER_RESULT = "audio-classifier-results"
KAFKA_TOPIC_DLQ = "audio-classifier-dlq"

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
SERVICE_NAME = "audio_classifier_service"
LOG_LEVEL = "INFO"

# 臨時檔案配置
TEMP_FILE_DIR = "/tmp/audio_processing"
CLASSIFICATION_RESULTS_DIR = os.path.join(TEMP_FILE_DIR, "classification_results")

# 確保目錄存在
Path(TEMP_FILE_DIR).mkdir(parents=True, exist_ok=True)
Path(CLASSIFICATION_RESULTS_DIR).mkdir(parents=True, exist_ok=True)

# PANNs 配置
PANNS_TOP_K = 5
PANNS_SEGMENT_LENGTH = 30.0

# LangSmith 配置
LANGSMITH_PROJECT = 'audioprocess'

# 支援的音頻檔案類型
SUPPORTED_AUDIO_TYPES = ['wav', 'mp3', 'mp4', 'flac', 'ogg', 'm4a', 'wma']
