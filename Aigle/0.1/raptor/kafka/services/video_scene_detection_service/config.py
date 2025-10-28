# services/video_scene_detection_service/config.py

SERVICE_NAME = "video_scene_detection_service"

# Kafka配置
#KAFKA_BOOTSTRAP_SERVERS = '192.168.157.165:19002,192.168.157.165:19003,192.168.157.165:19004'
KAFKA_GROUP_ID = 'video-scene-detection-service-group'

# Topic配置 - listen to
KAFKA_TOPIC_REQUEST = "video-scene-detection-requests"

# Topic配置 - produce to
KAFKA_TOPIC_RESPONSE = "video-scene-detection-results"
KAFKA_TOPIC_DLQ = "video-scene-detection-dlq"

# 場景檢測配置
SCENE_OUTPUT_BASE_DIR = "/tmp/video_processing/scene_detection"
SCENE_RESULTS_BASE_DIR = "/tmp/video_processing/scene_results"
SCENE_DETECTION_THRESHOLD = None  # 自動計算閾值
TEMP_FILE_CLEANUP_DELAY = 300  # 5分鐘後清理

# 狀態管理配置
STATE_TIMEOUT = 1800  # 30分鐘超時
MAX_RETRY_COUNT = 3

# 日誌配置
LOG_LEVEL = "INFO"
LOG_FILE = "video_scene_detection_service.log"
