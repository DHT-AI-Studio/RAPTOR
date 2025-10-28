# services/image_processing_service/config.py

SERVICE_NAME = "image_processing_service"

# Kafka配置
#KAFKA_BOOTSTRAP_SERVERS = '192.168.157.165:19002,192.168.157.165:19003,192.168.157.165:19004'
KAFKA_GROUP_ID = 'image-processing-service-group'

# Topic配置 - listen to
KAFKA_TOPIC_DESCRIPTION_REQUEST = "image-description-requests"
KAFKA_TOPIC_OCR_REQUEST = "image-ocr-requests"

# Topic配置 - produce to
KAFKA_TOPIC_DESCRIPTION_RESULT = "image-description-results"
KAFKA_TOPIC_OCR_RESULT = "image-ocr-results"
KAFKA_TOPIC_DLQ = "image-processing-dlq"

# Model配置
MODEL_PATH = 'OpenGVLab/InternVL3_5-4B'
CUDA_VISIBLE_DEVICES = "3"
MAX_MEMORY_PER_GPU = "36GiB"

# 處理配置
DESCRIPTION_PROMPT = "Provide a comprehensive summary of this image content. If there is text in the image, summarize the key information from the text. If there is no text, describe the visual elements, objects, people, scenes, colors, and any notable details you observe. Reply in traditional Chinese. 並忽略重複浮水印內容。"

OCR_PROMPT = "Extract all text from this image, maintaining original formatting. No additional explanation. 並忽略重複浮水印內容。 If there is no text in the image, respond with exactly: 'NO_TEXT_FOUND'"

# 臨時檔案配置
TEMP_FILE_DIR = "/tmp/image_processing"

# 日誌配置
LOG_LEVEL = "INFO"
LOG_FILE = "image_processing_service.log"
