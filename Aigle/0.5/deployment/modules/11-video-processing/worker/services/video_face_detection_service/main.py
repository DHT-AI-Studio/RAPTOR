# services/video_face_detection_service/main.py
# Kafka consumer（orchestrator 的平行請求）與 sync API（/embed-face）共用同一個容器、
# 同一組常駐的 ONNX session（v1.4 §4.5，照 video_ocr_frame_service 的做法）。

import asyncio
import logging
import os
import sys

import sync_api
import uvicorn
from config import FACE_MODEL_DOWNLOAD, FACE_MODEL_FILES, LOG_FILE, LOG_LEVEL
from detector import FaceDetector
from kafka_handler import FaceDetectionKafkaHandler
from model_files import ensure_models

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE)
    ]
)

logger = logging.getLogger(__name__)

SYNC_PORT = int(os.getenv("PORT_VIDEO_FACE_SYNC", "8033"))


async def main():
    logger.info("Starting Video Face Detection Service with Sync API...")
    logger.info(f"Sync API Port: {SYNC_PORT}")

    if FACE_MODEL_DOWNLOAD:
        # 第一次啟動時下載到 volume（約 270MB），之後只驗證 SHA-256；在 event loop 外執行
        await asyncio.to_thread(ensure_models, FACE_MODEL_FILES)
    detector = FaceDetector()
    sync_api.init(detector)
    kafka_handler = FaceDetectionKafkaHandler(detector)

    server_config = uvicorn.Config(
        sync_api.app, host="0.0.0.0", port=SYNC_PORT, log_level="info"
    )
    server = uvicorn.Server(server_config)

    try:
        await asyncio.gather(
            kafka_handler.start_consumer(),
            server.serve(),
        )
    except KeyboardInterrupt:
        logger.info("Shutting down Video Face Detection Service...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
