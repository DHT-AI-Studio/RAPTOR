# services/image_processing_service/main.py

import asyncio
import concurrent.futures
import logging
import sys
import uvicorn
from dotenv import load_dotenv
import os

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('image_processing_service.log')
    ]
)

logger = logging.getLogger(__name__)

# 載入 .env 從 worker/ 目錄
_services_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_worker_dir = os.path.dirname(_services_dir)
load_dotenv(os.path.join(_worker_dir, ".env"))

SYNC_API_PORT = int(os.getenv("PORT_IMAGE_SYNC_API", "8018"))

from kafka_handler import ImageProcessingKafkaHandler
from image_processor import ImageProcessor
import sync_api

async def main():
    """主程式入口 - 統一的圖片處理服務"""
    logger.info("Starting Unified Image Processing Service with Sync API...")
    logger.info("This service handles both image description and OCR requests")
    logger.info(f"Sync API Port: {SYNC_API_PORT}")
    
    # 初始化模型
    processor = ImageProcessor()
    processor.initialize_model()
    
    # 單一線程池序列化 VLM 調用（供 Kafka consumer 和 sync API 共用）
    thread_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="ImgProcessing"
    )
    
    # 初始化 Sync API
    sync_api.init(processor, thread_pool)
    
    # 初始化 Kafka handler
    kafka_handler = ImageProcessingKafkaHandler()
    
    server_config = uvicorn.Config(
        sync_api.app, host="0.0.0.0", port=SYNC_API_PORT, log_level="info"
    )
    server = uvicorn.Server(server_config)
    
    try:
        # 同時啟動 Kafka consumer 和 Sync API
        await asyncio.gather(
            kafka_handler.initialize(),
            kafka_handler.start_consumer(),
            server.serve(),
        )
    except KeyboardInterrupt:
        logger.info("Shutting down Image Processing Service...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
    finally:
        thread_pool.shutdown(wait=False)

if __name__ == "__main__":
    asyncio.run(main())
