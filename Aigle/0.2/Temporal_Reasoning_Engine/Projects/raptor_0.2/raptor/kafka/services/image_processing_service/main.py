# services/image_processing_service/main.py

import asyncio
import logging
import sys
from kafka_handler import ImageProcessingKafkaHandler

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

async def main():
    """主程式入口 - 統一的圖片處理服務"""
    logger.info("Starting Unified Image Processing Service...")
    logger.info("This service handles both image description and OCR requests")
    
    kafka_handler = ImageProcessingKafkaHandler()
    
    try:
        # 初始化服務（只初始化一次模型）
        await kafka_handler.initialize()
        
        # 啟動 Kafka consumer（同時監聽兩個 topic）
        logger.info("Starting to listen on multiple topics:")
        logger.info("- image-description-requests")
        logger.info("- image-ocr-requests")
        
        await kafka_handler.start_consumer()
        
    except KeyboardInterrupt:
        logger.info("Shutting down Image Processing Service...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
