# services/document_analysis_service/main.py

import asyncio
import logging
import sys


# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('document_analysis_service.log')
    ]
)

logger = logging.getLogger(__name__)

from kafka_handler import DocumentAnalysisKafkaHandler

async def main():
    """主程式入口"""
    logger.info("Starting Document Analysis Service with Standard Message Format...")
    
    kafka_handler = DocumentAnalysisKafkaHandler()
    
    try:
        await kafka_handler.start_consumer()
    except KeyboardInterrupt:
        logger.info("Shutting down Document Analysis Service...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
