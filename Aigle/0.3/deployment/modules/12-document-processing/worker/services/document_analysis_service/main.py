# services/document_analysis_service/main.py

import asyncio
import concurrent.futures
import logging
import os
import sys

import uvicorn
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('document_analysis_service.log')
    ]
)

logger = logging.getLogger(__name__)

# Load .env from parent (worker/) directory
# When invoked via top-level dispatcher: __file__ = /app/services/document_analysis_service/main.py
# services_dir = /app/services,  worker_dir = /app
_services_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_worker_dir = os.path.dirname(_services_dir)
load_dotenv(os.path.join(_worker_dir, ".env"))

INFERENCE_URL = os.getenv("INFERENCE_URL", "http://raptor-ai-lifecycle-api:8010")
INFERENCE_MODEL = os.getenv("INFERENCE_MODEL", "qwen3.5:9b")
SYNC_API_PORT = int(os.getenv("DOCUMENT_SYNC_API_PORT", "8020"))

# Append summary service so DocumentSummarizer can be imported,
# without shadowing document_analysis_service's own config.py
sys.path.append(os.path.join(_services_dir, "document_summary_service"))

from document_analysis import DocumentAnalysisClient
from document_summary import DocumentSummarizer
from kafka_handler import DocumentAnalysisKafkaHandler
import sync_api


async def main():
    logger.info("Starting Document Analysis Service (Kafka + Sync API)...")

    # Load VLM model once; share between Kafka consumer and sync API
    analysis_client = DocumentAnalysisClient(
        inference_url=INFERENCE_URL,
        inference_model=INFERENCE_MODEL,
    )
    summarizer = DocumentSummarizer(inference_url=INFERENCE_URL, model_name=INFERENCE_MODEL)

    # Single thread pool serializes VLM calls across both entry points
    thread_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="DocAnalysis"
    )

    sync_api.init(analysis_client, summarizer, thread_pool)

    kafka_handler = DocumentAnalysisKafkaHandler(
        analysis_client=analysis_client,
        thread_pool=thread_pool,
    )

    server_config = uvicorn.Config(
        sync_api.app, host="0.0.0.0", port=SYNC_API_PORT, log_level="info"
    )
    server = uvicorn.Server(server_config)

    try:
        await asyncio.gather(
            kafka_handler.start_consumer(),
            server.serve(),
        )
    except KeyboardInterrupt:
        logger.info("Shutting down Document Analysis Service...")
    finally:
        thread_pool.shutdown(wait=False)


if __name__ == "__main__":
    asyncio.run(main())
