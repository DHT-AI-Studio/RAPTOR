# services/audio_recognizer_service/main.py

import asyncio
import concurrent.futures
import functools
import logging
import os
import sys
import uvicorn

# torch.load patch: pyannote/lightning checkpoints require weights_only=False on torch>=2.6
import torch as _torch
_orig_load = _torch.load
@functools.wraps(_orig_load)
def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)
_torch.load = _patched_load

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('audio_recognizer_service.log')
    ]
)

logger = logging.getLogger(__name__)

# Load AudioSummary from summary service directory (same worker image, different service path)
_services_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(_services_dir, "audio_summary_service"))

SYNC_API_PORT  = int(os.getenv("PORT_AUDIO_SYNC_API", "8019"))
INFERENCE_URL  = os.getenv("INFERENCE_URL",  "http://raptor-ai-lifecycle-api:8010")
INFERENCE_MODEL = os.getenv("INFERENCE_MODEL", "qwen3.5:9b")

from kafka_handler import AudioRecognizerKafkaHandler
from audio_recognition import AudioRecognitionClient
from audio_summary import AudioSummary
import sync_api


async def main():
    logger.info("Starting Audio Recognizer Service with Sync API...")
    logger.info(f"Sync API Port: {SYNC_API_PORT}")

    recognition_client = AudioRecognitionClient()
    summarizer = AudioSummary(
        model_name=INFERENCE_MODEL,
        inference_url=INFERENCE_URL,
    )

    # max_workers=1 serializes GPU inference across Kafka consumer and sync API
    thread_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="AudioRecognition"
    )

    sync_api.init(recognition_client, summarizer, thread_pool)
    kafka_handler = AudioRecognizerKafkaHandler(
        recognition_client=recognition_client,
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
        logger.info("Shutting down Audio Recognizer Service...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
    finally:
        thread_pool.shutdown(wait=False)


if __name__ == "__main__":
    asyncio.run(main())
