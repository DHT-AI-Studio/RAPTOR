# services/audio_diarization_service/main.py

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
        logging.FileHandler('audio_diarization_service.log')
    ]
)

logger = logging.getLogger(__name__)

SYNC_API_PORT = int(os.getenv("PORT_AUDIO_DIARIZATION_SYNC", "8029"))

from audio_diarization import AudioDiarizationClient
from kafka_handler import AudioDiarizationKafkaHandler
import sync_api


async def main():
    logger.info("Starting Audio Diarization Service (Kafka + Sync API)...")
    logger.info(f"Sync API Port: {SYNC_API_PORT}")

    diarization_client = AudioDiarizationClient()

    # max_workers=1 serializes model calls across Kafka consumer and sync API
    thread_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="AudioDiarization"
    )

    sync_api.init(diarization_client, thread_pool)
    kafka_handler = AudioDiarizationKafkaHandler(
        diarization_client=diarization_client,
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
        logger.info("Shutting down Audio Diarization Service...")
    finally:
        thread_pool.shutdown(wait=False)


if __name__ == "__main__":
    asyncio.run(main())
