# services/audio_tts_service/tts_client.py

import logging
import aiohttp
from typing import Dict, Any
from config import TTS_TASK, TTS_ENGINE, TTS_MODEL, DEFAULT_VOICE, DEFAULT_SPEED, DEFAULT_FORMAT
from dotenv import load_dotenv
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(parent_dir, ".env"))
INFERENCE_URL = os.getenv("INFERENCE_URL", "http://raptor-ai-lifecycle-api:8010/inference/tts")

logger = logging.getLogger(__name__)


class TTSClient:
    async def synthesize(
        self,
        text: str,
        voice: str = DEFAULT_VOICE,
        speed: float = DEFAULT_SPEED,
        output_format: str = DEFAULT_FORMAT,
    ) -> Dict[str, Any]:
        payload = {
            "text": text,
            "voice": voice,
            "speed": speed,
            "output_format": output_format,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                INFERENCE_URL,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status != 200:
                    text_body = await resp.text()
                    raise RuntimeError(f"TTS API error {resp.status}: {text_body}")
                return await resp.json()
