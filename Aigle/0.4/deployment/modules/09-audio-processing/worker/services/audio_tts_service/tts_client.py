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
                body = await resp.json()

        # Module 07's /inference/tts (post-0.4 refactor) nests the actual
        # synthesis output under "result" (same envelope as /inference/infer),
        # unlike the pre-refactor endpoint which returned these fields flat.
        # Note: the new endpoint doesn't report audio duration (sample_rate is
        # available instead) — there's no "duration_seconds" to forward anymore.
        result = body.get("result", {})
        return {
            "audio_base64": result.get("audio_base64"),
            "format": result.get("format", output_format),
            "sample_rate": result.get("sample_rate"),
            "processing_time": body.get("processing_time"),
        }
