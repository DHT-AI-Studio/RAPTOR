"""Optional Breeze TTS 2 HTTP backend; requires licence opt-in."""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator

import numpy as np

from app.tts_backends.base import BackendUnavailable, TtsBackend, VoiceInfo

logger = logging.getLogger(__name__)

LICENCE = ("Breeze TTS 2 weights and outputs are licensed for research and non-commercial use only; "
           "set RT_TTS_BREEZE_ENABLED=true to accept the BreezeBlue licence")

DEFAULT_VOICES = {
    "zh-female": {"language": "zh", "gender": "female", "seed": 42, "cfg_scale": 4,
                  "instruction": "一位親切的台灣年輕女性客服人員，聲音清晰溫和，語速適中，語氣專業有禮。"},
    "zh-male": {"language": "zh", "gender": "male", "seed": 42, "cfg_scale": 4,
                "instruction": "一位沉穩友善的台灣成年男性客服人員，聲音清晰，語速適中，語氣專業有禮。"},
    "en-female": {"language": "en", "gender": "female", "seed": 42, "cfg_scale": 4,
                  "instruction": "A warm, clear young woman with a friendly, professional customer-service tone."},
}


class BreezeBackend(TtsBackend):
    sample_rate = 24000

    def __init__(self, settings) -> None:
        super().__init__(settings)
        self.url = settings.tts_breeze_url.rstrip("/")
        try:
            presets = json.loads(settings.tts_breeze_voices) if settings.tts_breeze_voices else DEFAULT_VOICES
            if not isinstance(presets, dict) or not presets:
                raise ValueError("expected a non-empty object of voice presets")
            for vid, preset in presets.items():
                if not vid or not isinstance(preset, dict):
                    raise ValueError("each voice must have a name and a preset object")
                for field in ("instruction", "ref_audio", "ref_text", "language", "gender"):
                    if field in preset and (not isinstance(preset[field], str) or not preset[field].strip()):
                        raise ValueError(f"{vid}: {field} must be a non-empty string")
                if not preset.get("instruction") and not (preset.get("ref_audio") and preset.get("ref_text")):
                    raise ValueError(f"{vid}: set instruction or both ref_audio and ref_text")
        except (ValueError, TypeError) as exc:
            raise BackendUnavailable(f"invalid RT_TTS_BREEZE_VOICES: {exc}") from exc
        self._presets: dict[str, dict] = presets
        self.model_info: dict = {}
        self._client = None
        self._t2s = None

    @classmethod
    def name(cls) -> str:
        return "breeze-tts-2"

    @classmethod
    def check(cls, settings) -> None:
        if not settings.tts_breeze_enabled:
            raise BackendUnavailable(LICENCE)
        if not settings.tts_breeze_url:
            raise BackendUnavailable("RT_TTS_BREEZE_URL is not set")
        import httpx

        try:
            httpx.get(f"{settings.tts_breeze_url.rstrip('/')}/health", timeout=3).raise_for_status()
        except httpx.HTTPError as exc:
            raise BackendUnavailable(f"Breeze server not reachable at {settings.tts_breeze_url}: {exc}") from exc

    def load(self) -> None:
        import httpx
        from opencc import OpenCC

        self._t2s = OpenCC("t2s").convert
        self._client = httpx.Client(timeout=httpx.Timeout(self.settings.tts_breeze_timeout_s, connect=5))
        self.model_info = self._client.get(f"{self.url}/v1/model").raise_for_status().json()
        logger.warning("Breeze TTS 2 enabled (%s, digest %s). %s", self.url,
                       str(self.model_info.get("model_digest", ""))[:12], LICENCE.split(";")[0])

    def list_voices(self) -> list[VoiceInfo]:
        return [VoiceInfo(vid, p.get("language", "zh"), p.get("gender"),
                          p.get("instruction") or f"clone of {p.get('ref_audio')}")
                for vid, p in self._presets.items()]

    @property
    def default_voice(self) -> str:
        return next(iter(self._presets))

    def stream(self, text: str, voice: str, language: str | None, speed: float) -> Iterator[np.ndarray]:
        preset = self._presets[voice]
        chinese = (language or preset.get("language", "zh")).split("-")[0].lower() == "zh"
        convert = self._t2s if chinese and self._t2s else (lambda t: t)
        data = {"text": convert(text), "cfg_scale": str(preset.get("cfg_scale", 4))}
        if preset.get("seed") is not None:
            data["seed"] = str(preset["seed"])
        if preset.get("instruction"):
            data["instruction"] = convert(preset["instruction"])
        if preset.get("ref_text"):
            data["ref_text"] = preset["ref_text"]
        for attempt in range(self.settings.tts_breeze_busy_retries + 1):
            files = {"ref_audio": open(preset["ref_audio"], "rb")} if preset.get("ref_audio") else None  # noqa: SIM115
            try:
                with self._client.stream("POST", f"{self.url}/v1/audio/speech", data=data, files=files) as resp:
                    if resp.status_code == 409 and attempt < self.settings.tts_breeze_busy_retries:
                        time.sleep(0.2)                       # the server synthesizes one request at a time
                        continue
                    resp.raise_for_status()
                    carry = b""
                    for chunk in resp.iter_bytes():
                        chunk = carry + chunk
                        usable = len(chunk) - len(chunk) % 2
                        carry = chunk[usable:]
                        if usable:
                            yield np.frombuffer(chunk[:usable], dtype="<i2").astype(np.float32) / 32768.0
                    return
            finally:
                if files:
                    files["ref_audio"].close()

    def synthesize(self, text: str, voice: str, language: str | None, speed: float) -> np.ndarray:
        parts = list(self.stream(text, voice, language, speed))
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
