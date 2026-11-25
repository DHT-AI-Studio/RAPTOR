"""In-process Kokoro TTS backend."""
from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from app.tts_backends.base import BackendUnavailable, TtsBackend, VoiceInfo

logger = logging.getLogger(__name__)

_REPO_ID = "hexgrad/Kokoro-82M-v1.1-zh"    # only labels the local snapshot; nothing is downloaded


def _speed_fn(base: float):
    """The model card's mitigation: v1.1-zh rushes past ~100 phonemes (little training data
    there), so long inputs are slowed down piecewise-linearly."""

    def speed(len_ps: int) -> float:
        if len_ps <= 83:
            factor = 1.0
        elif len_ps < 183:
            factor = 1 - (len_ps - 83) / 500
        else:
            factor = 0.8
        return factor * 1.1 * base

    return speed


class KokoroBackend(TtsBackend):
    sample_rate = 24000

    def __init__(self, settings) -> None:
        super().__init__(settings)
        self.dir = Path(settings.tts_kokoro_dir)
        self._lock = threading.Lock()
        self._voices: dict[str, VoiceInfo] = {}
        self._zh = self._en = None
        self._t2s = None

    @classmethod
    def name(cls) -> str:
        return "kokoro-82m"

    @classmethod
    def check(cls, settings) -> None:
        try:
            import kokoro  # noqa: F401
            import misaki  # noqa: F401
        except ImportError as exc:
            raise BackendUnavailable(f"package missing: {exc.name}") from exc
        directory = Path(settings.tts_kokoro_dir)
        if not (directory / "config.json").is_file() or not list(directory.glob("*.pth")):
            raise BackendUnavailable(f"no Kokoro weights (config.json, *.pth) in {directory}")
        if not list((directory / "voices").glob("*.pt")):
            raise BackendUnavailable(f"no voices in {directory / 'voices'}")

    def load(self) -> None:
        import torch
        from kokoro import KModel, KPipeline
        from opencc import OpenCC

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        device = self.settings.tts_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        weights = min(self.dir.glob("*.pth"))
        model = KModel(repo_id=_REPO_ID, config=str(self.dir / "config.json"), model=str(weights)).to(device).eval()
        english = KPipeline(lang_code="a", repo_id=_REPO_ID, model=False)

        def en_callable(text: str) -> str:
            return next(english(text)).phonemes

        self._zh = KPipeline(lang_code="z", repo_id=_REPO_ID, model=model, en_callable=en_callable)
        self._en = KPipeline(lang_code="a", repo_id=_REPO_ID, model=model)
        self._t2s = OpenCC("t2s").convert
        for path in sorted((self.dir / "voices").glob("*.pt")):
            vid = path.stem
            language = "zh" if vid[0] == "z" else "en"
            gender = {"f": "female", "m": "male"}.get(vid[1:2])
            self._voices[vid] = VoiceInfo(vid, language, gender)
        logger.info("Kokoro %s loaded on %s (%d voices)", weights.name, device, len(self._voices))

    def list_voices(self) -> list[VoiceInfo]:
        return list(self._voices.values())

    @property
    def default_voice(self) -> str:
        return "zf_001" if "zf_001" in self._voices else next(iter(self._voices))

    def stream(self, text: str, voice: str, language: str | None, speed: float) -> Iterator[np.ndarray]:
        english = (language or "").split("-")[0].lower() == "en"
        pipeline = self._en if english else self._zh
        if not english:
            text = self._t2s(text)
        voice_path = str(self.dir / "voices" / f"{voice}.pt")
        with self._lock:          # one pipeline: G2P state and the model are not shared across threads
            for result in pipeline(text, voice=voice_path, speed=_speed_fn(speed)):
                if result.audio is not None:
                    yield result.audio.detach().cpu().numpy().astype(np.float32)

    def synthesize(self, text: str, voice: str, language: str | None, speed: float) -> np.ndarray:
        parts = list(self.stream(text, voice, language, speed))
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
