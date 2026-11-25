"""INF-13: local XTTS-v2 synthesis using cached conditioning tensors.

Delete this whole file (plus its _BACKEND_CTORS entry in clone.py and
requirements-xtts.txt) to drop XTTS-v2 entirely -- nothing else in the
service names this module.
"""
from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from app.errors import ServiceError
from app.services.voice.audio import Audio, language_code

XTTS_LANGUAGES = frozenset({"en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl",
                            "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"})


def xtts_language(language: str) -> str:
    """Map supported BCP-47 locales to XTTS's trained language codes."""
    primary = language_code(language)
    code = "zh-cn" if primary == "zh" else primary
    if code not in XTTS_LANGUAGES:
        raise ServiceError(422, "unsupported_language", language=language,
                           supported=sorted(XTTS_LANGUAGES))
    return code


class XttsBackend:
    """Load a local checkpoint, never download at request time."""

    name = "xtts-v2"

    def __init__(self, root: str) -> None:
        import torch
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        base = Path(root) / "xtts"
        if not (base / "model.pth").exists():
            raise FileNotFoundError("XTTS-v2 weights missing")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        config = XttsConfig()
        config.load_json(str(base / "config.json"))
        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(config, checkpoint_dir=str(base), eval=True)
        # Audio preprocessing and normalization retain float32 buffers. Casting
        # the entire model to half breaks Coqui's float32 waveform encoder.
        self.model.cuda()
        self.model.eval()  # Coqui's override returns None, unlike torch.nn.Module.eval.

    def prepare_language(self, language: str) -> str:
        return xtts_language(language)

    def prepare_text(self, text: str, language: str) -> str:
        return text

    def encode(self, audio: Audio) -> bytes:
        """Extract conditioning once; cache NumPy arrays, never pickle objects."""
        import torch

        with tempfile.NamedTemporaryFile(suffix=".wav") as source:
            source.write(audio.wav())
            source.flush()
            with torch.inference_mode():
                latent, embedding = self.model.get_conditioning_latents(audio_path=[source.name])
        if not torch.isfinite(latent).all() or not torch.isfinite(embedding).all():
            raise ServiceError(503, "clone_unavailable", reason="non-finite speaker conditioning")
        buffer = io.BytesIO()
        np.savez_compressed(buffer, latent=latent.detach().cpu().float().numpy(),
                            embedding=embedding.detach().cpu().float().numpy())
        return buffer.getvalue()

    def synthesize(self, conditioning: bytes, text: str, language: str) -> bytes:
        """Use CUDA float16 autocast and encode the 24 kHz WAV output."""
        import torch

        with np.load(io.BytesIO(conditioning), allow_pickle=False) as arrays:
            latent = torch.as_tensor(arrays["latent"], device="cuda", dtype=torch.float32)
            embedding = torch.as_tensor(arrays["embedding"], device="cuda", dtype=torch.float32)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            output = self.model.inference(text, language, latent, embedding, enable_text_splitting=True)
        waveform = output["wav"]
        if hasattr(waveform, "detach"):
            waveform = waveform.detach().cpu().float().numpy()
        waveform = np.asarray(waveform, dtype=np.float32)
        if not waveform.size or not np.isfinite(waveform).all():
            raise ServiceError(503, "clone_unavailable", reason="non-finite or empty synthesized audio")
        buffer = io.BytesIO()
        sf.write(buffer, waveform, 24000, format="WAV", subtype="PCM_16")
        return buffer.getvalue()
