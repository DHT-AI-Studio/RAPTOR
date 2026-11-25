"""Resemble AI's Chatterbox Multilingual V3 (MIT), ported from the VIE01-419
evaluation spike (PR #378) as a permanent, side-by-side backend alongside
XTTS-v2 rather than a replacement for it.

t3_model="v3" (the recommended checkpoint) isn't on the chatterbox-tts==0.1.7
PyPI release's from_local()/from_pretrained() -- confirmed by reading the
installed package's actual signature, not the docs. Only GitHub's unreleased
master has it, which is why requirements-chatterbox.txt installs from a
pinned commit instead of a version; re-pin deliberately when a tagged
release exposes this.
"""
from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from app.errors import ServiceError
from app.services.voice.audio import Audio, language_code

CHATTERBOX_LANGUAGES = frozenset({"ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi",
                                  "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv",
                                  "sw", "tr", "zh"})


def chatterbox_language(language: str) -> str:
    """Map a BCP-47 locale to Chatterbox Multilingual's plain 2-letter code (no zh-cn-style
    region variant the way XTTS needs -- zh-TW and zh-CN both just read as "zh")."""
    primary = language_code(language)
    if primary not in CHATTERBOX_LANGUAGES:
        raise ServiceError(422, "unsupported_language", language=language,
                           supported=sorted(CHATTERBOX_LANGUAGES))
    return primary


_simplified_converter = None


def chatterbox_text(text: str, language: str) -> str:
    """Traditional and Simplified forms of the same character encode to disjoint
    Cangjie token sequences (confirmed against the model's own Cangjie5_TC.json), and
    live testing showed the Traditional side is badly undertrained -- garbled
    mispronunciations on Traditional input that vanished once converted to Simplified
    first. t2s (not tw2s): character-level only, so wording is never substituted."""
    if language != "zh":
        return text
    global _simplified_converter
    if _simplified_converter is None:
        from opencc import OpenCC
        _simplified_converter = OpenCC("t2s").convert
    return _simplified_converter(text)


class ChatterboxBackend:
    """Same-shape (encode/synthesize) backend as XttsBackend. Loads from a local
    checkpoint baked at build time (scripts/prefetch_voice.py's
    chatterbox_snapshot()), never downloads at request time -- same discipline
    as XttsBackend."""

    name = "chatterbox-multilingual-v3"

    def __init__(self, root: str) -> None:
        import torch
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        base = Path(root) / "chatterbox"
        if not (base / "t3_mtl23ls_v3.safetensors").exists():
            raise FileNotFoundError("Chatterbox Multilingual V3 weights missing")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        self.model = ChatterboxMultilingualTTS.from_local(base, "cuda", t3_model="v3")

    def prepare_language(self, language: str) -> str:
        return chatterbox_language(language)

    def prepare_text(self, text: str, language: str) -> str:
        return chatterbox_text(text, language)

    def encode(self, audio: Audio) -> bytes:
        """Extract conditioning once; Conditionals.save() is the model's own serializer
        (torch.save of a plain dataclass -- no custom pickling needed, same spirit as
        XttsBackend.encode()'s NumPy cache)."""
        with tempfile.NamedTemporaryFile(suffix=".wav") as source:
            source.write(audio.wav())
            source.flush()
            self.model.prepare_conditionals(source.name)
        buffer = io.BytesIO()
        self.model.conds.save(buffer)
        return buffer.getvalue()

    def synthesize(self, conditioning: bytes, text: str, language: str) -> bytes:
        """Reassign model.conds (shared, mutable instance state -- the caller MUST hold
        the same GPU-serializing lock XttsBackend relies on; Chatterbox has no internal
        thread-safety of its own) and generate. Every call is watermarked by the model
        itself (Resemble's PerTh), not something this service adds -- see
        _BACKEND_CTORS's watermarked=True in clone.py."""
        from chatterbox.mtl_tts import Conditionals

        self.model.conds = Conditionals.load(io.BytesIO(conditioning), map_location="cuda")
        waveform = self.model.generate(text, language_id=language).squeeze(0).cpu().numpy()
        waveform = np.asarray(waveform, dtype=np.float32)
        if not waveform.size or not np.isfinite(waveform).all():
            raise ServiceError(503, "clone_unavailable", reason="non-finite or empty synthesized audio")
        buffer = io.BytesIO()
        sf.write(buffer, waveform, self.model.sr, format="WAV", subtype="PCM_16")
        return buffer.getvalue()
