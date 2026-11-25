"""RA-4: the interface every TTS backend implements (see app/tts_backends/__init__.py)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VoiceInfo:
    id: str
    language: str                   # BCP-47 primary language the voice speaks ("zh", "en")
    gender: str | None = None
    description: str = ""

    def public(self) -> dict:
        return {"id": self.id, "language": self.language, "gender": self.gender,
                "description": self.description}


class BackendUnavailable(Exception):
    """The backend cannot run here (missing package, weights, server or licence opt-in)."""


class TtsBackend(ABC):
    """One speech synthesizer.  ``load()`` runs once at startup; ``synthesize`` and ``stream``
    run in worker threads and return mono float32 audio in [-1, 1] at ``sample_rate``."""

    sample_rate: int = 24000

    def __init__(self, settings) -> None:
        self.settings = settings

    @classmethod
    @abstractmethod
    def name(cls) -> str: ...

    @classmethod
    def check(cls, settings) -> None:
        """Raise BackendUnavailable(reason) when this backend cannot be loaded; cheap (no
        model load), used by GET /tts/models for the backends that are not active."""

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def list_voices(self) -> list[VoiceInfo]: ...

    @property
    @abstractmethod
    def default_voice(self) -> str: ...

    @abstractmethod
    def synthesize(self, text: str, voice: str, language: str | None, speed: float) -> np.ndarray: ...

    def stream(self, text: str, voice: str, language: str | None, speed: float) -> Iterator[np.ndarray]:
        """Audio in pieces as it is produced; backends that synthesize incrementally override it."""
        yield self.synthesize(text, voice, language, speed)
