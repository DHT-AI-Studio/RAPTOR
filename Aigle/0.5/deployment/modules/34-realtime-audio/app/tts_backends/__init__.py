"""TTS backend registry."""
from __future__ import annotations

from importlib import import_module

from app.tts_backends.base import BackendUnavailable, TtsBackend, VoiceInfo

REGISTRY: dict[str, str] = {
    "kokoro-82m": "app.tts_backends.kokoro:KokoroBackend",
    "breeze-tts-2": "app.tts_backends.breeze:BreezeBackend",
}

__all__ = ["REGISTRY", "BackendUnavailable", "TtsBackend", "UnknownBackend", "VoiceInfo",
           "backend_class", "load_backend"]


class UnknownBackend(ValueError):
    pass


def backend_class(name: str) -> type[TtsBackend]:
    try:
        target = REGISTRY[name]
    except KeyError:
        raise UnknownBackend(f"unknown TTS backend {name!r}; available: {', '.join(sorted(REGISTRY))}") from None
    module, _, cls = target.partition(":")
    return getattr(import_module(module), cls)


def load_backend(name: str, settings) -> TtsBackend:
    """Resolve and load the active backend once at startup (raises UnknownBackend /
    BackendUnavailable, which stop the service)."""
    cls = backend_class(name)
    cls.check(settings)
    backend = cls(settings)
    backend.load()
    return backend
