"""Voice-clone synthesis shared across backends: cache admission, GPU-serialized
lazy loading, and backend-agnostic dispatch. Each backend (XttsBackend,
ChatterboxBackend, ...) owns its own language mapping, text normalization,
conditioning format and sample rate -- see app/services/voice/backends/.
"""
from __future__ import annotations

import asyncio
import functools
import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from app.errors import ServiceError
from app.services.model_registry import ModelEntry
from app.services.voice.audio import Audio
from app.services.voice.backends.chatterbox import ChatterboxBackend
from app.services.voice.backends.xtts import XttsBackend
from app.services.voice.gpu_lock import GpuLock
from app.services.voice.speaker_cache import SpeakerCache


class CloneBackend(Protocol):
    name: str

    def prepare_language(self, language: str) -> str:
        """BCP-47 -> this backend's trained code; ServiceError(422, ...) if unsupported."""

    def prepare_text(self, text: str, language: str) -> str:
        """Backend-specific text normalization (identity if none needed)."""

    def encode(self, audio: Audio) -> bytes:
        """Blocking. Backend owns its own conditioning serialization format."""

    def synthesize(self, conditioning: bytes, text: str, language: str) -> bytes:
        """Blocking. Returns a complete WAV file; backend owns its own sample rate."""


# The one place in the service that maps a backend name to its constructor and
# whether it watermarks output. Delete a row (plus its backend module and
# requirements file) to drop that backend entirely -- nothing else names it.
_BACKEND_CTORS: dict[str, tuple[type, float | None, bool]] = {
    "xtts-v2": (XttsBackend, 1.8, False),
    "chatterbox-multilingual-v3": (ChatterboxBackend, 1.0, True),
}


def build_clone_backend_specs(model_dir: str, enabled: frozenset[str]) -> dict[str, CloneBackendSpec]:
    specs = {}
    for name in enabled:
        if name not in _BACKEND_CTORS:
            raise ValueError(f"unknown clone backend '{name}' in CLONE_BACKENDS_RUNTIME")
        ctor, size_gb, watermarked = _BACKEND_CTORS[name]
        specs[name] = CloneBackendSpec(name=name, loader=functools.partial(ctor, model_dir),
                                        size_gb=size_gb, watermarked=watermarked)
    return specs


@dataclass
class CloneBackendSpec:
    name: str
    loader: Callable[[], CloneBackend]   # blocking; e.g. functools.partial(XttsBackend, model_dir)
    size_gb: float | None = None
    watermarked: bool = False
    instance: CloneBackend | None = None
    error: str | None = None


class Cloning:
    """Lazily loads named backends, serializes cache admission, and shares CUDA
    execution (and construction) with all voice tasks via the one GpuLock."""

    def __init__(self, gpu: GpuLock, specs: dict[str, CloneBackendSpec] | None = None,
                 disabled_reason: str | None = None) -> None:
        self.gpu = gpu
        self.specs = specs or {}
        self.disabled_reason = disabled_reason
        self.requests = asyncio.Lock()                                    # cache-admission lock
        self._load_locks = {name: asyncio.Lock() for name in self.specs}  # per-backend load lock

    def configured_backends(self) -> list[str]:
        return list(self.specs)

    def loaded_backends(self) -> list[str]:
        return [name for name, spec in self.specs.items() if spec.instance is not None]

    def builtin_model_entries(self) -> list[ModelEntry]:
        """Feeds ModelRegistry's static_entries source. Empty when clone is
        policy-disabled, so GET /models never lists something /clone will 403 on."""
        if self.disabled_reason is not None:
            return []
        return [
            ModelEntry(name=name, type="clone", size_gb=spec.size_gb, capabilities=["clone"],
                       source="builtin", status="loaded" if spec.instance is not None else "available",
                       watermarked=spec.watermarked)
            for name, spec in self.specs.items()
        ]

    async def _get_backend(self, name: str) -> CloneBackend:
        spec = self.specs.get(name)
        if spec is None:
            raise ServiceError(503, "clone_unavailable", reason=f"backend '{name}' not configured")
        if spec.instance is not None:
            return spec.instance
        async with self._load_locks.setdefault(name, asyncio.Lock()):
            if spec.instance is not None:          # loaded by a concurrent request while we waited
                return spec.instance
            try:
                spec.instance = await self.gpu.run(spec.loader)
            except Exception as exc:
                spec.error = str(exc)
                raise ServiceError(503, "clone_unavailable", reason=f"{name} model not loaded: {exc}") from exc
            return spec.instance

    async def preload(self, name: str) -> None:
        """For settings.clone_backends_warm -- eagerly pay the load cost once."""
        await self._get_backend(name)

    async def clone(
        self, audio: Audio, text: str, language: str, cache: SpeakerCache, backend_name: str,
        speaker_name: str | None = None,
    ) -> tuple[bytes, str]:
        """Return WAV and stable speaker ID; cache hits skip the encoder."""
        if self.disabled_reason is not None:
            # VIE01-419: off by policy (CLONE_ENABLED unset), not a load failure --
            # checked before validating input, since none of that matters here.
            raise ServiceError(403, "clone_disabled", reason=self.disabled_reason)
        if not 1 <= audio.duration <= 60:
            raise ServiceError(400, "invalid_reference_duration", min_seconds=1, max_seconds=60)
        if not text.strip() or len(text) > 3000:
            raise ServiceError(400, "invalid_text", max_chars=3000)
        backend = await self._get_backend(backend_name)
        lang = backend.prepare_language(language)
        text = backend.prepare_text(text, lang)
        speaker_id = speaker_name if speaker_name is not None else hashlib.sha256(audio.original).hexdigest()
        # Qualify the cache key by backend only when 2+ are configured: a bare
        # speaker_id cached by one backend and fed to another's deserializer is a
        # real format-confusion bug (NumPy .npz vs. torch.save), not just quality
        # loss -- zero behavior/format change for the common single-backend case.
        cache_id = f"{backend_name}.{speaker_id}" if len(self.specs) > 1 else speaker_id
        cache.key(cache_id)
        async with self.requests:
            conditioning = await cache.get(cache_id)
            if conditioning is None:
                conditioning = await self.gpu.run(backend.encode, audio)
                await cache.put(cache_id, conditioning)
            return await self.gpu.run(backend.synthesize, conditioning, text, lang), cache_id


def _host_mem_available_bytes() -> int | None:
    """Linux's MemAvailable (free + reclaimable buff/cache); None when unreadable."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def vram_warning() -> list[str]:
    """Warn when currently free VRAM is below 12 GiB; remain usable without torch.

    On a unified-memory GPU (`is_integrated`, e.g. GX10/Grace-Blackwell, where
    CPU and GPU share one physical pool) `cudaMemGetInfo` reports the host's
    raw MemFree, not MemAvailable -- it doesn't know Linux's reclaimable
    buff/cache can be freed on demand, which can make this warn even when
    there's plenty of real headroom. Same fix as model_resource_manager.py's
    gpu_free_bytes() in 23-guardrail-service, confirmed live there on GX10
    (free -h: 2.1GiB free / 67GiB available). Doesn't affect a discrete GPU
    (`is_integrated` is False there; VRAM has no page-cache concept at all,
    so this never fires)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return ["CUDA unavailable; GPU voice inference unavailable"]
        free, _ = torch.cuda.mem_get_info()
        if getattr(torch.cuda.get_device_properties(0), "is_integrated", False):
            host_available = _host_mem_available_bytes()
            if host_available is not None:
                free = max(free, host_available)
        return ["Available GPU VRAM is below 12 GiB"] if free < 12 * 1024**3 else []
    except (ImportError, RuntimeError):
        return ["GPU VRAM check unavailable"]


def gpu_available() -> bool:
    """INF-1's /health contract. The text profile never installs torch at all."""
    try:
        import torch
        return torch.cuda.is_available()
    except (ImportError, RuntimeError):
        return False
