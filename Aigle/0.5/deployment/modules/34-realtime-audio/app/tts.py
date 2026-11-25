"""Load a TTS backend and bound concurrent synthesis."""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import numpy as np

from app.tts_backends import (
    REGISTRY,
    BackendUnavailable,
    TtsBackend,
    backend_class,
    load_backend,
)

logger = logging.getLogger(__name__)

_DONE = object()


class TtsService:
    def __init__(self, settings, loader: Callable[[str, Any], TtsBackend] = load_backend) -> None:
        self.settings = settings
        self._loader = loader
        self.backend: TtsBackend | None = None
        self.voice = ""
        self.slots = max(1, settings.tts_max_concurrent)
        self._sem = asyncio.Semaphore(self.slots)
        self.busy = 0                                 # slots in use
        self.waiting = 0                              # sentences waiting for a slot
        self._models_cache: list[dict[str, Any]] = []
        self._models_checked_at = 0.0
        self._models_lock = asyncio.Lock()

    async def start(self) -> None:
        self.backend = await asyncio.to_thread(self._loader, self.settings.tts_model, self.settings)
        self._models_cache = []
        voices = {v.id for v in self.backend.list_voices()}
        wanted = self.settings.tts_voice
        if wanted and wanted not in voices:
            logger.warning("RT_TTS_VOICE %r is not a %s voice; using %s", wanted, self.backend.name(),
                           self.backend.default_voice)
            wanted = ""
        self.voice = wanted or self.backend.default_voice
        # Warm-up: the first synthesis loads G2P dictionaries and CUDA kernels.
        await asyncio.to_thread(self.backend.synthesize, "您好。", self.voice, "zh-TW", 1.0)
        logger.info("TTS %s ready, voice %s", self.backend.name(), self.voice)

    @property
    def sample_rate(self) -> int:
        assert self.backend is not None
        return self.backend.sample_rate

    def info(self) -> dict[str, Any]:
        if self.backend is None:
            return {"loaded": False}
        return {"model": self.backend.name(), "voice": self.voice, "sample_rate": self.backend.sample_rate}

    async def stream(self, text: str, language: str | None) -> AsyncIterator[np.ndarray]:
        """Float32 audio pieces of ``text`` as the backend produces them."""
        assert self.backend is not None
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        stop = threading.Event()
        backend, voice, speed = self.backend, self.voice, self.settings.tts_speed

        def produce() -> None:                       # worker thread
            gen = backend.stream(text, voice, language, speed)
            try:
                for piece in gen:
                    if stop.is_set():
                        break
                    loop.call_soon_threadsafe(queue.put_nowait, piece)
            except Exception as exc:  # noqa: BLE001 -- handed to the consumer and raised there
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                gen.close()
                loop.call_soon_threadsafe(queue.put_nowait, _DONE)

        self.waiting += 1
        try:
            await self._sem.acquire()
        finally:
            self.waiting -= 1
        self.busy += 1

        def release(_future) -> None:
            self.busy -= 1
            self._sem.release()

        worker = loop.run_in_executor(None, produce)
        worker.add_done_callback(release)
        try:
            while True:
                item = await queue.get()
                if item is _DONE:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            stop.set()            # cancelled (e.g. barge-in): the thread stops at its next piece
            if worker.done() and not worker.cancelled():
                worker.result()

    async def models(self) -> list[dict[str, Any]]:
        """GET /tts/models: every registered backend, loaded / available / unavailable."""
        async with self._models_lock:
            if not self._models_cache or time.monotonic() - self._models_checked_at >= 60:
                self._models_cache = await self._check_models()
                self._models_checked_at = time.monotonic()
            return self._models_cache

    async def _check_models(self) -> list[dict[str, Any]]:
        out = []
        for name in REGISTRY:
            active = self.backend is not None and self.backend.name() == name
            entry: dict[str, Any] = {"name": name, "active": active}
            if active:
                entry.update(status="loaded", voice=self.voice, sample_rate=self.backend.sample_rate,
                             voices=[v.public() for v in self.backend.list_voices()])
            else:
                try:
                    await asyncio.to_thread(backend_class(name).check, self.settings)
                    entry["status"] = "available"
                except BackendUnavailable as exc:
                    entry.update(status="unavailable", reason=str(exc))
                except ImportError as exc:
                    entry.update(status="unavailable", reason=f"package missing: {exc.name}")
            out.append(entry)
        return out
