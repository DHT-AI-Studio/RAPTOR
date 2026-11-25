"""Synthesize response sentences into binary audio frames."""
from __future__ import annotations

import asyncio
import re
import unicodedata
from collections.abc import AsyncIterable, AsyncIterator
from contextlib import aclosing
from typing import Any, Protocol

import numpy as np

from app.codec import Encoder
from app.outbox import OutboxOverflow
from app.tts import TtsService


class ResponseLane(Protocol):
    """Where a response goes; app.outbox.Outbox in the service."""

    async def response_event(self, event: dict[str, Any], request_id: str) -> None: ...

    async def audio(self, data: bytes, audio_ms: float, request_id: str) -> None: ...


_URL = re.compile(r"https?://\S+")
_MARKUP = re.compile(r"\*\*|__|`+|^\s*#{1,6}\s*|^\s*>\s?|^\s*[-*•]\s+|(?<=\s)\*(?=\S)|(?<=\S)\*(?=\s|$)", re.MULTILINE)


def speakable(text: str) -> str:
    """Text as it should be spoken: no Markdown markers, links or emoji, one line."""
    text = _URL.sub("", text)
    text = _MARKUP.sub("", text)
    text = "".join(c for c in text if unicodedata.category(c) not in ("So", "Cs", "Co"))
    return re.sub(r"\s+", " ", text).strip()


def to_int16(audio: np.ndarray) -> np.ndarray:
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)


class Speaker:
    def __init__(self, tts: TtsService, encoder: Encoder, out: ResponseLane, language: str | None,
                 silence_pad_ms: int = 200) -> None:
        self.tts = tts
        self.encoder = encoder
        self.out = out
        self.language = language
        self.silence_pad_ms = silence_pad_ms

    async def speak(self, sentences: AsyncIterable[str], request_id: str, fallback: bool = False) -> dict[str, Any]:
        """Synthesize and queue ``sentences``; returns {duration_ms, sentences, started}.  An
        exception from ``sentences`` (e.g. the LLM timing out) still ends a started response
        with RESPONSE_END before it propagates; cancellation does not."""
        rate = self.tts.sample_rate
        state = {"started": False, "samples": 0, "sentences": 0}
        tag = {"request_id": request_id, **({"fallback": True} if fallback else {})}

        async def emit(frames: list[bytes]) -> None:
            if frames and not state["started"]:
                state["started"] = True
                await self.out.response_event({"type": "RESPONSE_START", **tag, "output": self.encoder.info()},
                                              request_id)
            for frame in frames:
                await self.out.audio(frame, self.encoder.duration_ms(frame), request_id)

        cancelled = False
        blocked = False
        try:
            async with aclosing(aiter(sentences)) as texts:
                async for raw in texts:
                    sentence = speakable(raw)
                    if not any(c.isalnum() for c in sentence):
                        continue
                    await self.out.response_event({"type": "RESPONSE_TEXT", **tag, "index": state["sentences"],
                                                   "text": sentence}, request_id)
                    async with aclosing(self.tts.stream(sentence, self.language)) as pieces:
                        async for piece in pieces:
                            pcm = to_int16(piece)
                            state["samples"] += len(pcm)
                            await emit(self.encoder.encode(pcm, rate))
                    state["sentences"] += 1
        except asyncio.CancelledError:
            cancelled = True
            raise
        except OutboxOverflow:
            blocked = True
            raise
        finally:
            if cancelled:
                self.encoder.flush()                                  # drop what is buffered
            elif state["started"] or state["samples"]:
                if not blocked:
                    pad = np.zeros(rate * self.silence_pad_ms // 1000, dtype=np.int16)
                    state["samples"] += len(pad)
                    try:
                        await emit(self.encoder.encode(pad, rate) + self.encoder.flush())
                    except OutboxOverflow:
                        blocked = True
                else:
                    self.encoder.flush()
                await self.out.response_event({"type": "RESPONSE_END", **tag,
                                               **({"interrupted": True, "reason": "output_blocked"} if blocked else {}),
                                               "duration_ms": round(state["samples"] * 1000 / rate),
                                               "sentences": state["sentences"]}, request_id)
            else:
                self.encoder.flush()
        return {"duration_ms": round(state["samples"] * 1000 / rate), "sentences": state["sentences"],
                "started": state["started"]}


async def one(text: str) -> AsyncIterator[str]:
    yield text
