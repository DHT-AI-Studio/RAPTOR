"""Module 15 streaming chat bridge and sentence splitting."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_TERMINATOR = re.compile(r"[。！？!?]+[」』”’）)]*|\.(?=\s|$)|…+")
_SOFT_BREAK = re.compile(r"[，、；,;：:]")
MIN_SENTENCE_CHARS = 2
LONG_RUN_CHARS = 60


class LlmError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class SentenceSplitter:
    def __init__(self) -> None:
        self._buf = ""

    def feed(self, text: str) -> list[str]:
        self._buf += text
        out: list[str] = []
        start = 0
        for match in _TERMINATOR.finditer(self._buf):
            piece = self._buf[start:match.end()].strip()
            if len(_letters(piece)) >= MIN_SENTENCE_CHARS:
                out.append(piece)
                start = match.end()
        self._buf = self._buf[start:]
        if len(self._buf) > LONG_RUN_CHARS:
            breaks = list(_SOFT_BREAK.finditer(self._buf))
            if breaks and breaks[-1].end() >= MIN_SENTENCE_CHARS:
                cut = breaks[-1].end()
                out.append(self._buf[:cut].strip())
                self._buf = self._buf[cut:]
        return out

    def flush(self) -> list[str]:
        rest, self._buf = self._buf.strip(), ""
        return [rest] if _letters(rest) else []


def _letters(text: str) -> str:
    return re.sub(r"[\W_]+", "", text)


class _Budget:
    """Time left for waiting on Module 15; only time actually spent awaiting it is charged.
    Until ``got_text()`` the first-token limit applies as well."""

    def __init__(self, seconds: float, first_s: float | None = None) -> None:
        self.left = seconds
        self.first_left = first_s

    @property
    def first_bound(self) -> bool:
        """No text yet, and the first-token limit is the one that runs out first."""
        return self.first_left is not None and self.first_left <= self.left

    def got_text(self) -> None:
        self.first_left = None

    async def wait(self, awaitable):
        limit = self.left if self.first_left is None else min(self.left, self.first_left)
        started = time.monotonic()
        try:
            return await asyncio.wait_for(awaitable, timeout=max(0.0, limit))
        finally:
            spent = time.monotonic() - started
            self.left -= spent
            if self.first_left is not None:
                self.first_left -= spent


class LlmBridge:
    # Module 15 appends every chat turn to Module 26 itself (chat_service.append_turn), so Module 34
    # must not archive its voice turns a second time (RTA-X6).
    archives_turns = True

    def __init__(self, base_url: str, timeout_s: float, stream: bool, client: httpx.AsyncClient,
                 first_token_timeout_s: float | None = None) -> None:
        self.url = base_url.rstrip("/") + "/chat"
        self.timeout_s = timeout_s
        self.first_token_timeout_s = first_token_timeout_s if first_token_timeout_s else None
        self.stream = stream
        self.client = client

    async def sentences(self, transcript: str, session_id: str, user_id: str,
                        request_id: str, guardrail_warning: bool = False) -> AsyncIterator[str]:
        budget = _Budget(self.timeout_s, self.first_token_timeout_s)
        body = {"message": transcript, "session_id": session_id, "user_id": user_id, "stream": self.stream}
        headers = {"X-Request-ID": request_id, "Accept": "text/event-stream" if self.stream else "application/json"}
        if guardrail_warning:
            headers["X-Guardrail-Warning"] = "true"          # RTA-X6: Module 23 warned on the transcript
        splitter = SentenceSplitter()
        request = self.client.build_request("POST", self.url, json=body, headers=headers,
                                            timeout=httpx.Timeout(self.timeout_s, connect=5))
        try:
            resp = await budget.wait(self.client.send(request, stream=True))
            try:
                if resp.status_code >= 400:
                    raise LlmError("LLM_UNAVAILABLE", f"Module 15 answered {resp.status_code}")
                if resp.headers.get("content-type", "").startswith("text/event-stream"):
                    ended = False
                    async for event, data in _sse(resp, budget):
                        if event == "delta":
                            if data.get("text"):
                                budget.got_text()
                            for s in splitter.feed(data.get("text", "")):
                                yield s
                        elif event == "start" and data.get("request_id") not in (None, request_id):
                            logger.info("request %s is Module 15 request %s", request_id, data["request_id"])
                        elif event == "error":
                            raise LlmError("LLM_UNAVAILABLE", str(data.get("message") or "Module 15 error"))
                        elif event == "end":
                            ended = True
                            break
                    if not ended:
                        raise LlmError("LLM_UNAVAILABLE", "Module 15 stream ended before its end event")
                else:
                    raw = await budget.wait(resp.aread())
                    budget.got_text()
                    for s in splitter.feed(json.loads(raw).get("response", "")):
                        yield s
            finally:
                await resp.aclose()
        except (asyncio.TimeoutError, httpx.TimeoutException) as exc:
            if budget.first_bound:
                raise LlmError("LLM_TIMEOUT", "no answer text from Module 15 within "
                                              f"{self.first_token_timeout_s:g} s") from exc
            raise LlmError("LLM_TIMEOUT", f"no answer from Module 15 within {self.timeout_s:g} s") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise LlmError("LLM_UNAVAILABLE", f"Module 15 request failed: {exc}") from exc
        for s in splitter.flush():
            yield s


class Prefetch:
    """Reads ``source`` ahead into a queue of ``maxsize`` items (RTA-X4: RT_TTS_QUEUE_MAX answer
    sentences wait for synthesis).  Errors of ``source`` are raised where the item would have
    been; closing (or cancelling) the iteration stops the reader and closes ``source``."""

    _END = object()

    def __init__(self, source: AsyncIterator[str], maxsize: int) -> None:
        self._source = source
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max(1, maxsize))

    @property
    def depth(self) -> int:
        return self._queue.qsize()

    async def _read(self) -> None:
        try:
            async for item in self._source:
                await self._queue.put((item, None))
        except Exception as exc:  # noqa: BLE001 -- raised to the consumer
            await self._queue.put((None, exc))
            return
        await self._queue.put((self._END, None))

    async def __aiter__(self) -> AsyncIterator[str]:
        reader = asyncio.create_task(self._read())
        try:
            while True:
                item, exc = await self._queue.get()
                if exc is not None:
                    raise exc
                if item is self._END:
                    return
                yield item
        finally:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
            await self._source.aclose()


async def _sse(resp: httpx.Response, budget: _Budget) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """Server-Sent Events of ``resp`` as (event, JSON data); each read is charged to ``budget``."""
    lines = resp.aiter_lines()
    event, data = "message", []
    while True:
        try:
            line = await budget.wait(anext(lines))
        except StopAsyncIteration:
            if data or event != "message":
                raise LlmError("LLM_UNAVAILABLE", "Module 15 stream ended inside an SSE event")
            return
        if line == "":
            if data:
                yield event, json.loads("\n".join(data))
            event, data = "message", []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
