"""Bounded, paced session output with sequence numbers and replay."""
from __future__ import annotations

import asyncio
import json
import logging
import struct
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class OutboxOverflow(RuntimeError):
    pass


class Transport(Protocol):
    """The current WebSocket; both calls raise when the connection is gone."""

    async def send_text(self, text: str) -> None: ...

    async def send_bytes(self, data: bytes) -> None: ...


@dataclass
class _Frame:
    data: dict[str, Any] | bytes
    request_id: str | None = None
    audio_ms: float = 0.0
    seq: int = 0
    wire: str | bytes = b""


@dataclass
class Replay:
    frames: int
    gap: bool            # some frames after the client's last seq are no longer kept


class Outbox:
    def __init__(self, session_id: str, *, pace: bool = True, lead_ms: int = 500, queue_ms: int = 10000,
                 retain_bytes: int = 1 << 20, seq_prefix: bool = False,
                 on_playing: Callable[[], Awaitable[None]] | None = None,
                 on_idle: Callable[[], Awaitable[None]] | None = None,
                 max_frames: int = 256, wait_timeout_s: float = 10.0) -> None:
        self.session_id = session_id
        self.pace = pace
        self.lead_ms = max(0, lead_ms)
        self.queue_ms = max(self.lead_ms, queue_ms)
        if self.queue_ms != queue_ms:
            logger.warning("session %s: output queue raised from %d to %d ms to cover output lead",
                           session_id, queue_ms, self.queue_ms)
        self.max_frames = max(1, max_frames)
        self.wait_timeout_s = wait_timeout_s
        self.retain_bytes = retain_bytes
        self.seq_prefix = seq_prefix
        self.on_playing = on_playing
        self.on_idle = on_idle
        self.seq = 0                                  # last seq sent
        self._control: deque[_Frame] = deque()
        self._response: deque[_Frame] = deque()
        self._unsent_ms = 0.0
        self._sent: deque[_Frame] = deque()           # kept for replay
        self._sent_bytes = 0
        self._transport: Transport | None = None
        self._detached_at: float | None = time.monotonic()
        self._wake = asyncio.Event()
        self._progress = asyncio.Event()              # a frame was sent, or the connection dropped
        self._space = asyncio.Condition()
        self._sending = False
        # playback timeline
        self._origin: float | None = None             # monotonic time of position 0
        self._timeline_ms = 0.0                       # audio sent on this timeline
        self._starts: list[tuple[str | None, float]] = []   # (request_id, timeline offset) per answer
        self._ended: set[str | None] = set()          # answers on the timeline whose RESPONSE_END went out

    # ----- producers

    def event(self, event: dict[str, Any], *, priority: bool = False) -> None:
        """A control event; never waits."""
        if len(self._control) >= self.max_frames:
            if not priority:
                raise OutboxOverflow("control event queue is full")
            self._control.pop()
        if priority:
            self._control.appendleft(_Frame(event))
        else:
            self._control.append(_Frame(event))
        self._wake.set()

    async def response_event(self, event: dict[str, Any], request_id: str) -> None:
        async with self._space:
            await self._wait_for_space(audio=False)
            self._response.append(_Frame(event, request_id))
        self._wake.set()

    async def audio(self, data: bytes, audio_ms: float, request_id: str) -> None:
        """An answer's audio frame; waits while more than RT_OUTPUT_QUEUE_MS of audio is unsent."""
        async with self._space:
            await self._wait_for_space(audio=True)
            self._response.append(_Frame(data, request_id, audio_ms))
            self._unsent_ms += audio_ms
        self._wake.set()

    async def _wait_for_space(self, *, audio: bool) -> None:
        def ready():
            return len(self._response) < self.max_frames and (not audio or self._unsent_ms < self.queue_ms)

        if ready():
            return
        # Only a connected client that stops reading counts as stalled; while the session is
        # suspended the answer waits for RESUME (expiry cancels the producer).
        waiter = asyncio.create_task(self._space.wait_for(ready))
        stalled, step = 0.0, min(1.0, self.wait_timeout_s)
        try:
            while True:
                done, _ = await asyncio.wait({waiter}, timeout=step)
                if done:
                    waiter.result()
                    return
                stalled = stalled + step if self.attached else 0.0
                if stalled >= self.wait_timeout_s:
                    raise OutboxOverflow(f"output queue did not drain within {self.wait_timeout_s:g} s")
        finally:
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)

    # ----- barge-in

    def position_ms(self) -> float:
        if self._origin is None:
            return 0.0
        now = self._detached_at if self._detached_at is not None else time.monotonic()
        return min(self._timeline_ms, (now - self._origin) * 1000)

    def answer_position(self) -> tuple[str | None, int]:
        """(request_id, position in ms) of the answer the client is hearing now."""
        pos = self.position_ms()
        rid, offset = None, 0.0
        for start_rid, start in self._starts:
            if start <= pos:
                rid, offset = start_rid, start
        return rid, round(pos - offset)

    @property
    def playing(self) -> bool:
        return self._origin is not None

    def answer_ended(self, request_id: str | None) -> bool:
        """Whether RESPONSE_END of this answer has been sent."""
        return request_id in self._ended

    async def interrupt(self) -> None:
        """Barge-in: drop every unsent answer frame and end the timeline where the client is now
        (``playing`` is false from here on; ``on_idle`` is not called)."""
        async with self._space:
            self._response.clear()
            self._unsent_ms = 0.0
            self._space.notify_all()
        self._origin, self._timeline_ms, self._starts, self._ended = None, 0.0, [], set()
        self._wake.set()
        self._progress.set()

    async def drop_request(self, request_id: str) -> None:
        """SPEAK_CANCEL: drop the unsent frames of one answer; whatever else is queued stays."""
        async with self._space:
            kept = deque(f for f in self._response if f.request_id != request_id)
            self._unsent_ms = sum(f.audio_ms for f in kept)
            self._response = kept
            self._space.notify_all()
        self._wake.set()
        self._progress.set()

    @property
    def unsent_ms(self) -> float:
        return self._unsent_ms

    @property
    def queued(self) -> int:
        return len(self._control) + len(self._response)

    # ----- connection

    def attach(self, transport: Transport) -> None:
        if self._detached_at is not None and self._origin is not None:
            self._origin += time.monotonic() - self._detached_at     # nothing played meanwhile
        self._detached_at = None
        self._transport = transport
        self._wake.set()

    def detach(self) -> None:
        if self._detached_at is None:
            self._detached_at = time.monotonic()
        self._transport = None
        self._wake.set()
        self._progress.set()

    async def flush(self, played: bool = False) -> bool:
        """Wait until everything queued has been sent (``played``: and the answer audio has
        finished playing); False when the connection drops first."""
        while self.queued or self._sending or (played and self._origin is not None):
            if self._transport is None:
                return False
            self._progress.clear()
            await self._progress.wait()
        return True

    @property
    def attached(self) -> bool:
        return self._transport is not None

    async def resume(self, transport: Transport, last_seq: int,
                     resumed: Callable[[Replay], dict[str, Any]]) -> Replay:
        """RESUME on a new connection: send again every kept frame after ``last_seq`` (with its
        original seq), then the ``resumed(...)`` event, then carry on with what is queued.
        Raises if the new connection fails while replaying."""
        frames = [f for f in self._sent if f.seq > last_seq]
        oldest = self._sent[0].seq if self._sent else self.seq + 1
        replay = Replay(len(frames), last_seq < self.seq and oldest > last_seq + 1)
        for f in frames:
            if isinstance(f.wire, bytes):
                await transport.send_bytes(f.wire)
            else:
                await transport.send_text(f.wire)
        self.event(resumed(replay), priority=True)
        self.attach(transport)
        return replay

    # ----- sender

    async def run(self) -> None:
        """Send frames whenever a transport is attached; runs for the whole session."""
        while True:
            self._wake.clear()
            delay = await self._step()
            if delay == 0:
                continue
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass

    async def _step(self) -> float | None:
        """Send one frame if one may go now; else how long to wait (None = until woken)."""
        transport = self._transport
        if transport is None:
            return None
        if self._control:
            await self._deliver(transport, self._control.popleft())
            return 0
        if self._response:
            head = self._response[0]
            wait = self._pace_delay(head)
            if wait > 0:
                return wait
            frame = self._response.popleft()
            async with self._space:
                if frame.audio_ms:
                    self._unsent_ms = max(0.0, self._unsent_ms - frame.audio_ms)
                self._space.notify_all()
            await self._deliver(transport, frame)
            return 0
        if self._origin is not None:
            left = self._timeline_ms - self.position_ms()
            if left > 0:
                return left / 1000
            self._origin, self._timeline_ms, self._starts, self._ended = None, 0.0, [], set()
            self._progress.set()
            if self.on_idle is not None:
                await self.on_idle()
        return None

    async def _deliver(self, transport: Transport, frame: _Frame) -> None:
        self._sending = True
        try:
            await self._send(transport, frame)
        finally:
            self._sending = False
            self._progress.set()

    def _pace_delay(self, frame: _Frame) -> float:
        if not (self.pace and frame.audio_ms and self._origin is not None):
            return 0.0
        ahead = self._timeline_ms - self.position_ms()
        return max(0.0, (ahead - self.lead_ms) / 1000)

    async def _send(self, transport: Transport, frame: _Frame) -> None:
        self.seq += 1
        frame.seq = self.seq
        if isinstance(frame.data, bytes):
            frame.wire = (struct.pack("<I", frame.seq & 0xFFFFFFFF) + frame.data) if self.seq_prefix else frame.data
        else:
            frame.wire = json.dumps({**frame.data, "seq": frame.seq, "session_id": self.session_id},
                                    ensure_ascii=False, separators=(",", ":"))
            kind = frame.data.get("type")
            if kind == "RESPONSE_START":
                self._starts.append((frame.request_id, self._timeline_ms))
            elif kind == "RESPONSE_END":
                self._ended.add(frame.request_id)
        if frame.audio_ms:
            started = self._origin is None
            if started:
                self._origin, self._timeline_ms = time.monotonic(), 0.0
                self._starts = [(frame.request_id, 0.0)]
            self._timeline_ms += frame.audio_ms
            if started and self.on_playing is not None:
                await self.on_playing()
        self._keep(frame)
        try:
            if isinstance(frame.wire, bytes):
                await transport.send_bytes(frame.wire)
            else:
                await transport.send_text(frame.wire)
        except Exception:  # noqa: BLE001 -- any failure means the connection is gone
            # The frame is kept and goes out again after RESUME.
            if self._transport is transport:
                self.detach()
        self._progress.set()

    def _keep(self, frame: _Frame) -> None:
        self._sent.append(frame)
        self._sent_bytes += len(frame.wire)
        while self._sent_bytes > self.retain_bytes and len(self._sent) > 1:
            self._sent_bytes -= len(self._sent.popleft().wire)
