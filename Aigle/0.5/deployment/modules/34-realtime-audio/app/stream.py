"""WebSocket session pipeline, backpressure, barge-in and resume."""
from __future__ import annotations

import asyncio
import logging
import os
import struct
import time
import uuid
from collections import Counter
from contextlib import aclosing, suppress
from dataclasses import dataclass, field
from itertools import groupby
from typing import Any

import numpy as np
from fastapi import WebSocket
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocketState

from app.audio import DecodeError, make_decoder
from app.codec import make_encoder
from app.config import Settings
from app.guardrail import Guardrail
from app.integrations import MemoryArchive, TranscriptIndexer
from app.llm_bridge import LlmBridge, LlmError, Prefetch, SentenceSplitter
from app.metrics import LatencyHistogram
from app.outbox import Outbox, OutboxOverflow, Replay
from app.protocol import (
    CLOSE_EXPIRED,
    CLOSE_FORBIDDEN,
    CLOSE_INTERNAL,
    CLOSE_NORMAL,
    CLOSE_PROTOCOL,
    CLOSE_REPLACED,
    CLOSE_TIMEOUT,
    ProtocolError,
    ResumeRequest,
    StreamConfig,
    is_end,
    parse_config,
    parse_json_text,
    parse_resume,
)
from app.recording import ConsentChecker, Recorder
from app.response import Speaker, one
from app.session import (
    ENDED,
    EXPIRED,
    LISTENING,
    PROCESSING,
    RESPONDING,
    TERMINAL,
    TOO_MANY_SESSIONS,
    InvalidTransition,
    SessionState,
    SessionStore,
    TicketStore,
)
from app.stt import RollingParams, SttService, SttSession, Transcript
from app.tts import TtsService
from app.vad import (
    Discarded,
    Segmenter,
    SpeechStart,
    Utterance,
    VadModelPool,
    VadParams,
)

logger = logging.getLogger(__name__)

_END, _DISCONNECT = "end", "disconnect"

ROLE_EXTERNAL_BRAIN, ROLE_DELEGATE = "rt-external-brain", "rt-delegate"
SESSION_ROLES = (ROLE_EXTERNAL_BRAIN, ROLE_DELEGATE)    # realm roles kept in the WS ticket


@dataclass
class _Speech:
    """One SPEAK utterance (RTA-X9): its texts as they arrive, None once final."""

    id: str
    texts: asyncio.Queue = field(default_factory=asyncio.Queue)
    final: bool = False
    cancelled: bool = False
    spoken: list[str] = field(default_factory=list)


class UtteranceSink:
    """Where finished utterances go (the STT hand-off).  The default only logs; a
    speech-to-text integration (e.g. Module 07 /v1/audio/transcriptions with
    ``utterance.wav()``) plugs in here."""

    async def handle(self, session_id: str, config: StreamConfig, utterance: Utterance) -> None:
        logger.info("session %s utterance %d: %d ms (%s)", session_id, utterance.index,
                    utterance.duration_ms, utterance.reason)


class WavDumpSink(UtteranceSink):
    """Debug sink (RT_UTTERANCE_DUMP_DIR): writes each utterance as a 16 kHz WAV file."""

    def __init__(self, directory: str) -> None:
        self._dir = directory
        os.makedirs(directory, exist_ok=True)

    async def handle(self, session_id: str, config: StreamConfig, utterance: Utterance) -> None:
        path = os.path.join(self._dir, f"{session_id}-{utterance.index:04d}.wav")
        await asyncio.to_thread(_write, path, utterance.wav())


def _write(path: str, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)


async def deny(ws: WebSocket, status: int, error: str, **extra: Any) -> None:
    """Reject the Upgrade with a real HTTP status (ASGI websocket denial-response extension);
    servers without the extension can only answer 403."""
    response = JSONResponse({"error": error, **extra}, status_code=status)
    # uvicorn's websockets-based implementations add their own Content-Length / Content-Type to
    # denial responses; sending ours too yields duplicate Content-Length, which strict clients
    # (e.g. the Python websockets library) reject as an invalid HTTP response.
    response.raw_headers = [
        (k, v) for k, v in response.raw_headers if k not in (b"content-length", b"content-type")
    ]
    try:
        await ws.send_denial_response(response)
    except RuntimeError:
        await ws.close(code=1008)


class StreamHandler:
    def __init__(
        self,
        settings: Settings,
        tickets: TicketStore,
        sessions: SessionStore,
        vad_pool: VadModelPool,
        sink: UtteranceSink,
        stt: SttService | None = None,
        tts: TtsService | None = None,
        llm: LlmBridge | None = None,
        *,
        guardrail: Guardrail | None = None,
        indexer: TranscriptIndexer | None = None,
        memory: MemoryArchive | None = None,
        consent: ConsentChecker | None = None,
        recorder_factory=None,
    ) -> None:
        self.settings = settings
        self.tickets = tickets
        self.sessions = sessions
        self.vad_pool = vad_pool
        self.sink = sink
        self.stt = stt
        self.tts = tts
        self.llm = llm
        self.guardrail = guardrail                     # RTA-X6: Module 23
        self.indexer = indexer                         # RTA-X6: Module 25 (Kafka)
        self.memory = memory                           # RTA-X6: Module 26, for brains that do not archive
        self.consent = consent                         # RTA-X7: Module 33
        self.recorder_factory = recorder_factory       # (session_id, user_id) -> Recorder
        self.rolling = RollingParams(stride_s=settings.stt_stride_s, first_decode_ms=settings.stt_first_decode_ms,
                                     enabled=settings.stt_streaming)
        self.params = VadParams(
            threshold=settings.vad_threshold,
            min_silence_ms=settings.vad_min_silence_ms,
            min_speech_ms=settings.vad_min_speech_ms,
            max_utterance_s=settings.max_utterance_s,
            speech_pad_ms=settings.vad_speech_pad_ms,
            barge_in_threshold=settings.barge_in_threshold,
        )
        self.live: dict[str, Connection] = {}          # sessions of this process (for RESUME, metrics)
        self.stats: Counter[str] = Counter()            # RTA-X4 counters (app/metrics.py)
        self.stt_latency = LatencyHistogram()           # STT finalize() latency (app/metrics.py)

    async def handle(self, ws: WebSocket, token: str | None, resume_id: str | None = None) -> None:
        # Validate before accepting: a missing/expired/used ticket never upgrades.
        if not await self.tickets.exists(token):
            await deny(ws, 401, "invalid_ticket", message="ticket missing, expired or already used")
            return
        if resume_id:
            # The session being resumed already holds a slot.
            ticket = await self.tickets.consume(token)
            if ticket is None:
                await deny(ws, 401, "invalid_ticket", message="ticket already used")
                return
            await ws.accept()
            await self._resume_connection(ws, str(ticket.get("user_id", "")), resume_id)
            return
        # Peek (not consume) the ticket for its user_id: capacity is checked per-user as well as
        # service-wide now, and a denial here must not burn the caller's one-time ticket.
        peeked = await self.tickets.peek(token)
        user_id = str((peeked or {}).get("user_id", ""))
        session_id, reason = await self.sessions.acquire_slot(user_id)
        if session_id is None:
            extra = {"max_sessions": self.settings.max_sessions} if reason == TOO_MANY_SESSIONS \
                else {"max_sessions_per_user": self.settings.max_sessions_per_user}
            await deny(ws, 503, reason, **extra)
            return
        try:
            ticket = await self.tickets.consume(token)   # atomic: a concurrent reuse loses here
            if ticket is None:
                await deny(ws, 401, "invalid_ticket", message="ticket already used")
                return
            await self._run(ws, session_id, str(ticket.get("user_id", "")), ticket.get("roles") or [])
        finally:
            # Shielded so a cancelled connection (server shutdown) still frees its slot.
            await asyncio.shield(self.sessions.release_slot(session_id, user_id))

    async def serve(self, ws: Any, user_id: str, roles: list[str] | tuple[str, ...] = (), *,
                    resumable: bool = True, ping: bool = True) -> bool | str:
        """A session on a connection authenticated elsewhere (RA-6 gRPC: the JWT, no ticket).
        ``ws`` is anything with the WebSocket calls Connection uses; ``resumable`` / ``ping``
        False when the transport has no RESUME / PING of its own. Returns the denial reason
        (TOO_MANY_SESSIONS / TOO_MANY_SESSIONS_FOR_USER) when every slot -- global or this
        user's own -- is taken, else True."""
        session_id, reason = await self.sessions.acquire_slot(user_id)
        if session_id is None:
            return reason
        try:
            await self._run(ws, session_id, user_id, roles, resumable=resumable, ping=ping)
        finally:
            await asyncio.shield(self.sessions.release_slot(session_id, user_id))
        return True

    async def _run(self, ws: Any, session_id: str, user_id: str, roles: list[str] | tuple[str, ...],
                   resumable: bool = True, ping: bool = True) -> None:
        await self.sessions.register(session_id, user_id)
        await ws.accept()
        conn = Connection(self, session_id, user_id, roles)
        conn.resume_allowed, conn.ping_enabled = resumable, ping
        self.live[session_id] = conn
        try:
            await conn.run(ws)
        finally:
            self.live.pop(session_id, None)

    async def _resume_connection(self, ws: WebSocket, user_id: str, resume_id: str) -> None:
        """?session_id= connection: the first frame must be RESUME for that session."""
        try:
            first = await asyncio.wait_for(ws.receive(), timeout=self.settings.config_timeout_s)
        except asyncio.TimeoutError:
            await _close_direct(ws, CLOSE_TIMEOUT, "config_timeout", "no RESUME frame received in time")
            return
        if first["type"] == "websocket.disconnect":
            return
        try:
            request = parse_resume(first)
        except ProtocolError as exc:
            await _close_direct(ws, CLOSE_PROTOCOL, exc.code, exc.message)
            return
        if request is None or request.session_id != resume_id:
            await _close_direct(ws, CLOSE_PROTOCOL, "resume_required",
                                "with ?session_id= the first frame must be RESUME for that session")
            return
        await self.resume(ws, user_id, request)

    async def resume(self, ws: WebSocket, user_id: str, request: ResumeRequest) -> None:
        """Hand ``ws`` to the suspended (or still connected) session; returns when ``ws`` is done.
        An unknown, ended or expired session -- or one of another user, which is not told
        apart -- gets ERROR SESSION_EXPIRED and close 4410."""
        conn = self.live.get(request.session_id)
        if conn is None or conn.user_id != user_id or not conn.resumable:
            self.stats["resume_rejected"] += 1
            await _close_direct(ws, CLOSE_EXPIRED, "SESSION_EXPIRED",
                                "the session has ended or expired and cannot be resumed",
                                session_id=request.session_id)
            return
        await conn.attach(ws, request.last_seq)


async def _close_direct(ws: WebSocket, code: int, error: str, message: str, **extra: Any) -> None:
    """ERROR then close on a connection that belongs to no session (yet)."""
    if ws.application_state == WebSocketState.CONNECTED:
        with suppress(Exception):                      # the client is gone already
            await ws.send_json({"type": "ERROR", "code": error, "message": message, **extra})
            await ws.close(code=code)


class _WsTransport:
    """Outbox transport over one WebSocket; raises when it is gone."""

    def __init__(self, ws: WebSocket) -> None:
        self.ws = ws

    async def send_text(self, text: str) -> None:
        if self.ws.application_state != WebSocketState.CONNECTED:
            raise ConnectionError("websocket closed")
        await self.ws.send_text(text)

    async def send_bytes(self, data: bytes) -> None:
        if self.ws.application_state != WebSocketState.CONNECTED:
            raise ConnectionError("websocket closed")
        await self.ws.send_bytes(data)


class Connection:
    """One session.  Usually one WebSocket; after RESUME, the one that resumed it."""

    def __init__(self, handler: StreamHandler, session_id: str, user_id: str = "",
                 roles: list[str] | tuple[str, ...] = ()) -> None:
        self.h = handler
        settings = handler.settings
        self.session_id = session_id
        self.user_id = user_id                        # the token's sub: who connects (the actor)
        self.roles = set(roles)
        self.subject = user_id                        # whom the session is for (RTA-X10)
        self.record_on = self.index_on = self.memory_on = False
        self._subject_ignored = False
        self.state = SessionState(handler.sessions, session_id)
        self.started = time.monotonic()
        self.outbox = Outbox(session_id, pace=settings.output_pace, lead_ms=settings.output_lead_ms,
                             queue_ms=settings.output_queue_ms, retain_bytes=settings.resume_buffer_kb * 1024,
                             on_playing=self._sync_state, on_idle=self._sync_state,
                             wait_timeout_s=settings.output_stall_s)
        self.ws: WebSocket | None = None
        self._ws_done: asyncio.Event | None = None    # released when the current ws is finished with
        self._drop = asyncio.Event()                  # stop reading the current ws (no PONG, replaced)
        self._resumed = asyncio.Event()
        self._pong = asyncio.Event()
        self.suspended = False
        self.resume_allowed = True                    # False: the transport cannot RESUME (gRPC)
        self.ping_enabled = True                      # False: the transport keeps itself alive (gRPC)
        self.audio_samples = 0
        self.dropped = 0
        self._last_drop_warning = 0.0
        self.utterances = 0
        self.config: StreamConfig | None = None
        self.stt: SttSession | None = None
        self.speaker: Speaker | None = None
        self.responses = 0
        self.barge_ins = 0
        self.resumes = 0
        self.last_client_seq = 0
        self.duplicate_frames = 0
        self.audio_q: asyncio.Queue | None = None
        self.utt_q: asyncio.Queue | None = None
        self.turn_q: asyncio.Queue = asyncio.Queue(maxsize=2)
        self._turn_task: asyncio.Task | None = None
        self._sentences: Prefetch | None = None
        self._stt_busy = 0
        self._turn_active = False
        self._barge_utterance: int | None = None
        self._speech_over_answer = False              # the open utterance began while an answer played
        self._vad_model = None
        self._segmenter: Segmenter | None = None
        self._vad_lock = asyncio.Lock()
        self._vad_ready = asyncio.Event()
        self.recorder: Recorder | None = None
        self.recording_skipped: str | None = None
        self._background: set[asyncio.Task] = set()
        # RTA-X9 external brain
        self._speeches: dict[str, _Speech] = {}
        self._speech_order: asyncio.Queue = asyncio.Queue()
        self._speech_now: _Speech | None = None
        self._speak_task: asyncio.Task | None = None
        self._speak_task_loop: asyncio.Task | None = None
        self._speaking = False
        self._spoken_ids: set[str] = set()
        self._speech_lock = asyncio.Lock()
        self._heard: list[str] = []                   # transcripts not yet archived with an answer

    # ----- helpers -----

    def emit(self, event: dict[str, Any]) -> None:
        self.outbox.event(event)

    async def send(self, event: dict[str, Any]) -> None:
        self.outbox.event(event)

    async def close(self, code: int, error: str | None = None, message: str = "") -> None:
        if error:
            await self.outbox.interrupt()
            self.outbox.event({"type": "ERROR", "code": error, "message": message}, priority=True)
        flushed = False
        try:
            budget = 2.0 + (self.outbox.unsent_ms / 1000 if not error else 0)
            flushed = await asyncio.wait_for(self.outbox.flush(), timeout=budget)
        except asyncio.TimeoutError:
            logger.warning("session %s: output flush timed out; abandoning %d frames (%.0f ms audio)",
                           self.session_id, self.outbox.queued, self.outbox.unsent_ms)
        if not flushed:
            logger.warning("session %s: connection closed before output flush completed", self.session_id)
            if error and self.ws is not None:
                with suppress(Exception):
                    await asyncio.wait_for(self.ws.send_json({"type": "ERROR", "code": error,
                                                             "message": message}), timeout=1)
        await self._close_ws(self.ws, code)

    async def _close_ws(self, ws: WebSocket | None, code: int) -> None:
        if ws is not None and ws.application_state == WebSocketState.CONNECTED:
            with suppress(Exception):                  # already closing
                await ws.close(code=code)

    def remaining_s(self) -> float:
        return self.h.settings.session_ttl_s - (time.monotonic() - self.started)

    @property
    def resumable(self) -> bool:
        return (self.resume_allowed and self.config is not None and self.state.state not in TERMINAL
                and self.h.settings.resume_window_s > 0)

    async def set_state(self, new_state: str, **fields: Any) -> None:
        await self.state.to(new_state, **fields)
        if new_state not in TERMINAL:
            self.emit({"type": "STATE", "state": new_state})

    def _pipeline_state(self) -> str:
        if self.outbox.playing:
            return RESPONDING
        if self._stt_busy or self._turn_active or self._speaking:
            return PROCESSING
        return LISTENING

    async def _sync_state(self, **fields: Any) -> None:
        if self.state.state in TERMINAL or self.config is None:
            return
        new = self._pipeline_state()
        try:
            if new != self.state.state:
                await self.set_state(new, **fields)
            elif fields:
                await self.h.sessions.update(self.session_id, **fields)
        except InvalidTransition:
            raise
        except Exception:
            logger.warning("session %s: cannot record state %s", self.session_id, new, exc_info=True)

    # ----- main flow -----

    async def run(self, ws: WebSocket) -> None:
        self.ws = ws
        self.outbox.attach(_WsTransport(ws))
        sender = asyncio.create_task(self.outbox.run())
        heartbeat = asyncio.create_task(self._heartbeat())
        tasks: list[asyncio.Task] = []
        try:
            try:
                first = await asyncio.wait_for(ws.receive(), timeout=self.h.settings.config_timeout_s)
            except asyncio.TimeoutError:
                await self._finish("config_timeout")
                await self.close(CLOSE_TIMEOUT, "config_timeout", "no CONFIG frame received in time")
                return
            if first["type"] == "websocket.disconnect":
                await self._finish(_DISCONNECT)
                return
            try:
                resume = parse_resume(first)
                if resume is not None:
                    # RESUME on a fresh connection: this session record was never used.
                    self.outbox.detach()
                    await self.h.sessions.delete(self.session_id)
                    self.ws = None                    # it belongs to the resumed session now
                    await self.h.resume(ws, self.user_id, resume)
                    return
                default_language = self.h.settings.stt_language
                self.config = parse_config(first, None if default_language == "auto" else default_language,
                                           resample=self.h.settings.codec_resample)
                decoder = make_decoder(self.config.codec, self.config.sample_rate)
                self.outbox.seq_prefix = self.config.framing == "seq"
                if self.h.stt is not None:
                    self.stt = SttSession(self.h.stt, self.config.language, self.send, self.h.rolling)
                if self.h.tts is not None:
                    encoder = make_encoder(self.config.codec, self.config.sample_rate)
                    self.speaker = Speaker(self.h.tts, encoder, self.outbox, self.config.language,
                                           self.h.settings.tts_silence_pad_ms)
            except (ProtocolError, DecodeError, ValueError) as exc:
                self.config = None
                code = exc.code if isinstance(exc, ProtocolError) else "invalid_config"
                await self._finish("invalid_config")
                await self.close(CLOSE_PROTOCOL, code, str(exc))
                return

            if not await self._authorize():
                return
            await self._acquire_vad()
            segmenter = self._segmenter
            recording = await self._start_recording()
            await self.set_state(LISTENING, **self.config.public(), recording_enabled=recording,
                                 recording_skipped_reason=self.recording_skipped, actor_sub=self.user_id,
                                 subject_id=self.subject if self.subject != self.user_id else "",
                                 record=self.record_on, index_transcripts=self.index_on, store_memory=self.memory_on)
            settings = self.h.settings
            self.emit({
                "type": "READY", "session_id": self.session_id, "state": LISTENING,
                "config": self.config.public(),
                "vad": {"chunk_ms": settings.vad_chunk_ms, "window_ms": 32,
                        "threshold": self.h.params.threshold, "min_silence_ms": self.h.params.min_silence_ms,
                        "min_speech_ms": self.h.params.min_speech_ms,
                        "max_utterance_s": self.h.params.max_utterance_s},
                "stt": self._stt_info(),
                "tts": self.h.tts.info() if self.speaker else None,
                "output": self.speaker.encoder.info() if self.speaker else None,
                "barge_in": {"threshold": settings.barge_in_threshold, "min_ms": settings.barge_in_min_ms}
                if self.speaker else None,
                "recording": recording,
                "subject_id": self.subject if self.subject != self.user_id else None,
                "side_effects": {"record": self.record_on, "index_transcripts": self.index_on,
                                 "store_memory": self.memory_on},
                "expires_in": settings.session_ttl_s,
                "resume_window_s": settings.resume_window_s,
                "ping_interval_s": settings.ping_interval_s,
            })

            if self._subject_ignored:
                self.emit({"type": "WARNING", "code": "subject_ignored",
                           "message": f"subject_id needs the {ROLE_DELEGATE} realm role; the session is yours"})
            self.audio_q = asyncio.Queue(maxsize=max(1, settings.audio_queue_max))
            self.utt_q = asyncio.Queue(maxsize=max(1, settings.stt_queue_max))
            vad_task = asyncio.create_task(self._vad_loop(segmenter, self.audio_q, self.utt_q))
            utt_task = asyncio.create_task(self._utterance_loop(self.utt_q))
            turn_task = asyncio.create_task(self._turn_loop())
            tasks = [vad_task, utt_task, turn_task, asyncio.create_task(self._pinger())]
            if self.config.brain == "external":
                self._speak_task_loop = asyncio.create_task(self._speak_loop())
                tasks.append(self._speak_task_loop)
            if self.stt is not None and self.h.rolling.enabled:
                tasks.append(asyncio.create_task(self.stt.rolling_loop()))
            connected = asyncio.create_task(self._connected(decoder))
            tasks.append(connected)
            watched = {connected, sender, vad_task, utt_task, turn_task}
            if self._speak_task_loop is not None:
                watched.add(self._speak_task_loop)
            done, _ = await asyncio.wait(watched,
                                         return_when=asyncio.FIRST_COMPLETED)
            for task in done - {connected}:
                task.result()
                raise RuntimeError("session pipeline stopped unexpectedly")
            reason = connected.result()
            if reason == _END:
                # Finish what was said, answer it and let it play -- unless the client leaves first.
                drain = asyncio.create_task(self._drain(vad_task, utt_task, turn_task))
                reader = asyncio.create_task(self._read_after_end(self.ws))
                tasks += [drain, reader]
                await asyncio.wait({drain, reader}, return_when=asyncio.FIRST_COMPLETED)
                if drain.done() and drain.result():
                    await self._finish(reason)
                    self.emit({"type": "ENDED", **self._summary()})
                    await self.close(CLOSE_NORMAL)
                else:
                    logger.info("session %s: client left before the answers were delivered", self.session_id)
                    await self._finish("disconnect_after_end")
            elif reason == "expired":
                await self._finish(reason, EXPIRED)
                self.h.stats["sessions_expired"] += 1
            else:
                await self._finish(reason)
                if reason == "session_expired":
                    await self.close(CLOSE_TIMEOUT, "session_expired", "RT_SESSION_TTL_S reached")
        except asyncio.CancelledError:
            await asyncio.shield(self._finish("cancelled"))
            raise
        except OutboxOverflow as exc:
            logger.warning("session %s: %s", self.session_id, exc)
            await self._finish("queue_overflow")
            await self.close(CLOSE_INTERNAL, "QUEUE_OVERFLOW", str(exc))
        except Exception:
            logger.exception("session %s failed", self.session_id)
            await self._finish("error")
            await self.close(CLOSE_INTERNAL, "internal_error", "internal error")
        finally:
            for task in [*tasks, self._turn_task, self._speak_task]:
                if task is not None:
                    task.cancel()
            heartbeat.cancel()
            sender.cancel()
            await asyncio.gather(*[t for t in [*tasks, self._turn_task, self._speak_task, heartbeat, sender] if t is not None],
                                 return_exceptions=True)
            await self._close_ws(self.ws, CLOSE_NORMAL)    # whatever the way out, the socket closes
            if self._ws_done is not None:
                self._ws_done.set()
            await self._release_vad()

    async def _acquire_vad(self) -> None:
        async with self._vad_lock:
            if self._vad_model is None:
                self._vad_model = await self.h.vad_pool.acquire()
                if self._segmenter is None:
                    self._segmenter = Segmenter(self._vad_model, self.h.params)
                else:
                    self._segmenter.replace_model(self._vad_model)
            self._vad_ready.set()

    async def _release_vad(self) -> None:
        self._vad_ready.clear()
        async with self._vad_lock:
            if self._vad_model is not None:
                self.h.vad_pool.release(self._vad_model)
                self._vad_model = None

    async def _feed_vad(self, segmenter: Segmenter, pcm: np.ndarray):
        while True:
            await self._vad_ready.wait()
            async with self._vad_lock:
                if self._vad_model is None:
                    continue
                worker = asyncio.create_task(asyncio.to_thread(segmenter.feed, pcm))
                try:
                    return await asyncio.shield(worker)
                except asyncio.CancelledError:
                    await worker
                    raise

    async def _connected(self, decoder) -> str:
        """Receive from whichever WebSocket the session has, through drops and resumes; returns
        END, session_expired (lifetime), expired (not resumed in time) or disconnect."""
        while True:
            ws = self.ws
            reason = await self._receive_loop(ws, decoder)
            if reason in (_END, "session_expired"):
                return reason
            if self.ws is not ws:
                continue                              # already replaced by a RESUME
            if self._ws_done is not None:
                self._ws_done.set()
            if not self.resumable:
                return reason
            if not await self._suspend():
                return "expired" if self.remaining_s() > 0 else "session_expired"

    async def _drain(self, vad_task: asyncio.Task, utt_task: asyncio.Task, turn_task: asyncio.Task) -> bool:
        assert self.audio_q is not None and self.utt_q is not None
        await self.audio_q.put(_END)                  # sentinel; VAD flushes on END only
        await vad_task
        await self.utt_q.put(None)
        await utt_task
        await self.turn_q.put(None)
        await turn_task
        if self._speak_task_loop is not None:
            self._close_speeches()
            await self._speak_task_loop
        return await self.outbox.flush(played=True)          # False: the client is gone

    async def _read_after_end(self, ws: WebSocket) -> None:
        """After END: keep answering PING / PONG; return when the client goes away."""
        while True:
            try:
                msg = await self._receive(ws, max(0.0, self.remaining_s()))
            except Exception:  # noqa: BLE001 -- timeout (lifetime) or a socket that is gone
                return
            if msg["type"] == "websocket.disconnect":
                return
            body = parse_json_text(msg.get("text") or "")
            kind = body.get("type") if body else None
            if kind == "PONG":
                self._pong.set()
            elif kind == "PING":
                self.emit({"type": "PONG", "ts": round(time.time() * 1000)})

    async def _suspend(self) -> bool:
        """Dropped without END: keep the session for RT_SESSION_RESUME_WINDOW_S."""
        self.outbox.detach()
        self.suspended = True
        await self._release_vad()
        self._resumed.clear()
        window = min(self.h.settings.resume_window_s, max(0.0, self.remaining_s()))
        logger.info("session %s suspended for %.0f s", self.session_id, window)
        await self._record(connection="suspended", resume_until=time.time() + window)
        try:
            await asyncio.wait_for(self._resumed.wait(), timeout=window)
        except asyncio.TimeoutError:
            logger.info("session %s not resumed: expired", self.session_id)
            return False
        return True

    async def attach(self, ws: WebSocket, last_seq: int) -> None:
        """RESUME (called from the new connection's handler): replay, then carry on with ``ws``.
        Returns when the session is done with ``ws``."""
        old = self.ws
        if old is not None and old is not ws and not self.suspended:
            self.outbox.detach()                      # still connected: the new connection wins
            self.ws = ws
            self._drop.set()
            await self._close_ws(old, CLOSE_REPLACED)
        done = asyncio.Event()
        self.ws, self._ws_done = ws, done
        self._drop.clear()
        self._pong.set()

        def resumed(replay: Replay) -> dict[str, Any]:
            return {"type": "RESUMED", "session_id": self.session_id, "replayed": replay.frames,
                    "gap": replay.gap, "last_client_seq": self.last_client_seq, "state": self.state.state}

        try:
            replay = await self.outbox.resume(_WsTransport(ws), last_seq, resumed)
        except Exception:  # noqa: BLE001 -- the new connection failed at once; stay suspended
            logger.info("session %s: resume connection failed while replaying", self.session_id)
            self.outbox.detach()
            return
        await self._acquire_vad()
        self.suspended = False
        self.resumes += 1
        self.h.stats["resumes"] += 1
        logger.info("session %s resumed after seq %d: %d frames replayed%s", self.session_id, last_seq,
                    replay.frames, " (gap)" if replay.gap else "")
        await self._record(connection="connected", resume_until="")
        self._resumed.set()
        await done.wait()

    async def _record(self, **fields: Any) -> None:
        try:
            await self.h.sessions.update(self.session_id, **fields)
        except Exception:
            logger.warning("session %s: cannot record %s", self.session_id, fields, exc_info=True)

    def _stt_info(self) -> dict[str, Any] | None:
        if self.stt is None:
            return None
        info = self.h.stt.info() if self.h.stt else {}
        return {"model": info.get("model"), "language": self.config.language if self.config else None,
                "streaming": self.h.rolling.enabled, "stride_s": self.h.rolling.stride_s}

    def _summary(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "utterances": self.utterances,
            "audio_ms": int(self.audio_samples / 16),
            "dropped_chunks": self.dropped,
            "responses": self.responses,
            "barge_ins": self.barge_ins,
            "resumes": self.resumes,
        }

    async def _finish(self, reason: str, final_state: str = ENDED) -> None:
        """Write the final (partial) session state; the Redis record lives on until its TTL."""
        if self.state.state in TERMINAL:
            return
        if self.recorder is not None:
            try:
                await asyncio.shield(self.recorder.finish(reason))
            except Exception:
                logger.exception("session %s: recording not finalized", self.session_id)
        try:
            s = self._summary()
            await self.state.to(final_state, end_reason=reason, ended_at=time.time(),
                                utterances=s["utterances"], audio_ms=s["audio_ms"],
                                dropped_chunks=s["dropped_chunks"], connection="closed")
        except Exception:
            logger.exception("session %s: cannot write final state", self.session_id)

    # ----- voice turns -----

    async def _turn_loop(self) -> None:
        while True:
            item = await self.turn_q.get()
            if item is None:
                return
            utt, transcript = item
            index = utt.index
            self._turn_active = True
            await self._sync_state()
            task = asyncio.create_task(self._turn(utt, transcript))
            self._turn_task = task
            await asyncio.wait({task})                 # a barge-in cancels the turn, not this loop
            if not task.cancelled() and task.exception() is not None:
                if isinstance(task.exception(), InvalidTransition):
                    raise task.exception()
                logger.error("session %s: voice turn failed", self.session_id, exc_info=task.exception())
                self.h.stats["response_failed"] += 1
                self.emit({"type": "WARNING", "code": "response_failed",
                           "message": "the spoken answer failed", "utterance": index})
            self._turn_task = None
            self._turn_active = False
            await self._sync_state()

    @property
    def _external(self) -> bool:
        return self.config is not None and self.config.brain == "external"

    def _guard(self, direction: str):
        """Module 23 for this direction in this session, or None."""
        if self.h.guardrail is None:
            return None
        if self._external:                             # RTA-X9: RT_EXTERNAL_GUARDRAIL
            mode = self.h.settings.external_guardrail
            return self.h.guardrail if mode == "both" or mode == direction else None
        if direction == "output" and not self.h.settings.guardrail_output:
            return None
        return self.h.guardrail

    @property
    def _routes_transcripts(self) -> bool:
        """Whether a final transcript goes anywhere: a voice reply, Module 25, an external brain."""
        return bool((self.speaker and self.h.llm) or (self.h.indexer and self.index_on) or self._external)

    def _later(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _authorize(self) -> bool:
        """RTA-X9 / RTA-X10: brain mode, delegated subject and side effects for this session."""
        assert self.config is not None
        cfg, settings = self.config, self.h.settings
        if cfg.brain == "external" and ROLE_EXTERNAL_BRAIN not in self.roles:
            logger.warning("session %s: user %s may not use brain=external (no %s role)", self.session_id,
                           self.user_id, ROLE_EXTERNAL_BRAIN)
            await self._finish("forbidden")
            await self.close(CLOSE_FORBIDDEN, "BRAIN_MODE_FORBIDDEN",
                             f"brain=external needs the {ROLE_EXTERNAL_BRAIN} realm role")
            return False
        if cfg.subject_id:
            if ROLE_DELEGATE in self.roles:
                self.subject = cfg.subject_id
            else:
                logger.warning("session %s: subject_id from %s ignored (no %s role)", self.session_id,
                               self.user_id, ROLE_DELEGATE)
                self._subject_ignored = True              # told right after READY
        external = cfg.brain == "external"

        def flag(value: bool | None, default: bool) -> bool:
            return default if value is None else value

        self.record_on = flag(cfg.record, not external)
        self.index_on = settings.index_transcripts and flag(cfg.index_transcripts, not external)
        if self.roles & set(SESSION_ROLES) and self.subject == self.user_id and cfg.index_transcripts is not True:
            self.index_on = False                      # never into a service account's own branch by default
        self.memory_on = settings.memory_enabled and flag(cfg.store_memory, not external)
        logger.info("session %s: actor %s subject %s brain %s record=%s index=%s memory=%s", self.session_id,
                    self.user_id, self.subject, cfg.brain, self.record_on, self.index_on, self.memory_on)
        return True

    async def _turn(self, utt: Utterance, transcript: Transcript) -> None:
        """One final transcript: screened (RTA-X6), indexed (Module 25), then answered (RA-5) -- or,
        with an external brain, left to the client (RTA-X9)."""
        request_id = uuid.uuid4().hex
        text, warned = transcript.text, False
        guard = self._guard("input")
        if guard is not None:
            decision = await guard.decide(text, "input", request_id=request_id, user_id=self.subject,
                                          session_id=self.session_id)
            if decision.blocked:
                await self._blocked(utt.index, request_id, "input", decision.reason)
                return
            if decision.warned or decision.decision == "redact":
                logger.warning("session %s request %s: guardrail %s on the transcript (%s)",
                               self.session_id, request_id, decision.decision, decision.reason)
            text, warned = decision.text, decision.warned
        if self.h.indexer is not None and self.index_on:
            self._later(self.h.indexer.publish(self.subject, self.session_id, utt.index, text,
                                               language=transcript.language, start_ms=utt.start_ms, end_ms=utt.end_ms))
        if self._external:
            self._heard.append(text)                   # archived with the client's answer (store_memory)
            return
        if self.speaker is not None and self.h.llm is not None:
            await self._voice_turn(utt.index, text, request_id, warned)

    async def _blocked(self, index: int | None, request_id: str, direction: str, reason: str) -> None:
        self.h.stats["guardrail_blocks"] += 1
        logger.warning("session %s request %s: guardrail blocked the %s (%s)", self.session_id, request_id,
                       direction, reason)
        self.emit({"type": "GUARDRAIL_BLOCK", "request_id": request_id, "utterance": index,
                   "direction": direction, "reason": reason})
        if direction == "input" and self.speaker is not None and not self._external:
            await self.speaker.speak(one(self.h.settings.guardrail_block_text), request_id, fallback=True)
            self.responses += 1

    async def _screened(self, sentences, index: int | None, request_id: str, spoken: list[str]):
        """RTA-X6: every answer sentence through Module 23 before it is synthesized."""
        guard = self._guard("output")
        async with aclosing(sentences) as source:
            async for sentence in source:
                if guard is not None:
                    decision = await guard.decide(sentence, "output", request_id=request_id, user_id=self.subject,
                                                  session_id=self.session_id)
                    if decision.blocked:
                        await self._blocked(index, request_id, "output", decision.reason)
                        spoken.append(self.h.settings.guardrail_block_text)
                        yield self.h.settings.guardrail_block_text
                        return                         # the rest of the answer is dropped
                    if decision.warned:
                        logger.warning("session %s request %s: guardrail warn on the answer (%s)",
                                       self.session_id, request_id, decision.reason)
                    sentence = decision.text
                spoken.append(sentence)
                yield sentence

    async def _voice_turn(self, index: int, text: str, request_id: str, guardrail_warning: bool = False) -> None:
        """RA-5: one Module 15 turn, spoken back as it arrives (RA-4)."""
        assert self.speaker is not None and self.h.llm is not None
        # Audit correlation of this TRANSCRIPT_FINAL with the Module 15 request (no transcript text).
        logger.info("session %s actor %s subject %s utterance %d TRANSCRIPT_FINAL (%d chars) -> Module 15 "
                    "request_id=%s", self.session_id, self.user_id, self.subject, index, len(text), request_id)
        spoken: list[str] = []
        self.responses += 1
        answer = self.h.llm.sentences(text, self.session_id, self.subject, request_id,
                                      guardrail_warning=guardrail_warning)
        self._sentences = Prefetch(self._screened(answer, index, request_id, spoken), self.h.settings.tts_queue_max)
        try:
            await self.speaker.speak(self._sentences, request_id)
        except LlmError as exc:
            self._sentences = None
            logger.warning("session %s request %s: %s (%s)", self.session_id, request_id, exc.code, exc.message)
            if exc.code == "LLM_TIMEOUT":
                self.h.stats["llm_timeouts"] += 1
            self.emit({"type": "ERROR", "code": exc.code, "message": exc.message, "fatal": False,
                       "request_id": request_id, "utterance": index})
            fallback_id = uuid.uuid4().hex
            await self.speaker.speak(one(self.h.settings.llm_fallback_text), fallback_id, fallback=True)
            return
        finally:
            self._sentences = None
        if self.h.memory is not None and self.memory_on and not self.h.llm.archives_turns and spoken:
            # RTA-X6: Module 26 -- only for a brain that does not archive its own turns (Module 15 does).
            self._later(self.h.memory.append_turn(self.subject, self.session_id, text, "".join(spoken)))

    # ----- external brain (RTA-X9) -----

    def _on_speak(self, body: dict[str, Any]) -> None:
        uid, text, final = body.get("utterance_id"), body.get("text", ""), body.get("final", False)
        if not isinstance(uid, str) or not uid or len(uid) > 128 or not isinstance(text, str) \
                or not isinstance(final, bool):
            self.emit({"type": "WARNING", "code": "invalid_speak",
                       "message": "SPEAK needs utterance_id (string), text (string) and final (bool)"})
            return
        if self.speaker is None:
            self.emit({"type": "WARNING", "code": "tts_unavailable", "message": "this server has no TTS"})
            return
        speech = self._speeches.get(uid)
        if speech is None:
            if uid in self._spoken_ids:
                self.emit({"type": "WARNING", "code": "speak_after_final", "utterance_id": uid,
                           "message": "this utterance_id is finished"})
                return
            speech = self._speeches[uid] = _Speech(uid)
            self._speech_order.put_nowait(speech)
        if speech.final or speech.cancelled:
            self.emit({"type": "WARNING", "code": "speak_after_final", "utterance_id": uid,
                       "message": "this utterance_id is finished"})
            return
        splitter = SentenceSplitter()                  # each SPEAK frame stands on its own
        for sentence in splitter.feed(text) + splitter.flush():
            speech.texts.put_nowait(sentence)
        if final:
            speech.final = True
            speech.texts.put_nowait(None)

    async def _on_speak_cancel(self, body: dict[str, Any]) -> None:
        speech = self._speeches.get(body.get("utterance_id"))
        if speech is None or speech.cancelled:
            return
        speech.cancelled = True
        if speech is self._speech_now and self._speak_task is not None and not self._speak_task.done():
            self._speak_task.cancel()
            await asyncio.wait({self._speak_task})
            ended = self.outbox.answer_ended(speech.id)
            await self.outbox.drop_request(speech.id)
            if not ended:
                self.emit({"type": "RESPONSE_END", "request_id": speech.id, "cancelled": True})
        self._speeches.pop(speech.id, None)
        self._spoken_ids.add(speech.id)

    def _close_speeches(self) -> None:
        """END: what was sent is all there is."""
        for speech in self._speeches.values():
            if not speech.final:
                speech.final = True
                speech.texts.put_nowait(None)
        self._speech_order.put_nowait(None)

    async def _speak_loop(self) -> None:
        while True:
            speech = await self._speech_order.get()
            if speech is None:
                return
            async with self._speech_lock:
                if speech.cancelled:
                    continue
                self._speech_now, self._speaking = speech, True
                await self._sync_state()
                task = asyncio.create_task(self._speak(speech))
                self._speak_task = task
            await asyncio.wait({task})
            if not task.cancelled() and task.exception() is not None:
                if isinstance(task.exception(), InvalidTransition):
                    raise task.exception()
                logger.error("session %s: SPEAK %s failed", self.session_id, speech.id, exc_info=task.exception())
                self.h.stats["response_failed"] += 1
                self.emit({"type": "WARNING", "code": "response_failed", "message": "the spoken answer failed",
                           "utterance_id": speech.id})
            self._speeches.pop(speech.id, None)
            self._spoken_ids.add(speech.id)
            self._speech_now, self._speak_task, self._speaking = None, None, False
            self.responses += 1
            await self._sync_state()

    async def _speak(self, speech: _Speech) -> None:
        assert self.speaker is not None

        async def texts():
            while True:
                text = await speech.texts.get()
                if text is None:
                    return
                yield text

        # Read ahead so Module 23 checks later sentences while earlier ones are synthesized.
        self._sentences = Prefetch(self._screened(texts(), None, speech.id, speech.spoken),
                                   self.h.settings.tts_queue_max)
        try:
            await self.speaker.speak(self._sentences, speech.id)
        finally:
            self._sentences = None
        if self.h.memory is not None and self.memory_on and self._heard and speech.spoken:
            heard, self._heard = "".join(self._heard), []
            self._later(self.h.memory.append_turn(self.subject, self.session_id, heard, "".join(speech.spoken)))

    async def _start_recording(self) -> bool:
        """RTA-X7: record only with the subject's voice_recording consent (and RT_RECORDING_ENABLED)."""
        if self.h.consent is None or self.h.recorder_factory is None:
            self.recording_skipped = "disabled_by_config"
            return False
        if not self.record_on:
            self.recording_skipped = "disabled_for_session"
            return False
        enabled, reason = await self.h.consent.check(self.subject)
        if not enabled:
            self.recording_skipped = reason
            return False
        try:
            recorder = self.h.recorder_factory(self.session_id, self.subject)
            await recorder.start()
        except Exception as exc:  # noqa: BLE001 -- storage problems never stop the conversation
            logger.error("session %s: recording not started: %s", self.session_id, exc)
            self.recording_skipped = "storage_unavailable"
            return False
        self.recorder = recorder
        return True

    async def _barge_in(self, segmenter: Segmenter) -> None:
        """RTA-X5: the user speaks over the answer."""
        request_id, position = self.outbox.answer_position()
        ended = self.outbox.answer_ended(request_id)
        self._barge_utterance = segmenter.next_index if segmenter.triggered else None
        async with self._speech_lock:
            for speech in list(self._speeches.values()):  # RTA-X9: every queued SPEAK goes too
                speech.cancelled = True
                self._spoken_ids.add(speech.id)
            self._speeches.clear()
            for task in (self._turn_task, self._speak_task):
                if task is not None and not task.done():
                    task.cancel()
                    await asyncio.wait({task})
            await self.outbox.interrupt()
            self.barge_ins += 1
            self.h.stats["barge_ins"] += 1
            logger.info("session %s: barge-in at %d ms of answer %s", self.session_id, position, request_id)
            self.emit({"type": "BARGE_IN", "request_id": request_id, "tts_position_ms": position})
            if not ended:
                self.emit({"type": "RESPONSE_END", "request_id": request_id, "interrupted": True,
                           "duration_ms": position})
            await self._sync_state()

    # ----- tasks -----

    async def _heartbeat(self) -> None:
        while True:
            await asyncio.sleep(self.h.settings.heartbeat_s)
            try:
                await self.h.sessions.heartbeat(self.session_id, self.user_id)
            except Exception:
                logger.warning("session %s heartbeat failed", self.session_id, exc_info=True)

    async def _pinger(self) -> None:
        """RTA-X3: PING every RT_PING_INTERVAL_S; no PONG in RT_PONG_TIMEOUT_S drops the connection."""
        interval = self.h.settings.ping_interval_s
        if interval <= 0 or not self.ping_enabled:
            return
        while True:
            await asyncio.sleep(interval)
            ws = self.ws
            if self.suspended or ws is None:
                continue
            self._pong.clear()
            self.emit({"type": "PING", "ts": round(time.time() * 1000)})
            try:
                await asyncio.wait_for(self._pong.wait(), timeout=self.h.settings.pong_timeout_s)
            except asyncio.TimeoutError:
                if self.ws is ws and not self.suspended:
                    logger.info("session %s: no PONG within %.0f s", self.session_id,
                                self.h.settings.pong_timeout_s)
                    self.h.stats["pong_timeouts"] += 1
                    self.emit({"type": "ERROR", "code": "pong_timeout", "message": "no PONG received"})
                    try:
                        await asyncio.wait_for(self.outbox.flush(), timeout=1.0)
                    except asyncio.TimeoutError:
                        pass
                    await self._close_ws(ws, CLOSE_TIMEOUT)
                    self._drop.set()                  # the receive loop treats it as a drop

    async def _receive(self, ws: WebSocket, timeout: float) -> dict[str, Any]:
        """ws.receive(), or a disconnect when the session drops this connection."""
        receive = asyncio.ensure_future(ws.receive())
        drop = asyncio.ensure_future(self._drop.wait())
        try:
            done, _ = await asyncio.wait({receive, drop}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        finally:
            drop.cancel()
        if receive in done:
            return receive.result()
        receive.cancel()
        if drop in done:
            return {"type": "websocket.disconnect", "code": CLOSE_TIMEOUT}
        raise asyncio.TimeoutError

    async def _receive_loop(self, ws: WebSocket, decoder) -> str:
        assert self.audio_q is not None
        seq_framing = self.config is not None and self.config.framing == "seq"
        while True:
            remaining = self.remaining_s()
            if remaining <= 0:
                return "session_expired"
            try:
                msg = await self._receive(ws, remaining)
            except asyncio.TimeoutError:
                return "session_expired"
            except Exception:  # noqa: BLE001 -- receive on a socket that is already gone
                return _DISCONNECT
            if msg["type"] == "websocket.disconnect":
                return _DISCONNECT
            data = msg.get("bytes")
            if data is not None:
                if seq_framing:
                    if len(data) < 4:
                        self.emit({"type": "WARNING", "code": "invalid_audio", "message": "frame shorter than its seq"})
                        continue
                    seq = struct.unpack_from("<I", data)[0]
                    if seq <= self.last_client_seq:
                        self.duplicate_frames += 1            # resent after RESUME
                        continue
                    self.last_client_seq, data = seq, data[4:]
                try:
                    pcm = decoder.decode(data)
                except DecodeError as exc:
                    self.emit({"type": "WARNING", "code": "invalid_audio", "message": str(exc)})
                    continue
                self._enqueue_pcm(pcm)
                continue
            text = msg.get("text") or ""
            if is_end(text):
                tail = decoder.flush()                  # resampler delay
                self._enqueue_pcm(tail)
                return _END
            body = parse_json_text(text)
            kind = body.get("type") if body else None
            if body is not None and isinstance(body.get("seq"), int) and not isinstance(body["seq"], bool):
                self.last_client_seq = max(self.last_client_seq, body["seq"])
            if kind == "PONG":
                self._pong.set()
            elif kind == "PING":
                self.emit({"type": "PONG", "ts": round(time.time() * 1000)})
            elif kind in ("SPEAK", "SPEAK_CANCEL") and self._external:
                if kind == "SPEAK":
                    self._on_speak(body)
                else:
                    await self._on_speak_cancel(body)
            else:
                self.emit({"type": "WARNING", "code": "unexpected_frame",
                           "message": f"ignored text frame of type {kind!r}"})

    def _enqueue_pcm(self, pcm: np.ndarray) -> None:
        self.audio_samples += len(pcm)
        if self.recorder is not None:
            self.recorder.add(pcm)
        chunk = max(1, int(16 * self.h.settings.vad_chunk_ms))
        for off in range(0, len(pcm), chunk):
            self._enqueue_audio(pcm[off:off + chunk])

    def _enqueue_audio(self, chunk: np.ndarray) -> None:
        """RTA-X4: a full audio queue drops its oldest chunk."""
        assert self.audio_q is not None
        try:
            self.audio_q.put_nowait((chunk, self.speaker is not None and self.outbox.playing))
            return
        except asyncio.QueueFull:
            pass
        try:
            oldest = self.audio_q.get_nowait()
            if isinstance(oldest, str):              # never drop the END sentinel
                self.audio_q.put_nowait(oldest)
                return
        except asyncio.QueueEmpty:
            pass
        self.audio_q.put_nowait((chunk, self.speaker is not None and self.outbox.playing))
        self.dropped += 1
        self.h.stats["audio_dropped"] += 1
        now = time.monotonic()
        if now - self._last_drop_warning >= 1.0:
            self._last_drop_warning = now
            self.emit({"type": "WARNING", "code": "QUEUE_OVERFLOW",
                       "message": "audio arrives faster than VAD processes it; the oldest audio was dropped",
                       "dropped_chunks": self.dropped})

    async def _vad_loop(self, segmenter: Segmenter, audio_q: asyncio.Queue, utt_q: asyncio.Queue) -> None:
        barge_ms = self.h.settings.barge_in_min_ms
        while True:
            item = await audio_q.get()
            chunks: list[tuple[np.ndarray, bool]] = []
            sentinel = None
            while True:
                if isinstance(item, str):
                    sentinel = item
                    break
                chunks.append(item)
                try:
                    item = audio_q.get_nowait()
                except asyncio.QueueEmpty:
                    break
            for playing, group in groupby(chunks, key=lambda item: item[1]):
                samples = [pcm for pcm, _ in group]
                batches = samples if playing else [np.concatenate(samples)]
                for chunk in batches:
                    events = await self._feed_vad(segmenter, chunk)
                    await self._events(events, utt_q, playing)
                    if self.speaker is not None and self.outbox.playing and segmenter.loud_ms >= barge_ms:
                        await self._barge_in(segmenter)
                    if self.stt is not None and segmenter.triggered and not self._suppress_speech(segmenter.next_index):
                        self.stt.speech_progress(segmenter.next_index, segmenter.speech_elapsed_ms, segmenter.snapshot)
            if sentinel is not None:
                if sentinel == _END:
                    await self._events(segmenter.flush(), utt_q)
                return

    def _suppress_speech(self, index: int) -> bool:
        return self._speech_over_answer and index != self._barge_utterance

    async def _events(self, events, utt_q: asyncio.Queue, playing: bool = False) -> None:
        for ev in events:
            if isinstance(ev, SpeechStart):
                self._speech_over_answer = playing
                self.emit({"type": "SPEECH_START", "start_ms": ev.start_ms})
            elif isinstance(ev, Utterance):
                if self._suppress_speech(ev.index):
                    # Began while an answer was playing and never barged in: without echo
                    # cancellation this is the answer itself coming back through the microphone.
                    # (Speech that began before the answer did is the user's, whenever it ends.)
                    if self.stt is not None:
                        self.stt.utterance_discarded()
                    self.emit({"type": "DISCARDED", "start_ms": ev.start_ms, "end_ms": ev.end_ms,
                               "speech_ms": ev.speech_ms, "reason": "during_response"})
                    continue
                if self.stt is not None:
                    self.stt.utterance_closed(ev.index)
                self.emit({"type": "UTTERANCE", "index": ev.index, "start_ms": ev.start_ms,
                           "end_ms": ev.end_ms, "duration_ms": ev.duration_ms,
                           "speech_ms": ev.speech_ms, "reason": ev.reason})
                try:
                    utt_q.put_nowait(ev)
                except asyncio.QueueFull:
                    self.h.stats["stt_overloaded"] += 1
                    self.emit({"type": "ERROR", "code": "STT_OVERLOADED", "fatal": False,
                               "message": "speech recognition is behind; this utterance was dropped",
                               "utterance": ev.index})
            elif isinstance(ev, Discarded):
                if self.stt is not None:
                    self.stt.utterance_discarded()
                self.emit({"type": "DISCARDED", "start_ms": ev.start_ms, "end_ms": ev.end_ms,
                           "speech_ms": ev.speech_ms, "reason": ev.reason})

    async def _utterance_loop(self, utt_q: asyncio.Queue) -> None:
        min_chars = self.h.settings.min_utterance_chars
        while True:
            utt = await utt_q.get()
            if utt is None:
                return
            self._stt_busy += 1
            await self._sync_state()
            transcript = None
            try:
                if self.stt is not None:
                    started = time.monotonic()
                    try:
                        transcript = await self.stt.finalize(utt.index, utt.pcm, utt.duration_ms)
                    except Exception:
                        self.h.stats["stt_failed"] += 1
                        logger.exception("session %s: STT failed on utterance %d", self.session_id, utt.index)
                        self.emit({"type": "WARNING", "code": "stt_failed",
                                   "message": "speech recognition failed", "utterance": utt.index})
                    else:
                        self.h.stt_latency.observe(time.monotonic() - started)
                try:
                    await self.h.sink.handle(self.session_id, self.config, utt)
                except Exception:
                    logger.exception("session %s: utterance sink failed", self.session_id)
            finally:
                self._stt_busy -= 1
            if (transcript is not None and self._routes_transcripts
                    and sum(c.isalnum() for c in transcript.text) >= min_chars):
                # RTA-X5 hysteresis: at least RT_MIN_UTTERANCE_CHARS letters make a turn.
                await self.turn_q.put((utt, transcript))
                self._turn_active = True
            self.utterances += 1
            await self._sync_state(utterances=self.utterances)
