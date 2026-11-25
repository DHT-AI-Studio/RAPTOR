"""RA-6: AudioService.AudioChat -- a voice session over gRPC bidirectional streaming.

One call is one session of the same pipeline as WS /audio/stream (app/stream.py: VAD -> STT ->
Module 23 / Module 15 -> TTS, barge-in, recording, Module 25 / 26 side effects, session slots
and Redis records).  ``GrpcSocket`` plays the WebSocket for app.stream.Connection:

  caller -> server   the first AudioChunk becomes the CONFIG frame (codec, sample_rate,
                     language), every payload a binary AUDIO frame, half-close the END frame,
                     cancellation a disconnect
  server -> caller   TRANSCRIPT_PROVISIONAL / TRANSCRIPT_FINAL -> TranscriptEvent(is_final),
                     answer audio -> AudioResponseChunk(is_last=false), RESPONSE_END ->
                     AudioResponseChunk(is_last=true), ERROR / WARNING / GUARDRAIL_BLOCK ->
                     ErrorEvent(fatal); the WebSocket's other events have no gRPC counterpart

The caller authenticates with ``authorization: Bearer <JWT>`` metadata, checked by
``JwtInterceptor`` with the same KeyProvider as POST /audio/session/token (Keycloak JWKS or
RT_JWT_PUBLIC_KEY, RT_JWT_AUDIENCE).  gRPC calls cannot RESUME and have HTTP/2 keep-alive, so
their sessions skip RESUME and PING.  Stubs: scripts/gen_proto.py (proto/raptor/audio/v1).
"""
from __future__ import annotations

import asyncio
import contextvars
import json
import logging
from contextlib import aclosing, suppress
from typing import Any

import grpc
from raptor.audio.v1 import audio_pb2, audio_pb2_grpc
from starlette.websockets import WebSocketState

from app.auth import KeyProvider
from app.config import Settings
from app.errors import ServiceError
from app.protocol import (
    CLOSE_FORBIDDEN,
    CLOSE_INTERNAL,
    CLOSE_NORMAL,
    CLOSE_PROTOCOL,
    CLOSE_TIMEOUT,
)
from app.session import TOO_MANY_SESSIONS
from app.stream import SESSION_ROLES, StreamHandler

logger = logging.getLogger(__name__)

SERVICE_NAME = "raptor.audio.v1.AudioService"
DEFAULT_RATES = {"pcm": 16000, "opus": 48000, "pcmu": 8000, "pcma": 8000}
INCOMING_MAX = 256                     # caller frames waiting for the session (gRPC flow control beyond)
OUTGOING_MAX = 64                      # events waiting for the caller to read them
DRAIN_IDLE_S, DRAIN_MAX_S = 0.05, 1.0  # reading a refused caller's audio before the status (_drain)

_STATUS_BY_CLOSE = {
    CLOSE_PROTOCOL: grpc.StatusCode.INVALID_ARGUMENT,
    CLOSE_FORBIDDEN: grpc.StatusCode.PERMISSION_DENIED,
    CLOSE_TIMEOUT: grpc.StatusCode.DEADLINE_EXCEEDED,
    CLOSE_INTERNAL: grpc.StatusCode.INTERNAL,
}
_DISCONNECT = {"type": "websocket.disconnect", "code": 1000}

_claims: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("rt_grpc_claims", default=None)


# ---------------------------------------------------------------- auth


def bearer_token(metadata) -> str | None:
    for key, value in metadata or ():
        if key.lower() == "authorization" and isinstance(value, str):
            scheme, _, token = value.partition(" ")
            if scheme.lower() == "bearer" and token.strip():
                return token.strip()
    return None


def _same_shape(handler: grpc.RpcMethodHandler, unary_unary=None, unary_stream=None, stream_unary=None,
                stream_stream=None) -> grpc.RpcMethodHandler:
    kw = {"request_deserializer": handler.request_deserializer, "response_serializer": handler.response_serializer}
    if handler.stream_stream:
        return grpc.stream_stream_rpc_method_handler(stream_stream, **kw)
    if handler.unary_stream:
        return grpc.unary_stream_rpc_method_handler(unary_stream, **kw)
    if handler.stream_unary:
        return grpc.stream_unary_rpc_method_handler(stream_unary, **kw)
    return grpc.unary_unary_rpc_method_handler(unary_unary, **kw)


async def _drain(requests, idle_s: float = DRAIN_IDLE_S, max_s: float = DRAIN_MAX_S) -> None:
    """Read what the caller has already sent before refusing the call.  A grpc.aio client whose
    write is still in flight when the status arrives reports INTERNAL "Internal error from Core"
    instead of that status; refusing when the caller pauses (``idle_s`` without a message) or
    half-closes -- or, past ``max_s``, right after a message, between two writes -- avoids that."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max_s
    messages = requests.__aiter__()
    while True:
        try:
            await asyncio.wait_for(messages.__anext__(), timeout=idle_s)
        except (StopAsyncIteration, asyncio.TimeoutError):
            return
        except Exception:  # noqa: BLE001 -- the call is gone already
            return
        if loop.time() >= deadline:
            return


def _denied(handler: grpc.RpcMethodHandler, code: grpc.StatusCode, message: str) -> grpc.RpcMethodHandler:
    async def single(_request, context):
        await context.abort(code, message)

    async def single_from_stream(requests, context):
        await _drain(requests)
        await context.abort(code, message)

    async def streaming(_request, context):
        await context.abort(code, message)
        yield  # pragma: no cover -- makes this an async generator; abort() raises

    async def streaming_from_stream(requests, context):
        await _drain(requests)
        await context.abort(code, message)
        yield  # pragma: no cover

    return _same_shape(handler, single, streaming, single_from_stream, streaming_from_stream)


def _with_claims(handler: grpc.RpcMethodHandler, claims: dict[str, Any]) -> grpc.RpcMethodHandler:
    """The handler, with the verified claims readable through ``_claims`` inside the call.
    (Real async functions: grpc.aio picks how to run a behaviour by inspecting it.)"""
    behavior = handler.stream_stream or handler.unary_stream or handler.stream_unary or handler.unary_unary

    async def single(request, context):
        _claims.set(claims)
        return await behavior(request, context)

    async def streaming(request, context):
        _claims.set(claims)
        async with aclosing(behavior(request, context)) as responses:
            async for response in responses:
                yield response

    return _same_shape(handler, single, streaming, single, streaming)


class JwtInterceptor(grpc.aio.ServerInterceptor):
    """Every AudioService call needs ``authorization: Bearer <JWT>`` that passes the REST check
    (app.auth.KeyProvider.verify) and has a ``sub``: UNAUTHENTICATED otherwise, UNAVAILABLE
    when Keycloak's JWKS cannot be fetched."""

    def __init__(self, keys: KeyProvider) -> None:
        self._keys = keys

    async def intercept_service(self, continuation, handler_call_details):
        handler = await continuation(handler_call_details)
        if handler is None or not handler_call_details.method.startswith(f"/{SERVICE_NAME}/"):
            return handler
        token = bearer_token(handler_call_details.invocation_metadata)
        try:
            if token is None:
                raise ServiceError(401, "missing_token", "metadata 'authorization: Bearer <JWT>' required")
            claims = await self._keys.verify(token)
            if not claims.get("sub"):
                raise ServiceError(401, "invalid_token", "token has no sub claim")
        except ServiceError as exc:
            code = grpc.StatusCode.UNAUTHENTICATED if exc.status_code == 401 else grpc.StatusCode.UNAVAILABLE
            logger.info("gRPC %s rejected: %s %s", handler_call_details.method, exc.error, exc.message)
            return _denied(handler, code, f"{exc.error}: {exc.message}" if exc.message else exc.error)
        return _with_claims(handler, claims)


# ---------------------------------------------------------------- the WebSocket stand-in


class GrpcSocket:
    """The part of starlette's WebSocket that app.stream uses, over one AudioChat call."""

    def __init__(self) -> None:
        self.application_state = WebSocketState.CONNECTING
        self.session_id = ""
        self.client_session_id = ""
        self.close_code: int | None = None
        self.failure: tuple[grpc.StatusCode, str] | None = None   # set by the adapter itself
        self._fatal: tuple[str, str] | None = None                 # the last fatal ERROR event
        self._incoming: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=INCOMING_MAX)
        self._out: asyncio.Queue[audio_pb2.AudioEvent] = asyncio.Queue(maxsize=OUTGOING_MAX)
        self._closed = asyncio.Event()
        self._gone = False
        self._discard = False                  # the session is over: drop what still arrives
        self._ended = False                    # the request stream has ended
        self._read = asyncio.Event()           # set on every caller message (settle)
        self._config: tuple[str, int, str] | None = None
        self._codec, self._rate, self._request_id = "", 0, ""

    # ----- caller -> session

    @staticmethod
    def config_frame(chunk: audio_pb2.AudioChunk | None) -> dict[str, Any]:
        codec = (chunk.codec if chunk is not None else "") or "pcm"
        rate = (chunk.sample_rate if chunk is not None else 0) or DEFAULT_RATES.get(codec, 0)
        frame: dict[str, Any] = {"type": "CONFIG", "codec": codec, "sample_rate": rate, "brain": "module15"}
        language = chunk.language if chunk is not None else ""
        if language:
            frame["language"] = None if language == "auto" else language
        return frame

    async def pump(self, requests, context=None) -> None:
        """Feed the caller's chunks to the session; half-close is END.  (A cancelled call also
        ends the request stream normally: ``context.cancelled()`` tells the two apart.)  Once
        the session is over, what still arrives is read and dropped (see ``settle``)."""
        first = True
        try:
            async for chunk in requests:
                self._read.set()
                if self._gone or self._discard:
                    continue
                if first:
                    first = False
                    self.client_session_id = chunk.session_id
                    self._config = (chunk.codec, chunk.sample_rate, chunk.language)
                    await self._incoming.put({"type": "websocket.receive", "text": json.dumps(self.config_frame(chunk))})
                elif self._changes_config(chunk):
                    await self.fail(grpc.StatusCode.INVALID_ARGUMENT, "invalid_config",
                                    "codec, sample_rate and language are set by the first AudioChunk")
                    continue
                if chunk.payload:
                    await self._incoming.put({"type": "websocket.receive", "bytes": chunk.payload})
            self._ended = True
            await asyncio.sleep(0)                     # let a cancellation that ended the stream land
            if self._gone or self._discard or (context is not None and context.cancelled()):
                self.disconnect()
                return
            if first:                                  # no chunk at all: a session without audio
                await self._incoming.put({"type": "websocket.receive", "text": json.dumps(self.config_frame(None))})
            await self._incoming.put({"type": "websocket.receive", "text": "END"})
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.info("gRPC session %s: request stream ended abnormally", self.session_id or "-", exc_info=True)
            self._ended = True
            self.disconnect()

    async def settle(self, pump: asyncio.Task, idle_s: float = DRAIN_IDLE_S, max_s: float = DRAIN_MAX_S) -> None:
        """Before a non-OK status: let ``pump`` read and drop what the caller already sent; returns
        as ``_drain`` decides (caller paused or half-closed, or right after a message past ``max_s``)."""
        self._discard = True
        while not self._incoming.empty():               # unblock a pump waiting to queue audio
            self._incoming.get_nowait()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max_s
        while not pump.done() and not self._ended:
            self._read.clear()
            try:
                await asyncio.wait_for(self._read.wait(), timeout=idle_s)
            except asyncio.TimeoutError:
                return
            if loop.time() >= deadline:
                return

    def _changes_config(self, chunk: audio_pb2.AudioChunk) -> bool:
        assert self._config is not None
        codec, rate, language = self._config
        return bool((chunk.codec and chunk.codec != codec) or (chunk.sample_rate and chunk.sample_rate != rate)
                    or (chunk.language and chunk.language != language))

    async def fail(self, code: grpc.StatusCode, error: str, message: str) -> None:
        """End the call with ``code`` after an ErrorEvent; the session sees a disconnect."""
        self.failure = (code, f"{error}: {message}")
        await self._out.put(audio_pb2.AudioEvent(session_id=self.session_id, error=audio_pb2.ErrorEvent(
            code=error, message=message, fatal=True)))
        self.disconnect()

    def disconnect(self) -> None:
        """The caller is gone: the session's next receive() is a disconnect, sends fail."""
        if self._gone:
            return
        self._gone = True
        while not self._incoming.empty():
            self._incoming.get_nowait()
        self._incoming.put_nowait(_DISCONNECT)

    async def accept(self) -> None:
        self.application_state = WebSocketState.CONNECTED

    async def receive(self) -> dict[str, Any]:
        if self._gone and self._incoming.empty():
            return _DISCONNECT
        return await self._incoming.get()

    # ----- session -> caller

    def _check(self) -> None:
        if self._gone or self.application_state != WebSocketState.CONNECTED:
            raise ConnectionError("gRPC call closed")

    async def send_text(self, text: str) -> None:
        self._check()
        try:
            event = json.loads(text)
        except ValueError:
            return
        out = self.translate(event)
        if out is not None:
            await self._out.put(out)

    async def send_json(self, data: dict[str, Any]) -> None:
        await self.send_text(json.dumps(data))

    async def send_bytes(self, data: bytes) -> None:
        self._check()
        await self._out.put(audio_pb2.AudioEvent(session_id=self.session_id, audio=audio_pb2.AudioResponseChunk(
            payload=data, codec=self._codec, sample_rate=self._rate, request_id=self._request_id, is_last=False)))

    async def close(self, code: int = CLOSE_NORMAL) -> None:
        if self.application_state == WebSocketState.DISCONNECTED:
            return
        self.application_state = WebSocketState.DISCONNECTED
        self.close_code = code
        self._closed.set()

    def finish(self) -> None:
        """The session is over (or never started): no more events."""
        self.application_state = WebSocketState.DISCONNECTED
        self._closed.set()

    def translate(self, event: dict[str, Any]) -> audio_pb2.AudioEvent | None:
        kind = event.get("type")
        if event.get("session_id"):
            self.session_id = event["session_id"]
        sid = self.session_id
        if kind == "READY":
            output = event.get("output") or {}
            self._codec, self._rate = output.get("codec") or "", int(output.get("sample_rate") or 0)
            return None
        if kind in ("TRANSCRIPT_PROVISIONAL", "TRANSCRIPT_FINAL"):
            final = kind == "TRANSCRIPT_FINAL"
            return audio_pb2.AudioEvent(session_id=sid, transcript=audio_pb2.TranscriptEvent(
                text=event.get("text") or "", is_final=final, utterance=int(event.get("utterance") or 0),
                confidence=float(event.get("confidence") or 0.0) if final else 0.0,
                language=(event.get("language") or "") if final else ""))
        if kind == "RESPONSE_START":
            self._request_id = event.get("request_id") or ""
            output = event.get("output") or {}
            self._codec = output.get("codec") or self._codec
            self._rate = int(output.get("sample_rate") or self._rate)
            return None
        if kind == "RESPONSE_END":
            return audio_pb2.AudioEvent(session_id=sid, audio=audio_pb2.AudioResponseChunk(
                codec=self._codec, sample_rate=self._rate, request_id=event.get("request_id") or self._request_id,
                is_last=True, interrupted=bool(event.get("interrupted"))))
        if kind in ("ERROR", "WARNING"):
            fatal = kind == "ERROR" and event.get("fatal", True) is not False
            code, message = str(event.get("code") or "error"), str(event.get("message") or "")
            if fatal:
                self._fatal = (code, message)
            return audio_pb2.AudioEvent(session_id=sid, error=audio_pb2.ErrorEvent(code=code, message=message,
                                                                                  fatal=fatal))
        if kind == "GUARDRAIL_BLOCK":
            return audio_pb2.AudioEvent(session_id=sid, error=audio_pb2.ErrorEvent(
                code="GUARDRAIL_BLOCK", message=f"{event.get('direction', '')}: {event.get('reason', '')}".strip(": "),
                fatal=False))
        return None

    async def events(self):
        """What to send the caller, until the session is over and everything is out."""
        closed = asyncio.ensure_future(self._closed.wait())
        try:
            while True:
                if not self._out.empty():
                    yield self._out.get_nowait()
                    continue
                if self._closed.is_set():
                    return
                get = asyncio.ensure_future(self._out.get())
                await asyncio.wait({get, closed}, return_when=asyncio.FIRST_COMPLETED)
                if get.done():
                    yield get.result()
                else:
                    get.cancel()
        finally:
            closed.cancel()

    def status(self) -> tuple[grpc.StatusCode, str] | None:
        """How the call ends: None = OK."""
        if self.failure is not None:
            return self.failure
        if self.close_code in (None, CLOSE_NORMAL):
            return None
        code, message = self._fatal or ("closed", f"session closed ({self.close_code})")
        status = grpc.StatusCode.RESOURCE_EXHAUSTED if code == "QUEUE_OVERFLOW" else \
            _STATUS_BY_CLOSE.get(self.close_code, grpc.StatusCode.INTERNAL)
        return status, f"{code}: {message}"


# ---------------------------------------------------------------- service


class AudioChatServicer(audio_pb2_grpc.AudioServiceServicer):
    def __init__(self, handler: StreamHandler) -> None:
        self.handler = handler

    async def AudioChat(self, request_iterator, context):
        claims = _claims.get() or {}
        user_id = str(claims.get("sub", ""))
        roles = [r for r in ((claims.get("realm_access") or {}).get("roles") or []) if r in SESSION_ROLES]
        sock = GrpcSocket()
        pump = asyncio.create_task(sock.pump(request_iterator, context))
        session = asyncio.create_task(self.handler.serve(sock, user_id, roles, resumable=False, ping=False))
        session.add_done_callback(lambda _task: sock.finish())
        logged = False
        try:
            async for event in sock.events():
                if not logged and sock.session_id:
                    logged = True
                    logger.info("gRPC AudioChat: session %s for user %s (caller's id %r)", sock.session_id,
                                user_id, sock.client_session_id)
                yield event
            status = self._status(session, sock)
            if status is not None:
                await sock.settle(pump)
        finally:
            pump.cancel()
            if not session.done():                 # finishing up -- or the caller left / the server stops
                sock.disconnect()
                with suppress(asyncio.TimeoutError, asyncio.CancelledError):
                    await asyncio.wait_for(asyncio.shield(session), timeout=2.0)
                session.cancel()
            with suppress(asyncio.CancelledError):
                await pump
        if status is not None:
            await context.abort(*status)

    def _status(self, session: asyncio.Task, sock: GrpcSocket) -> tuple[grpc.StatusCode, str] | None:
        """How the call ends once the session has sent its last event: None = OK."""
        if session.done() and not session.cancelled():
            if session.exception() is not None:
                logger.error("gRPC AudioChat session failed", exc_info=session.exception())
                return grpc.StatusCode.INTERNAL, "internal_error"
            if session.result() is not True:
                reason = session.result()
                limit = (f"RT_MAX_SESSIONS={self.handler.settings.max_sessions}" if reason == TOO_MANY_SESSIONS
                         else f"RT_MAX_SESSIONS_PER_USER={self.handler.settings.max_sessions_per_user}")
                return grpc.StatusCode.RESOURCE_EXHAUSTED, f"{reason}: {limit}"
        return sock.status()


async def start_server(handler: StreamHandler, keys: KeyProvider, settings: Settings,
                       port: int | None = None) -> tuple[grpc.aio.Server, int]:
    """Serve AudioService on ``port`` (default RT_GRPC_PORT; 0 = any free port), insecure like
    the HTTP port: TLS is the gateway's job.  Returns the server and the bound port."""
    server = grpc.aio.server(interceptors=[JwtInterceptor(keys)])
    audio_pb2_grpc.add_AudioServiceServicer_to_server(AudioChatServicer(handler), server)
    bound = server.add_insecure_port(f"[::]:{settings.grpc_port if port is None else port}")
    if not bound:
        raise RuntimeError(f"cannot listen on gRPC port {settings.grpc_port if port is None else port}")
    await server.start()
    logger.info("gRPC %s listening on :%d", SERVICE_NAME, bound)
    return server, bound
