"""RA-6: AudioService.AudioChat on a real grpc.aio server (port 0), with the unit-test stand-ins
for silero-VAD, Whisper, the TTS backend, Module 15 and Redis."""
from __future__ import annotations

import asyncio
import time

import httpx
import numpy as np
import pytest

grpc = pytest.importorskip("grpc")
audio_pb2 = pytest.importorskip("raptor.audio.v1.audio_pb2")

from app.codec import g711_encode
from app.grpc_audio import GrpcSocket, bearer_token, start_server
from app.main import build_services, create_app
from fakeredis import aioredis
from raptor.audio.v1 import audio_pb2_grpc

from .conftest import (
    OTHER_PRIVATE_KEY,
    AmplitudeVad,
    FakeChat,
    FakeTts,
    FakeWhisper,
    make_jwt,
    make_settings,
    silence,
    speech,
)

CHUNK = 320 * 2   # 20 ms of 16 kHz s16le


@pytest.fixture
async def make_grpc(server):
    """``make(settings=..., chat=..., stt=..., tts=...)`` -> (app, AudioServiceStub)."""
    running: list = []

    async def _make(settings=None, stt: FakeWhisper | None = None, tts: FakeTts | None = None,
                    chat: FakeChat | None = None):
        settings = settings or make_settings(output_pace=False)
        app = create_app(use_lifespan=False)
        http = httpx.AsyncClient(transport=httpx.MockTransport(chat)) if chat is not None else None
        build_services(app, settings, aioredis.FakeRedis(server=server), AmplitudeVad,
                       stt_loader=(lambda _n: stt) if stt is not None else None,
                       tts_loader=(lambda _n, _s: tts) if tts is not None else None, http=http)
        if stt is not None:
            app.state.stt.final = app.state.stt.provisional = stt
        if tts is not None:
            app.state.tts.backend, app.state.tts.voice = tts, tts.default_voice
        srv, port = await start_server(app.state.stream_handler, app.state.key_provider, settings, port=0)
        channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
        running.append((srv, channel, http))
        return app, audio_pb2_grpc.AudioServiceStub(channel)

    yield _make
    for srv, channel, http in running:
        await channel.close()
        await srv.stop(None)
        if http is not None:
            await http.aclose()


def auth(token: str | None = None) -> tuple[tuple[str, str], ...]:
    return (("authorization", f"Bearer {token or make_jwt()}"),)


def pcm_chunks(pcm: np.ndarray, first: dict | None = None):
    data = pcm.astype("<i2").tobytes()
    out = []
    for i in range(0, len(data), CHUNK):
        fields = (first or {"codec": "pcm", "sample_rate": 16000, "language": "zh-TW"}) if i == 0 else {}
        out.append(audio_pb2.AudioChunk(payload=data[i:i + CHUNK], **fields))
    return out


async def requests(chunks, delay_s: float = 0.0):
    for c in chunks:
        yield c
        if delay_s:
            await asyncio.sleep(delay_s)


async def collect(call) -> tuple[list, grpc.StatusCode | None, str]:
    events = []
    try:
        async for ev in call:
            events.append(ev)
    except grpc.aio.AioRpcError as exc:
        return events, exc.code(), exc.details()
    return events, grpc.StatusCode.OK, ""


def utterance() -> np.ndarray:
    return np.concatenate([silence(200), speech(800), silence(700)])


async def test_voice_turn_transcript_and_spoken_answer(make_grpc):
    chat = FakeChat()
    app, stub = await make_grpc(stt=FakeWhisper(), tts=FakeTts(), chat=chat)
    events, code, details = await collect(stub.AudioChat(requests(pcm_chunks(utterance())), metadata=auth()))
    assert code == grpc.StatusCode.OK, details
    finals = [e.transcript for e in events if e.WhichOneof("event") == "transcript" and e.transcript.is_final]
    assert [t.text for t in finals] == ["你好嗎"]
    assert finals[0].utterance == 1 and finals[0].language == "zh" and finals[0].confidence > 0
    audio = [e.audio for e in events if e.WhichOneof("event") == "audio"]
    assert audio and audio[-1].is_last and not audio[-1].payload and not audio[-1].interrupted
    body = [a for a in audio if not a.is_last]
    assert body and all(a.payload for a in body)
    assert {a.codec for a in audio} == {"pcm"} and {a.sample_rate for a in audio} == {16000}
    assert len({a.request_id for a in audio}) == 1 and audio[0].request_id
    assert sum(a.is_last for a in audio) == 1
    # the transcript comes before the answer, every event names the session
    kinds = [e.WhichOneof("event") for e in events]
    assert kinds.index("transcript") < kinds.index("audio")
    session_ids = {e.session_id for e in events}
    assert len(session_ids) == 1 and next(iter(session_ids))
    assert chat.requests and chat.requests[0]["body"]["message"] == "你好嗎"
    assert await app.state.sessions.active_count() == 0


async def test_rolling_transcripts_are_not_final(make_grpc):
    settings = make_settings(output_pace=False, stt_first_decode_ms=200, stt_stride_s=0.2)
    _, stub = await make_grpc(settings=settings, stt=FakeWhisper(), tts=FakeTts(), chat=FakeChat())
    chunks = pcm_chunks(np.concatenate([speech(1200), silence(700)]))
    events, code, _ = await collect(stub.AudioChat(requests(chunks, delay_s=0.02), metadata=auth()))
    assert code == grpc.StatusCode.OK
    transcripts = [e.transcript for e in events if e.WhichOneof("event") == "transcript"]
    rolling = [t for t in transcripts if not t.is_final]
    assert rolling and all(t.confidence == 0 and t.utterance == 1 for t in rolling)
    assert transcripts[-1].is_final


async def test_pcmu_caller_gets_pcmu_answer(make_grpc):
    _, stub = await make_grpc(stt=FakeWhisper(), tts=FakeTts(), chat=FakeChat())
    pcm8k = utterance()[::2]
    data = g711_encode(pcm8k, "pcmu")
    chunks = [audio_pb2.AudioChunk(payload=data[i:i + 160], **({"codec": "pcmu"} if i == 0 else {}))
              for i in range(0, len(data), 160)]
    events, code, details = await collect(stub.AudioChat(requests(chunks), metadata=auth()))
    assert code == grpc.StatusCode.OK, details
    body = [e.audio for e in events if e.WhichOneof("event") == "audio" and not e.audio.is_last]
    assert body and {(a.codec, a.sample_rate) for a in body} == {("pcmu", 8000)}
    assert {len(a.payload) for a in body} == {160}                        # 20 ms G.711 frames


async def test_empty_call_ends_ok_without_events(make_grpc):
    app, stub = await make_grpc(stt=FakeWhisper(), tts=FakeTts(), chat=FakeChat())
    events, code, details = await collect(stub.AudioChat(requests([]), metadata=auth()))
    assert (events, code) == ([], grpc.StatusCode.OK), details
    events, code, _ = await collect(stub.AudioChat(requests([audio_pb2.AudioChunk(codec="pcm")]), metadata=auth()))
    assert (events, code) == ([], grpc.StatusCode.OK)
    assert await app.state.sessions.active_count() == 0


@pytest.mark.parametrize("token, detail", [
    (None, "missing_token"),
    (make_jwt(key=OTHER_PRIVATE_KEY), "invalid_token"),
    (make_jwt(exp_in=-10), "token_expired"),
    (make_jwt(sub=None), "invalid_token"),
])
async def test_bad_token_is_unauthenticated(make_grpc, token, detail):
    app, stub = await make_grpc(stt=FakeWhisper(), tts=FakeTts(), chat=FakeChat())
    metadata = auth(token) if token else ()
    # sent in one burst: the refusal waits for the writes in flight (app.grpc_audio._drain), else a
    # grpc.aio client would see INTERNAL "Internal error from Core" instead of the status
    events, code, details = await collect(stub.AudioChat(requests(pcm_chunks(utterance())), metadata=metadata))
    assert code == grpc.StatusCode.UNAUTHENTICATED and events == []
    assert details.startswith(detail)
    assert await app.state.sessions.active_count() == 0


async def test_refused_live_caller_hears_back_within_the_drain_budget(make_grpc):
    """A refused caller that keeps streaming (never pauses) gets its status after DRAIN_MAX_S."""
    from app.grpc_audio import DRAIN_MAX_S

    _, stub = await make_grpc(stt=FakeWhisper(), tts=FakeTts(), chat=FakeChat())
    endless = pcm_chunks(np.concatenate([speech(400), silence(400)] * 10))
    t0 = time.monotonic()
    events, code, _ = await collect(stub.AudioChat(requests(endless, delay_s=0.02), metadata=()))
    assert code == grpc.StatusCode.UNAUTHENTICATED and events == []
    assert time.monotonic() - t0 < DRAIN_MAX_S + 1.0


async def test_unsupported_codec_is_invalid_argument(make_grpc):
    _, stub = await make_grpc(stt=FakeWhisper(), tts=FakeTts(), chat=FakeChat())
    chunk = audio_pb2.AudioChunk(payload=b"\0" * 320, codec="mp3", sample_rate=16000)
    events, code, details = await collect(stub.AudioChat(requests([chunk]), metadata=auth()))
    assert code == grpc.StatusCode.INVALID_ARGUMENT
    errors = [e.error for e in events if e.WhichOneof("event") == "error"]
    assert [(e.code, e.fatal) for e in errors] == [("UNSUPPORTED_CODEC", True)]
    assert details.startswith("UNSUPPORTED_CODEC")


async def test_bad_sample_rate_and_language(make_grpc):
    _, stub = await make_grpc(stt=FakeWhisper(), tts=FakeTts(), chat=FakeChat())
    for first in ({"codec": "opus", "sample_rate": 44100}, {"codec": "pcm", "language": "not a tag!"}):
        events, code, _ = await collect(stub.AudioChat(requests([audio_pb2.AudioChunk(**first)]), metadata=auth()))
        assert code == grpc.StatusCode.INVALID_ARGUMENT
        assert [e.error.code for e in events] == ["invalid_config"]


async def test_changing_codec_mid_call_is_invalid_argument(make_grpc):
    app, stub = await make_grpc(stt=FakeWhisper(), tts=FakeTts(), chat=FakeChat())
    chunks = pcm_chunks(speech(200)) + [audio_pb2.AudioChunk(payload=b"\xff" * 160, codec="pcmu")]
    events, code, details = await collect(stub.AudioChat(requests(chunks), metadata=auth()))
    assert code == grpc.StatusCode.INVALID_ARGUMENT and details.startswith("invalid_config")
    assert [(e.error.code, e.error.fatal) for e in events if e.WhichOneof("event") == "error"] == [
        ("invalid_config", True)]
    for _ in range(50):
        if await app.state.sessions.active_count() == 0:
            break
        await asyncio.sleep(0.05)
    assert await app.state.sessions.active_count() == 0


async def test_full_server_is_resource_exhausted(make_grpc):
    # heartbeat_s=5: the first session's start may block the loop for a moment (first imports),
    # longer than the 3 x 0.2 s after which make_settings' heartbeat would count its slot stale
    app, stub = await make_grpc(settings=make_settings(output_pace=False, max_sessions=1, heartbeat_s=5),
                                stt=FakeWhisper(), tts=FakeTts(), chat=FakeChat())
    hold = asyncio.Event()

    async def held():
        yield pcm_chunks(silence(100))[0]
        await hold.wait()

    first = stub.AudioChat(held(), metadata=auth())
    for _ in range(100):
        if await app.state.sessions.active_count() == 1:
            break
        await asyncio.sleep(0.02)
    events, code, details = await collect(stub.AudioChat(requests(pcm_chunks(utterance())), metadata=auth()))
    assert code == grpc.StatusCode.RESOURCE_EXHAUSTED and events == []
    assert "too_many_sessions" in details
    hold.set()
    _, code, _ = await collect(first)
    assert code == grpc.StatusCode.OK


async def test_cancelled_call_ends_the_session(make_grpc):
    app, stub = await make_grpc(settings=make_settings(output_pace=False, heartbeat_s=5),
                                stt=FakeWhisper(), tts=FakeTts(), chat=FakeChat())
    forever = asyncio.Event()

    async def talking():
        for c in pcm_chunks(speech(300)):
            yield c
        await forever.wait()

    call = stub.AudioChat(talking(), metadata=auth())
    for _ in range(100):
        if await app.state.sessions.active_count() == 1:
            break
        await asyncio.sleep(0.02)
    handler = app.state.stream_handler
    session_id = next(iter(handler.live))
    call.cancel()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and (handler.live or await app.state.sessions.active_count()):
        await asyncio.sleep(0.05)
    assert not handler.live and await app.state.sessions.active_count() == 0
    record = await app.state.sessions.get(session_id)
    assert record["state"] == "ENDED" and record["end_reason"] == "disconnect"   # no RESUME window for gRPC


async def test_roles_from_the_token_reach_the_session(make_grpc):
    """rt-delegate in the JWT is kept like the WS ticket does (subject_id itself has no gRPC field)."""
    app, stub = await make_grpc(stt=FakeWhisper(), tts=FakeTts(), chat=FakeChat())
    token = make_jwt("svc-1", realm_access={"roles": ["rt-delegate", "offline_access"]})
    seen = {}
    original = app.state.stream_handler.serve

    async def spy(ws, user_id, roles=(), **kw):
        seen.update(user_id=user_id, roles=list(roles), **kw)
        return await original(ws, user_id, roles, **kw)

    app.state.stream_handler.serve = spy
    _, code, _ = await collect(stub.AudioChat(requests([]), metadata=auth(token)))
    assert code == grpc.StatusCode.OK
    assert seen == {"user_id": "svc-1", "roles": ["rt-delegate"], "resumable": False, "ping": False}


# ----- the adapter on its own


def test_bearer_token_from_metadata():
    assert bearer_token((("Authorization", "Bearer abc"),)) == "abc"
    assert bearer_token((("authorization", "bearer  abc "),)) == "abc"
    assert bearer_token((("authorization", "Basic abc"),)) is None
    assert bearer_token((("authorization", "Bearer "),)) is None
    assert bearer_token(None) is None


def test_config_frame_defaults():
    frame = GrpcSocket.config_frame
    assert frame(None) == {"type": "CONFIG", "codec": "pcm", "sample_rate": 16000, "brain": "module15"}
    assert frame(audio_pb2.AudioChunk(codec="opus"))["sample_rate"] == 48000
    assert frame(audio_pb2.AudioChunk(codec="pcmu"))["sample_rate"] == 8000
    assert frame(audio_pb2.AudioChunk(codec="pcm", sample_rate=8000))["sample_rate"] == 8000
    assert frame(audio_pb2.AudioChunk(language="auto"))["language"] is None
    assert frame(audio_pb2.AudioChunk(language="en"))["language"] == "en"
    assert "language" not in frame(audio_pb2.AudioChunk())


def test_translate_events():
    sock = GrpcSocket()
    t = sock.translate
    assert t({"type": "READY", "session_id": "s1", "output": {"codec": "pcm", "sample_rate": 16000}}) is None
    assert t({"type": "STATE", "state": "LISTENING"}) is None
    final = t({"type": "TRANSCRIPT_FINAL", "utterance": 2, "text": "好", "confidence": 0.9, "language": "zh"})
    assert final.session_id == "s1" and final.transcript.is_final and final.transcript.utterance == 2
    rolling = t({"type": "TRANSCRIPT_PROVISIONAL", "utterance": 2, "text": "好", "stable_text": ""})
    assert not rolling.transcript.is_final and rolling.transcript.confidence == 0
    assert t({"type": "RESPONSE_START", "request_id": "r1", "output": {"codec": "pcmu", "sample_rate": 8000}}) is None
    end = t({"type": "RESPONSE_END", "request_id": "r1", "interrupted": True}).audio
    assert (end.is_last, end.interrupted, end.codec, end.sample_rate, end.request_id) == (True, True, "pcmu", 8000, "r1")
    warning = t({"type": "WARNING", "code": "QUEUE_OVERFLOW", "message": "m"}).error
    assert (warning.code, warning.fatal) == ("QUEUE_OVERFLOW", False)
    soft = t({"type": "ERROR", "code": "STT_OVERLOADED", "message": "m", "fatal": False}).error
    assert not soft.fatal
    blocked = t({"type": "GUARDRAIL_BLOCK", "direction": "input", "reason": "policy"}).error
    assert (blocked.code, blocked.message, blocked.fatal) == ("GUARDRAIL_BLOCK", "input: policy", False)
    assert sock.status() is None
    hard = t({"type": "ERROR", "code": "QUEUE_OVERFLOW", "message": "client too slow"}).error
    assert hard.fatal
    sock.close_code = 1011
    assert sock.status() == (grpc.StatusCode.RESOURCE_EXHAUSTED, "QUEUE_OVERFLOW: client too slow")
