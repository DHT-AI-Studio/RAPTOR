"""Fixtures for Module 34 unit tests: fakeredis (with Lua), a self-signed RS256 key pair and an
amplitude-based stand-in for silero-VAD, so no torch / Keycloak / Redis server is needed."""
from __future__ import annotations

import time
from dataclasses import replace
from typing import Any, ClassVar

import fakeredis
import httpx
import jwt
import numpy as np
import pytest
from app.config import Settings
from app.main import build_services, create_app
from app.stt import Transcript, Word
from app.tts_backends import TtsBackend, VoiceInfo
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fakeredis import aioredis
from starlette.testclient import TestClient

SR = 16000


def _keypair():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                serialization.NoEncryption())
    public = key.public_key().public_bytes(serialization.Encoding.PEM,
                                           serialization.PublicFormat.SubjectPublicKeyInfo)
    return private, public.decode()


PRIVATE_KEY, PUBLIC_KEY = _keypair()
OTHER_PRIVATE_KEY, _ = _keypair()


def make_jwt(sub: str | None = "user-1", exp_in: int = 300, key: bytes = PRIVATE_KEY, **claims: Any) -> str:
    payload: dict[str, Any] = {"iss": "http://kc/realms/raptor", "exp": int(time.time()) + exp_in, **claims}
    if sub is not None:
        payload["sub"] = sub
    return jwt.encode(payload, key, algorithm="RS256")


class AmplitudeVad:
    """Deterministic VAD stand-in: P(speech)=0.9 when window RMS > 0.05, else 0.05."""

    instances: ClassVar[list[AmplitudeVad]] = []

    def __init__(self, delay_s: float = 0.0) -> None:
        self.delay_s = delay_s
        self.resets = 0
        self.calls = 0
        AmplitudeVad.instances.append(self)

    def reset(self) -> None:
        self.resets += 1

    def __call__(self, window: np.ndarray) -> float:
        self.calls += 1
        if self.delay_s:
            time.sleep(self.delay_s)
        return 0.9 if float(np.sqrt(np.mean(window ** 2))) > 0.05 else 0.05


class FakeWhisper:
    """Stand-in STT backend.  Returns ``words`` (text, start_ms, end_ms relative to the audio it
    is given) for the final decode (beam_size > 1) and ``provisional`` for rolling decodes,
    cut to the audio length; records every call."""

    name, device, compute_type = "fake-whisper", "cpu", "int8"

    def __init__(self, words=None, provisional=None, language="zh", delay_s: float = 0.0,
                 fail: bool = False) -> None:
        self.words = words if words is not None else [("你", 0, 150), ("好", 150, 300), ("嗎", 300, 450)]
        self.provisional = provisional if provisional is not None else self.words
        self.language = language
        self.delay_s = delay_s
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    def transcribe(self, audio, language, *, beam_size, prompt=None, on_words=None, max_new_tokens=None,
                   fallback=True) -> Transcript:
        self.calls.append({"samples": len(audio), "language": language, "beam_size": beam_size,
                           "prompt": prompt, "dtype": str(audio.dtype), "max_new_tokens": max_new_tokens,
                           "fallback": fallback})
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.fail:
            raise RuntimeError("decoder exploded")
        ms = len(audio) * 1000 / SR
        spec = self.words if beam_size > 1 else self.provisional
        words = tuple(Word(t, s, e, 0.9) for t, s, e in spec if e <= ms)
        if on_words:
            for w in words:                           # one "segment" per word
                on_words([w], language or self.language)
        return Transcript("".join(w.text for w in words), words, language or self.language, 0.99)


class FakeTts(TtsBackend):
    """Stand-in TTS backend: 50 ms of 24 kHz tone per character, produced in two pieces."""

    sample_rate = 24000

    def __init__(self, settings=None, fail: bool = False) -> None:
        super().__init__(settings)
        self.fail = fail
        self.calls: list[tuple[str, str, str | None]] = []

    @classmethod
    def name(cls) -> str:
        return "kokoro-82m"            # stands in for the default backend in GET /tts/models

    def load(self) -> None:
        pass

    def list_voices(self):
        return [VoiceInfo("fake-f", "zh", "female"), VoiceInfo("fake-m", "zh", "male")]

    @property
    def default_voice(self) -> str:
        return "fake-f"

    def stream(self, text, voice, language, speed):
        self.calls.append((text, voice, language))
        if self.fail:
            raise RuntimeError("synthesis exploded")
        n = int(self.sample_rate * 0.05 * len(text))
        tone = (np.sin(2 * np.pi * 300 * np.arange(n) / self.sample_rate) * 0.3).astype(np.float32)
        yield tone[: n // 2]
        yield tone[n // 2:]

    def synthesize(self, text, voice, language, speed):
        return np.concatenate(list(self.stream(text, voice, language, speed)))


def sse(*events: tuple[str, dict]) -> bytes:
    import json

    return "".join(f"event: {e}\ndata: {json.dumps(d, ensure_ascii=False)}\n\n" for e, d in events).encode()


class FakeChat:
    """Module 15 behind httpx.MockTransport: answers ``answer`` as SSE (or JSON), records requests."""

    def __init__(self, answer: str = "你好。今天天氣很好！", stream: bool = True, delay_s: float = 0.0,
                 status: int = 200) -> None:
        self.answer, self.stream, self.delay_s, self.status = answer, stream, delay_s, status
        self.requests: list[dict[str, Any]] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        import asyncio
        import json

        body = json.loads(request.content)
        self.requests.append({"url": str(request.url), "body": body, "headers": dict(request.headers),
                              "request_id": request.headers.get("x-request-id")})
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.status >= 400:
            return httpx.Response(self.status, json={"detail": "boom"})
        rid = request.headers.get("x-request-id")
        if not self.stream:
            return httpx.Response(200, json={"response": self.answer, "request_id": rid})
        deltas = [("delta", {"text": part}) for part in self.answer.split("|")]
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              content=sse(("start", {"request_id": rid}), *deltas,
                                          ("end", {"request_id": rid, "response": self.answer})))


def speech(ms: int, freq: float = 220.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(SR * ms / 1000)) / SR
    return (np.sin(2 * np.pi * freq * t) * amp * 32767).astype(np.int16)


def silence(ms: int) -> np.ndarray:
    return np.zeros(int(SR * ms / 1000), dtype=np.int16)


def make_settings(**overrides: Any) -> Settings:
    # resume_window_s=0: a dropped test connection ends its session at once (the resume tests
    # set a window); otherwise the server side would wait for a RESUME that never comes.
    # Leave ping_interval_s at its real default here -- it doubles as an accidental watchdog:
    # Session never answers PING on its own, so a test that hangs waiting on an event that never
    # arrives still gets force-closed (pong_timeout, 4408) instead of blocking CI forever. A test
    # that's long enough to legitimately cross it on its own should disable it locally instead.
    # Module 23 screening and Module 25 indexing are off unless a test turns them on.
    base = Settings(redis_url="redis://unused", jwt_public_key=PUBLIC_KEY, heartbeat_s=0.2,
                    vad_pool_size=1, resume_window_s=0, guardrail_enabled=False, index_transcripts=False,
                    gd_internal_token="test-internal-token")
    return replace(base, **overrides)


def bare(event: dict[str, Any]) -> dict[str, Any]:
    """A server event without the framing fields every event carries (seq, session_id)."""
    return {k: v for k, v in event.items() if k not in ("seq", "session_id")}


@pytest.fixture
def server() -> fakeredis.FakeServer:
    return fakeredis.FakeServer()


@pytest.fixture
def sync_redis(server) -> fakeredis.FakeRedis:
    """Synchronous view on the same fake server, for assertions from test code."""
    return fakeredis.FakeRedis(server=server)


@pytest.fixture
def make_client(server):
    clients: list[TestClient] = []

    def _make(settings: Settings | None = None, vad_delay_s: float = 0.0, sink=None,
              stt: FakeWhisper | None = None, tts: FakeTts | None = None,
              chat: FakeChat | None = None, services: dict | None = None, kafka=None, blobs=None,
              recordings_db=None) -> TestClient:
        """``services``: URL path prefix -> MockTransport handler (Modules 23 / 26 / 33); other
        requests go to ``chat`` (Module 15)."""
        AmplitudeVad.instances = []
        app = create_app(use_lifespan=False)
        redis = aioredis.FakeRedis(server=server)

        async def route(request: httpx.Request) -> httpx.Response:
            for prefix, handler in (services or {}).items():
                if request.url.path.startswith(prefix):
                    return await handler(request)
            if chat is not None:
                return await chat(request)
            return httpx.Response(404)

        http = httpx.AsyncClient(transport=httpx.MockTransport(route)) if chat is not None or services else None
        build_services(app, settings or make_settings(), redis, lambda: AmplitudeVad(vad_delay_s), sink=sink,
                       stt_loader=(lambda _name: stt) if stt is not None else None,
                       tts_loader=(lambda _name, _settings: tts) if tts is not None else None, http=http,
                       kafka=kafka, blobs=blobs, recordings_db=recordings_db)
        if stt is not None:
            app.state.stt.final = app.state.stt.provisional = stt     # what start() would load
        if tts is not None:
            app.state.tts.backend, app.state.tts.voice = tts, tts.default_voice
        client = TestClient(app)
        client.__enter__()
        clients.append(client)
        return client

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture
def client(make_client) -> TestClient:
    return make_client()


def issue_ticket(client: TestClient, sub: str = "user-1", roles: list[str] | None = None) -> str:
    claims = {"realm_access": {"roles": roles}} if roles else {}
    resp = client.post("/audio/session/token", headers={"Authorization": f"Bearer {make_jwt(sub, **claims)}"})
    assert resp.status_code == 200, resp.text
    return resp.json()["token"]


PCM_CONFIG = {"type": "CONFIG", "codec": "pcm", "sample_rate": 16000, "language": "zh-TW"}


class FakeKafka:
    """Records what would be published."""

    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.sent: list[tuple[str, dict, str | None]] = []

    async def send(self, topic, value, key=None):
        self.sent.append((topic, value, key))
        return self.ok

    async def stop(self):
        pass

    def topic(self, name):
        return [v for t, v, _ in self.sent if t == name]


class FakeBlobs:
    """In-memory SeaweedFS."""

    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}
        self.n = 0

    async def put(self, data, name, mime="application/octet-stream"):
        self.n += 1
        fid = f"3,{self.n:08x}"
        self.blobs[fid] = bytes(data)
        return fid

    async def get(self, fid):
        from app.blobstore import BlobError

        if fid not in self.blobs:
            raise BlobError(f"{fid} not found")
        return self.blobs[fid]

    async def delete(self, fid):
        return len(self.blobs.pop(fid, b""))


class FakeRecordingsDb:
    """In-memory rt_audio_recordings."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    async def insert(self, user_id, session_id, fid, duration_s, size_bytes, retention_days):
        import uuid
        from datetime import datetime, timedelta, timezone

        if any(r["session_id"] == session_id for r in self.rows):
            return False
        now = datetime.now(timezone.utc)
        self.rows.append({"id": str(uuid.uuid4()), "user_id": user_id, "session_id": session_id, "fid": fid,
                          "duration_s": duration_s, "size_bytes": size_bytes, "created_at": now,
                          "expires_at": now + timedelta(days=retention_days), "erased_at": None})
        return True

    def _rec(self, r):
        from app.recordings_db import Recording

        return Recording(r["id"], r["user_id"], r["session_id"], r["fid"], r["duration_s"], r["size_bytes"],
                         r["created_at"], r["expires_at"])

    async def live_for_user(self, user_id):
        return [self._rec(r) for r in self.rows if r["user_id"] == user_id and r["erased_at"] is None]

    async def expired(self, limit=100):
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        return [self._rec(r) for r in self.rows if r["erased_at"] is None and r["expires_at"] < now][:limit]

    async def mark_erased(self, ids):
        from datetime import datetime, timezone

        for r in self.rows:
            if r["id"] in ids:
                r["erased_at"] = datetime.now(timezone.utc)

    async def close(self):
        pass


def json_handler(fn):
    """MockTransport handler from ``fn(request, body) -> (status, json)``; records the calls."""
    import json

    calls: list = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        calls.append({"url": str(request.url), "method": request.method, "headers": dict(request.headers),
                      "body": body})
        status, payload = fn(request, body)
        return httpx.Response(status, json=payload)

    handler.calls = calls
    return handler
