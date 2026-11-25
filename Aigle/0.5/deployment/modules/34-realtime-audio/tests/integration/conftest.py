"""RA-10 integration suite: the real Module 34 app -- its own lifespan, real silero-VAD,
faster-whisper and TTS backend, uvicorn's WebSocket stack and the RA-6 gRPC server -- started
once per run in a background thread on free local ports.

    pytest tests/ -m integration            (inside the Module 34 image; see README "Integration tests")

Environment (all optional):
  RT_STT_MODEL / RT_STT_MODEL_DIR / RT_STT_DEVICE   as for the service; with RT_STT_DEVICE=auto and
                                       no GPU the service itself falls back to RT_STT_CPU_MODEL (tiny).
                                       CPU-only CI: RT_STT_MODEL=tiny
  RT_TTS_MODEL / RT_TTS_KOKORO_DIR     the TTS backend (default kokoro-82m in /models/tts/kokoro)
  RT_IT_CHAT_URL                       a real Module 15 (e.g. http://raptor-chat-service:8021/api/v1);
                                       default: benchmarks/fake_chat.py answering RT_IT_ANSWER after
                                       RT_IT_CHAT_DELAY_S (0.2 s), so the timings measure Module 34
  RT_IT_REDIS_URL                      a real Redis (default: fakeredis inside the process)
  RT_IT_RESPONSE_TIMEOUT_S             the RESPONSE_END deadline after END (default 8, the RA-10 bound)
  RT_IT_STARTUP_TIMEOUT_S              model loading budget (default 600)
  RT_IT_USER                           the test JWT's sub (default rt-test-ra10)
Module 23 / 25 / 26 and recording are off unless RT_GUARDRAIL_ENABLED / RT_INDEX_TRANSCRIPTS /
RT_MEMORY_ENABLED / RT_RECORDING_ENABLED say otherwise.
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest import mock

import httpx
import jwt
import numpy as np
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
ANSWER = os.environ.get("RT_IT_ANSWER", "好的，我知道了。")
RESPONSE_TIMEOUT_S = float(os.environ.get("RT_IT_RESPONSE_TIMEOUT_S", "8"))
TRANSCRIPT_TIMEOUT_S = 3.0
# the JWT sub: a test identity, in case RT_IT_CHAT_URL / Module 23 / 26 point at real services
TEST_USER = os.environ.get("RT_IT_USER", "rt-test-ra10")


# ---------------------------------------------------------------- fixtures on disk


def read_wav(name: str) -> tuple[np.ndarray, int]:
    with wave.open(str(FIXTURES / name), "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2, f"{name}: expected mono s16le"
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").copy(), w.getframerate()


def read_wav_payload(name: str) -> tuple[bytes, int, int]:
    """(data chunk, format tag, sample rate) of a RIFF/WAVE file -- for G.711 WAVs, which the
    wave module does not read."""
    raw = (FIXTURES / name).read_bytes()
    assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE", f"{name}: not a WAV file"
    pos, fmt, rate = 12, None, None
    while pos + 8 <= len(raw):
        cid, size = raw[pos:pos + 4], int.from_bytes(raw[pos + 4:pos + 8], "little")
        body = raw[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt, rate = int.from_bytes(body[0:2], "little"), int.from_bytes(body[4:8], "little")
        elif cid == b"data":
            assert fmt is not None, f"{name}: data before fmt"
            return body, fmt, rate
        pos += 8 + size + (size & 1)
    raise AssertionError(f"{name}: no data chunk")


# ---------------------------------------------------------------- the live server


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _keypair() -> tuple[bytes, str]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                serialization.NoEncryption())
    public = key.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    return private, public.decode()


@dataclass
class Live:
    http_url: str
    ws_url: str
    grpc_target: str
    private_key: bytes
    other_key: bytes            # a key Module 34 does not trust
    chat_url: str
    fake_chat: bool

    def jwt(self, sub: str | None = TEST_USER, key: bytes | None = None, exp_in: int = 300, **claims: Any) -> str:
        payload: dict[str, Any] = {"iss": "http://keycloak/realms/raptor", "exp": int(time.time()) + exp_in, **claims}
        if sub is not None:
            payload["sub"] = sub
        return jwt.encode(payload, key or self.private_key, algorithm="RS256")

    async def ticket(self, token: str | None = None) -> str:
        async with httpx.AsyncClient(base_url=self.http_url, timeout=10) as client:
            resp = await client.post("/audio/session/token",
                                     headers={"Authorization": f"Bearer {token or self.jwt()}"})
        assert resp.status_code == 200, resp.text
        return resp.json()["token"]

    def stream_url(self, ticket: str | None) -> str:
        return f"{self.ws_url}/audio/stream" + (f"?token={ticket}" if ticket is not None else "")

    async def active_sessions(self) -> int:
        async with httpx.AsyncClient(base_url=self.http_url, timeout=10) as client:
            return (await client.get("/audio/status")).json()["active_sessions"]

    async def wait_idle(self, timeout_s: float = 10.0) -> int:
        deadline = time.monotonic() + timeout_s
        while True:
            n = await self.active_sessions()
            if n == 0 or time.monotonic() > deadline:
                return n
            await asyncio.sleep(0.1)


class _FakeRedisFactory:
    """Stands in for redis.asyncio.Redis in app.main when no RT_IT_REDIS_URL is given."""

    @staticmethod
    def from_url(*_args, **_kwargs):
        from fakeredis import aioredis

        return aioredis.FakeRedis()


@pytest.fixture(scope="session")
def live() -> Live:
    import uvicorn

    private, public = _keypair()
    other, _ = _keypair()
    http_port, grpc_port = _free_port(), _free_port()
    real_chat = os.environ.get("RT_IT_CHAT_URL", "").rstrip("/")
    chat_port = None if real_chat else _free_port()
    chat_url = real_chat or f"http://127.0.0.1:{chat_port}/api/v1"
    env = {"RT_JWT_PUBLIC_KEY": public, "RT_JWT_AUDIENCE": "", "RT_GRPC_ENABLED": "true",
           "RT_GRPC_PORT": str(grpc_port), "RT_CHAT_SERVICE_URL": chat_url}
    defaults = {"RT_GUARDRAIL_ENABLED": "false", "RT_INDEX_TRANSCRIPTS": "false", "RT_MEMORY_ENABLED": "false",
                "RT_RECORDING_ENABLED": "false", "RT_MAX_SESSIONS": "4", "RT_VAD_POOL_SIZE": "2",
                "LOG_LEVEL": "WARNING"}
    redis_url = os.environ.get("RT_IT_REDIS_URL")
    if redis_url:
        env["REDIS_URL"] = redis_url
    saved = {k: os.environ.get(k) for k in [*env, *defaults]}
    os.environ.update(env)
    for k, v in defaults.items():
        os.environ.setdefault(k, v)

    from app import main
    from app.config import get_settings

    get_settings.cache_clear()
    patches = [] if redis_url else [mock.patch.object(main, "Redis", _FakeRedisFactory)]
    for p in patches:
        p.start()
    servers = [uvicorn.Server(uvicorn.Config(main.create_app(), host="127.0.0.1", port=http_port,
                                             log_level="warning", lifespan="on"))]
    if chat_port is not None:
        from benchmarks.fake_chat import create_app as fake_chat_app

        delay = float(os.environ.get("RT_IT_CHAT_DELAY_S", "0.2"))
        servers.append(uvicorn.Server(uvicorn.Config(fake_chat_app(ANSWER, delay, 0.0), host="127.0.0.1",
                                                     port=chat_port, log_level="warning", lifespan="off")))
    failure: list[BaseException] = []

    def run() -> None:
        async def serve_all():
            await asyncio.gather(*(s.serve() for s in servers))

        try:
            asyncio.run(serve_all())
        except BaseException as exc:  # noqa: BLE001 -- reported to the waiting fixture
            failure.append(exc)

    thread = threading.Thread(target=run, name="module34-it", daemon=True)
    thread.start()
    budget = float(os.environ.get("RT_IT_STARTUP_TIMEOUT_S", "600"))
    deadline = time.monotonic() + budget
    while not all(s.started for s in servers):
        if not thread.is_alive() or time.monotonic() > deadline:
            for s in servers:
                s.should_exit = True
            pytest.fail(f"Module 34 did not start within {budget:.0f} s (models, TTS weights, ports?): "
                        f"{failure[0] if failure else 'see the log above'}", pytrace=False)
        time.sleep(0.2)
    try:
        yield Live(http_url=f"http://127.0.0.1:{http_port}", ws_url=f"ws://127.0.0.1:{http_port}",
                   grpc_target=f"127.0.0.1:{grpc_port}", private_key=private, other_key=other,
                   chat_url=chat_url, fake_chat=chat_port is not None)
    finally:
        for s in servers:
            s.should_exit = True
        thread.join(timeout=30)
        for p in patches:
            p.stop()
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        get_settings.cache_clear()


# ---------------------------------------------------------------- client helpers


async def paced(chunks: list[bytes], chunk_ms: float, send) -> float:
    """Send ``chunks`` at playback speed (a live microphone); returns when the last one went."""
    start = time.monotonic()
    for i, chunk in enumerate(chunks):
        await send(chunk)
        delay = start + (i + 1) * chunk_ms / 1000 - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
    return time.monotonic()


def frames(data: bytes, size: int) -> list[bytes]:
    return [data[i:i + size] for i in range(0, len(data), size)]


@dataclass
class Received:
    t: float
    event: dict[str, Any] | None       # JSON event, or None for a binary frame
    data: bytes | None = None


async def read_all(ws, into: list[Received]) -> tuple[int | None, str]:
    """Every server frame with its arrival time, until the server closes; (close code, reason)."""
    from websockets.exceptions import ConnectionClosed

    try:
        async for message in ws:
            now = time.monotonic()
            if isinstance(message, bytes):
                into.append(Received(now, None, message))
            else:
                into.append(Received(now, json.loads(message)))
    except ConnectionClosed:
        pass
    rcvd = ws.close_code if ws.close_code is not None else None
    return rcvd, ws.close_reason or ""


def of_type(received: list[Received], kind: str) -> list[Received]:
    return [r for r in received if r.event is not None and r.event.get("type") == kind]
