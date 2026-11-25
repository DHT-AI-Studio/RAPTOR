"""INF-10/11 (VIE01-316/317): /inference/voice/stt and /tts forward to Module 34's already
loaded Whisper/TTS models instead of loading a second copy in this service."""
from __future__ import annotations

import httpx

from .test_diarize import wav

ALICE = {"x-user-id": "alice"}


async def test_stt_forwards_normalized_wav_and_the_internal_token(client, upstream):
    resp = await client.post("/inference/voice/stt", files={"audio": ("a.wav", wav())},
                             data={"language": "zh-TW"}, headers=ALICE)
    assert resp.status_code == 200
    assert resp.json() == {
        "text": upstream.transcribe_text, "language": upstream.transcribe_language,
        "language_probability": 0.99, "confidence": 0.9,
    }
    assert upstream.module34_calls[-1] == ("/audio/transcribe", "test-internal-token", "alice")


async def test_stt_without_language_omits_it(client, upstream):
    resp = await client.post("/inference/voice/stt", files={"audio": ("a.wav", wav())}, headers=ALICE)
    assert resp.status_code == 200


async def test_stt_without_user_id_is_401(client, upstream):
    resp = await client.post("/inference/voice/stt", files={"audio": ("a.wav", wav())})
    assert resp.status_code == 401
    assert resp.json()["error"] == "missing_user_identity"
    assert upstream.module34_calls == []


async def test_stt_module34_down_is_503(client, upstream):
    upstream.module34_down = True
    resp = await client.post("/inference/voice/stt", files={"audio": ("a.wav", wav())}, headers=ALICE)
    assert resp.status_code == 503
    assert resp.json()["error"] == "module34_unavailable"


async def test_stt_module34_error_is_mapped(client, upstream):
    upstream.module34_status = 400
    resp = await client.post("/inference/voice/stt", files={"audio": ("a.wav", wav())}, headers=ALICE)
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_audio"


async def test_stt_module34_quota_exceeded_is_429(client, upstream):
    upstream.module34_status = 429
    upstream.module34_detail = "too_many_sessions_for_user"
    resp = await client.post("/inference/voice/stt", files={"audio": ("a.wav", wav())}, headers=ALICE)
    assert resp.status_code == 429
    assert resp.json()["error"] == "too_many_sessions_for_user"


# ---------------------------------------------------------------- /tts

async def test_tts_returns_the_upstream_wav_bytes(client, upstream):
    resp = await client.post("/inference/voice/tts", data={"text": "你好", "language": "zh-TW"}, headers=ALICE)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert resp.content == upstream.synthesize_wav
    assert upstream.module34_calls[-1] == ("/audio/synthesize", "test-internal-token", "alice")


async def test_tts_without_user_id_is_401(client, upstream):
    resp = await client.post("/inference/voice/tts", data={"text": "hi"})
    assert resp.status_code == 401
    assert resp.json()["error"] == "missing_user_identity"
    assert upstream.module34_calls == []


async def test_tts_module34_down_is_503(client, upstream):
    upstream.module34_down = True
    resp = await client.post("/inference/voice/tts", data={"text": "hi"}, headers=ALICE)
    assert resp.status_code == 503
    assert resp.json()["error"] == "module34_unavailable"


async def test_tts_module34_timeout_is_504(client):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/audio/synthesize":
            raise httpx.ReadTimeout("slow", request=request)
        raise AssertionError(f"unexpected call: {request.url}")

    client.app.state.module34._client._transport = httpx.MockTransport(handler)
    resp = await client.post("/inference/voice/tts", data={"text": "hi"}, headers=ALICE)
    assert resp.status_code == 504
    assert resp.json()["error"] == "upstream_timeout"
