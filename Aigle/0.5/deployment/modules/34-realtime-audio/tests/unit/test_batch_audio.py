"""INF-10/11 (VIE01-316/317): POST /audio/transcribe, POST /audio/synthesize -- the one-shot
endpoints Module 35 forwards to instead of loading a second copy of these models."""
from __future__ import annotations

import io

import soundfile as sf

from .conftest import FakeTts, FakeWhisper, make_settings, speech

SR = 16000
TOKEN = {"X-Internal-Token": "test-internal-token", "X-User-Id": "user-1"}
TOKEN_ONLY = {"X-Internal-Token": "test-internal-token"}


def _wav(samples) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, samples, SR, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


# ---------------------------------------------------------------- /audio/transcribe

def test_transcribe_returns_the_backend_text(make_client):
    stt = FakeWhisper(words=[("你", 0, 150), ("好", 150, 300)])
    client = make_client(stt=stt)
    resp = client.post("/audio/transcribe", files={"audio": ("a.wav", _wav(speech(300)), "audio/wav")},
                       headers=TOKEN)
    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == "你好"
    assert body["language"] == "zh"
    # The final decode path (beam_size > 1), not the rolling/provisional one.
    assert stt.calls[-1]["beam_size"] > 1


def test_transcribe_without_internal_token_is_401(make_client):
    client = make_client(stt=FakeWhisper())
    resp = client.post("/audio/transcribe", files={"audio": ("a.wav", _wav(speech(300)), "audio/wav")})
    assert resp.status_code == 401


def test_transcribe_wrong_internal_token_is_401(make_client):
    client = make_client(stt=FakeWhisper())
    resp = client.post("/audio/transcribe", files={"audio": ("a.wav", _wav(speech(300)), "audio/wav")},
                       headers={"X-Internal-Token": "wrong"})
    assert resp.status_code == 401


def test_transcribe_without_stt_configured_is_503(make_client):
    client = make_client()                             # no stt= -- app.state.stt is None
    resp = client.post("/audio/transcribe", files={"audio": ("a.wav", _wav(speech(300)), "audio/wav")},
                       headers=TOKEN)
    assert resp.status_code == 503


def test_transcribe_rejects_a_non_16khz_clip(make_client):
    client = make_client(stt=FakeWhisper())
    buffer = io.BytesIO()
    sf.write(buffer, speech(300), 8000, format="WAV", subtype="PCM_16")
    resp = client.post("/audio/transcribe", files={"audio": ("a.wav", buffer.getvalue(), "audio/wav")},
                       headers=TOKEN)
    assert resp.status_code == 400


def test_transcribe_rejects_invalid_audio_bytes(make_client):
    client = make_client(stt=FakeWhisper())
    resp = client.post("/audio/transcribe", files={"audio": ("a.wav", b"not a wav file", "audio/wav")},
                       headers=TOKEN)
    assert resp.status_code == 400


def test_transcribe_converts_to_traditional_when_requested(make_client):
    stt = FakeWhisper(words=[("头", 0, 150), ("发", 150, 300)], language="zh")
    client = make_client(stt=stt)
    resp = client.post("/audio/transcribe", files={"audio": ("a.wav", _wav(speech(300)), "audio/wav")},
                       data={"language": "zh-TW"}, headers=TOKEN)
    assert resp.status_code == 200
    assert resp.json()["text"] == "頭髮"


def test_transcribe_without_user_id_header_is_422(make_client):
    client = make_client(stt=FakeWhisper())
    resp = client.post("/audio/transcribe", files={"audio": ("a.wav", _wav(speech(300)), "audio/wav")},
                       headers=TOKEN_ONLY)
    assert resp.status_code == 422


def test_transcribe_denies_when_per_user_quota_exceeded(make_client, sync_redis):
    client = make_client(settings=make_settings(max_sessions_per_user=0), stt=FakeWhisper())
    resp = client.post("/audio/transcribe", files={"audio": ("a.wav", _wav(speech(300)), "audio/wav")},
                       headers=TOKEN)
    assert resp.status_code == 429
    assert resp.json()["detail"] == "too_many_sessions_for_user"


def test_transcribe_releases_its_slot_so_a_later_call_still_fits(make_client, sync_redis):
    client = make_client(settings=make_settings(max_sessions_per_user=1), stt=FakeWhisper())
    for _ in range(2):
        resp = client.post("/audio/transcribe", files={"audio": ("a.wav", _wav(speech(300)), "audio/wav")},
                           headers=TOKEN)
        assert resp.status_code == 200
    assert sync_redis.zcard("rt:sessions:user:user-1") == 0


# ---------------------------------------------------------------- /audio/synthesize

def test_synthesize_returns_wav_audio(make_client):
    tts = FakeTts()
    client = make_client(tts=tts)
    resp = client.post("/audio/synthesize", data={"text": "你好", "language": "zh-TW"}, headers=TOKEN)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    data, sr = sf.read(io.BytesIO(resp.content))
    assert sr == tts.sample_rate
    assert len(data) > 0
    assert tts.calls[-1][0] == "你好"


def test_synthesize_without_internal_token_is_401(make_client):
    client = make_client(tts=FakeTts())
    resp = client.post("/audio/synthesize", data={"text": "hi"})
    assert resp.status_code == 401


def test_synthesize_without_tts_configured_is_503(make_client):
    client = make_client()                             # no tts= -- app.state.tts is None
    resp = client.post("/audio/synthesize", data={"text": "hi"}, headers=TOKEN)
    assert resp.status_code == 503


def test_synthesize_backend_failure_is_502(make_client):
    client = make_client(tts=FakeTts(fail=True))
    resp = client.post("/audio/synthesize", data={"text": "hi"}, headers=TOKEN)
    assert resp.status_code == 502


def test_synthesize_denies_when_per_user_quota_exceeded(make_client):
    client = make_client(settings=make_settings(max_sessions_per_user=0), tts=FakeTts())
    resp = client.post("/audio/synthesize", data={"text": "hi"}, headers=TOKEN)
    assert resp.status_code == 429
    assert resp.json()["detail"] == "too_many_sessions_for_user"


def test_synthesize_releases_its_slot_even_on_backend_failure(make_client, sync_redis):
    client = make_client(settings=make_settings(max_sessions_per_user=1), tts=FakeTts(fail=True))
    resp = client.post("/audio/synthesize", data={"text": "hi"}, headers=TOKEN)
    assert resp.status_code == 502
    assert sync_redis.zcard("rt:sessions:user:user-1") == 0
