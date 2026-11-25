"""VIE01-373 (INF-14) Tests 2, 5 -- the voice path, against a real module 35
(--profile voice) on a real GPU host. See README.md for the audio fixtures these need.
"""
from __future__ import annotations

import concurrent.futures

import httpx
import pytest

from .conftest import fixture_path


@pytest.fixture(autouse=True)
def _require_voice(voice_enabled: bool):
    if not voice_enabled:
        pytest.skip("this module 35 instance has no voice backend loaded "
                    "(VOICE_ENABLED=false, or no GPU -- see /health)")


def test_diarize_then_clone(client: httpx.Client, consented_user_id: str | None):
    """Test 2: diarize a 2-speaker clip -> 2 speakers identified; clone a cloned-voice clip
    from the first speaker's own words using a separate reference clip -> non-silent audio."""
    if consented_user_id is None:
        pytest.skip("E2E_GATEWAY_USER/PASSWORD not set (or gateway unreachable/login failed) -- cannot grant voice_clone "
                    "consent for this real user (see consented_user_id in conftest.py)")
    two_speakers = fixture_path("two_speakers_zh.wav")
    with two_speakers.open("rb") as f:
        r = client.post("/inference/voice/diarize", files={"audio": ("two_speakers_zh.wav", f, "audio/wav")})
    assert r.status_code == 200
    result = r.json()
    assert len(result["speakers"]) == 2
    assert result["segments"], "diarize returned no segments at all"
    first_segment_text = next((s.get("text", "") for s in result["segments"] if s.get("text", "").strip()), "")
    assert first_segment_text, "no segment carried a non-empty transcription to clone from"

    reference = fixture_path("reference_clip.wav")
    with reference.open("rb") as f:
        r = client.post("/inference/voice/clone",
                         data={"text": first_segment_text, "language": "zh-TW"},
                         files={"reference_audio": ("reference_clip.wav", f, "audio/wav")},
                         headers={"X-User-ID": consented_user_id})
    assert r.status_code == 200
    audio_bytes = r.content
    assert len(audio_bytes) > 1000, "clone returned suspiciously little audio data"
    assert "X-Speaker-Id" in r.headers


def test_speaker_cache_list_and_delete(client: httpx.Client, consented_user_id: str | None):
    """GET /clone/speakers and DELETE /clone/speakers/{id} -- not touched by
    test_diarize_then_clone, which only checks the X-Speaker-Id header is present."""
    if consented_user_id is None:
        pytest.skip("E2E_GATEWAY_USER/PASSWORD not set (or gateway unreachable/login failed) -- cannot grant voice_clone "
                    "consent for this real user (see consented_user_id in conftest.py)")
    user = {"X-User-ID": consented_user_id}
    reference = fixture_path("reference_clip.wav")
    with reference.open("rb") as f:
        r = client.post("/inference/voice/clone",
                         data={"text": "測試聲紋快取", "language": "zh-TW"},
                         files={"reference_audio": ("reference_clip.wav", f, "audio/wav")},
                         headers=user)
    assert r.status_code == 200
    speaker_id = r.headers["X-Speaker-Id"]

    r = client.get("/inference/voice/clone/speakers", headers=user)
    assert r.status_code == 200
    ids = [s["id"] for s in r.json()["speakers"]]
    assert speaker_id in ids

    r = client.delete(f"/inference/voice/clone/speakers/{speaker_id}", headers=user)
    assert r.status_code == 204

    r = client.get("/inference/voice/clone/speakers", headers=user)
    ids = [s["id"] for s in r.json()["speakers"]]
    assert speaker_id not in ids


def test_three_concurrent_voice_requests_do_not_deadlock(client: httpx.Client):
    """Test 5: 3 concurrent voice requests complete without deadlock or a CUDA OOM crash --
    using /diarize (already-existing endpoint) rather than a specific task the ticket didn't
    name."""
    clip = fixture_path("two_speakers_zh.wav")
    data = clip.read_bytes()

    def _diarize(i: int) -> int:
        r = client.post("/inference/voice/diarize",
                         files={"audio": (f"clip_{i}.wav", data, "audio/wav")})
        return r.status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        statuses = list(pool.map(_diarize, range(3)))
    assert statuses == [200, 200, 200]
