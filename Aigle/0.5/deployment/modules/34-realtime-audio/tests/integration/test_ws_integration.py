"""RA-10: WS /audio/stream end to end -- uvicorn's WebSocket stack, real silero-VAD, faster-whisper
and the TTS backend; Module 15 is benchmarks/fake_chat.py unless RT_IT_CHAT_URL is set."""
from __future__ import annotations

import asyncio
import json
import time

import httpx
import numpy as np
import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

from .conftest import (
    RESPONSE_TIMEOUT_S,
    TRANSCRIPT_TIMEOUT_S,
    Received,
    frames,
    of_type,
    paced,
    read_all,
    read_wav,
    read_wav_payload,
)

pytestmark = pytest.mark.integration

PCM_CONFIG = {"type": "CONFIG", "codec": "pcm", "sample_rate": 16000, "language": "zh-TW"}
FRAME_MS = 20


def has_cjk(text: str) -> bool:
    return any("一" <= c <= "鿿" for c in text)


async def wait_until(received: list[Received], kind: str, timeout_s: float = 15.0) -> Received:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        found = of_type(received, kind)
        if found:
            return found[0]
        await asyncio.sleep(0.02)
    raise AssertionError(f"no {kind} within {timeout_s} s; got {[r.event.get('type') for r in received if r.event]}")


async def session(live, config: dict, audio: list[bytes], frame_ms: float, *, end: bool = True,
                  budget_s: float = 60.0) -> tuple[list[Received], float, int | None]:
    """CONFIG, ``audio`` at playback speed, END; every server frame until the server closes.
    Returns (frames, time END was sent, close code)."""
    received: list[Received] = []
    async with connect(live.stream_url(await live.ticket()), max_size=None) as ws:
        reader = asyncio.create_task(read_all(ws, received))
        await ws.send(json.dumps(config))
        await wait_until(received, "READY")
        await paced(audio, frame_ms, ws.send)
        await ws.send(json.dumps({"type": "END"}))
        t_end = time.monotonic()
        code, _ = await asyncio.wait_for(reader, timeout=budget_s)
    return received, t_end, code


def first_response(received: list[Received]) -> tuple[Received, list[Received], Received]:
    """RESPONSE_START, its binary frames and its RESPONSE_END."""
    start = of_type(received, "RESPONSE_START")
    assert start, f"no RESPONSE_START; events: {[r.event.get('type') for r in received if r.event]}"
    rid = start[0].event["request_id"]
    i = received.index(start[0])
    ends = [r for r in received[i:] if r.event and r.event.get("type") == "RESPONSE_END"
            and r.event.get("request_id") == rid]
    assert ends, f"no RESPONSE_END for {rid}"
    j = received.index(ends[0])
    audio = [r for r in received[i:j] if r.data is not None]
    return start[0], audio, ends[0]


async def test_config_audio_end_gives_transcript_then_spoken_answer(live):
    """CONFIG -> 5 s zh-TW PCM -> END: TRANSCRIPT_FINAL within 3 s, RESPONSE_START + >= 1 binary TTS
    chunk + RESPONSE_END within 8 s (RT_IT_RESPONSE_TIMEOUT_S) of END."""
    pcm, rate = read_wav("zh_tw_5s.wav")
    assert rate == 16000 and len(pcm) == 5 * 16000
    received, t_end, code = await session(live, PCM_CONFIG, frames(pcm.tobytes(), 640), FRAME_MS)

    assert code == 1000
    assert not of_type(received, "ERROR"), of_type(received, "ERROR")[0].event
    # VAD: the speech was found and cut into an utterance
    assert of_type(received, "SPEECH_START")
    utterances = [r.event for r in of_type(received, "UTTERANCE")]
    assert utterances and sum(u["speech_ms"] for u in utterances) >= 2500, utterances
    # STT
    finals = of_type(received, "TRANSCRIPT_FINAL")
    assert finals and all(has_cjk(f.event["text"]) for f in finals), [f.event for f in finals]
    final_after_end = max(f.t for f in finals) - t_end
    assert final_after_end <= TRANSCRIPT_TIMEOUT_S, f"TRANSCRIPT_FINAL {final_after_end:.2f} s after END"
    # the spoken answer
    start, audio, end = first_response(received)
    assert start.event["output"]["codec"] == "pcm" and start.event["output"]["sample_rate"] == 16000
    assert audio and all(len(a.data) > 0 and len(a.data) % 2 == 0 for a in audio)
    assert not end.event.get("interrupted"), end.event
    end_after_end = end.t - t_end
    assert end_after_end <= RESPONSE_TIMEOUT_S, f"RESPONSE_END {end_after_end:.2f} s after END"
    ended = of_type(received, "ENDED")
    assert ended and ended[0].event["responses"] >= 1
    print(f"\n[ws pcm] final {final_after_end:+.2f} s, RESPONSE_START {start.t - t_end:+.2f} s, "
          f"RESPONSE_END {end_after_end:+.2f} s after END; {len(audio)} audio frames; "
          f"text={[f.event['text'] for f in finals]}")
    assert await live.wait_idle() == 0


async def test_end_without_audio_closes_gracefully(live):
    received, _, code = await session(live, PCM_CONFIG, [], FRAME_MS, budget_s=15)
    assert code == 1000
    kinds = [r.event["type"] for r in received if r.event]
    assert "READY" in kinds and kinds[-1] == "ENDED", kinds
    assert not {"ERROR", "WARNING", "UTTERANCE", "TRANSCRIPT_FINAL", "RESPONSE_START"} & set(kinds), kinds
    assert of_type(received, "ENDED")[0].event["utterances"] == 0
    assert await live.wait_idle() == 0


async def test_silence_makes_no_utterance(live):
    """VAD path: low noise only -- no SPEECH_START, no transcript, no answer."""
    rng = np.random.default_rng(34)
    noise = (rng.standard_normal(3 * 16000) * 30).astype("<i2")
    received, _, code = await session(live, PCM_CONFIG, frames(noise.tobytes(), 640), FRAME_MS, budget_s=20)
    assert code == 1000
    kinds = {r.event["type"] for r in received if r.event}
    assert not {"SPEECH_START", "UTTERANCE", "TRANSCRIPT_FINAL", "RESPONSE_START", "ERROR"} & kinds, kinds
    assert await live.wait_idle() == 0


async def test_pcmu_call_gets_transcript_and_pcmu_answer(live):
    """The telephony path end to end: G.711 mu-law in, 20 ms G.711 frames back."""
    data, fmt, rate = read_wav_payload("zh_tw_5s_pcmu.wav")
    assert (fmt, rate) == (7, 8000)
    config = {"type": "CONFIG", "codec": "pcmu", "sample_rate": 8000, "language": "zh-TW"}
    received, _, code = await session(live, config, frames(data, 160), FRAME_MS)
    assert code == 1000
    assert not of_type(received, "ERROR")
    finals = of_type(received, "TRANSCRIPT_FINAL")
    assert finals and all(has_cjk(f.event["text"]) for f in finals)
    start, audio, _ = first_response(received)
    assert (start.event["output"]["codec"], start.event["output"]["sample_rate"]) == ("pcmu", 8000)
    assert audio and {len(a.data) for a in audio} == {160}
    print(f"\n[ws pcmu] text={[f.event['text'] for f in finals]}; {len(audio)} G.711 frames")
    assert await live.wait_idle() == 0


async def _upgrade_status(url: str, headers: dict[str, str] | None = None) -> int:
    with pytest.raises(InvalidStatus) as exc:
        async with connect(url, additional_headers=headers):
            pass
    return exc.value.response.status_code


async def test_invalid_jwt_on_upgrade_is_401_before_the_websocket_opens(live):
    """The upgrade takes a one-time ticket; a JWT -- forged, expired or even valid -- in its place,
    or no credential at all, is refused with HTTP 401 and never becomes a WebSocket."""
    forged = live.jwt(key=live.other_key)
    assert await _upgrade_status(live.stream_url(forged)) == 401
    assert await _upgrade_status(live.stream_url(live.jwt(exp_in=-60))) == 401
    assert await _upgrade_status(live.stream_url(live.jwt())) == 401
    assert await _upgrade_status(live.stream_url(None)) == 401
    assert await _upgrade_status(live.stream_url(None), {"Authorization": f"Bearer {forged}"}) == 401
    assert await live.active_sessions() == 0


async def test_ticket_endpoint_rejects_invalid_jwt(live):
    async with httpx.AsyncClient(base_url=live.http_url, timeout=10) as client:
        for headers in ({"Authorization": f"Bearer {live.jwt(key=live.other_key)}"},
                        {"Authorization": f"Bearer {live.jwt(exp_in=-60)}"},
                        {"Authorization": f"Bearer {live.jwt(sub=None)}"},
                        {}):
            resp = await client.post("/audio/session/token", headers=headers)
            assert resp.status_code == 401, (headers, resp.text)
    assert await live.active_sessions() == 0
