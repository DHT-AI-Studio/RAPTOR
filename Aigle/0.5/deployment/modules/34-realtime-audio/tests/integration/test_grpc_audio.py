"""RA-10 / RA-6: AudioService.AudioChat end to end through the generated stub, a
grpc.aio.insecure_channel and test JWT metadata -- the same models and pipeline as the WS test."""
from __future__ import annotations

import asyncio
import time
from collections import Counter

import pytest

from .conftest import read_wav

grpc = pytest.importorskip("grpc")
audio_pb2 = pytest.importorskip("raptor.audio.v1.audio_pb2")
from raptor.audio.v1 import audio_pb2_grpc

pytestmark = pytest.mark.integration

FRAME_BYTES = 640            # 20 ms of 16 kHz s16le


async def microphone(data: bytes, first: dict):
    """AudioChunks at playback speed; the first one carries the stream settings."""
    start = time.monotonic()
    for i, off in enumerate(range(0, len(data), FRAME_BYTES)):
        yield audio_pb2.AudioChunk(payload=data[off:off + FRAME_BYTES], **(first if i == 0 else {}))
        delay = start + (i + 1) * 0.02 - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)


async def chat(live, requests, metadata, budget_s: float = 60.0):
    events: list[tuple[float, audio_pb2.AudioEvent]] = []
    async with grpc.aio.insecure_channel(live.grpc_target) as channel:
        call = audio_pb2_grpc.AudioServiceStub(channel).AudioChat(requests, metadata=metadata)

        async def read():
            async for event in call:
                events.append((time.monotonic(), event))

        try:
            await asyncio.wait_for(read(), timeout=budget_s)
        except grpc.aio.AioRpcError as exc:
            return events, exc.code(), exc.details()
        return events, await call.code(), await call.details()


async def test_audiochat_10s_wav_gives_final_transcript_and_spoken_answer(live):
    pcm, rate = read_wav("zh_tw_10s.wav")
    assert rate == 16000 and len(pcm) == 10 * 16000
    first = {"codec": "pcm", "sample_rate": 16000, "language": "zh-TW", "session_id": "ra10-grpc-10s"}
    metadata = (("authorization", f"Bearer {live.jwt()}"),)
    t0 = time.monotonic()
    events, code, details = await chat(live, microphone(pcm.tobytes(), first), metadata)
    assert code == grpc.StatusCode.OK, details

    errors = [e.error for _, e in events if e.WhichOneof("event") == "error"]
    assert not [e for e in errors if e.fatal], errors
    finals = [e.transcript for _, e in events if e.WhichOneof("event") == "transcript" and e.transcript.is_final]
    assert len(finals) >= 1 and all(f.text and f.language for f in finals), finals
    audio = [e.audio for _, e in events if e.WhichOneof("event") == "audio"]
    body = [a for a in audio if not a.is_last]
    assert body and all(a.payload and a.codec == "pcm" and a.sample_rate == 16000 for a in body)
    last = Counter(a.request_id for a in audio if a.is_last)
    assert last and set(last.values()) == {1}, last                      # one is_last per answer ...
    assert set(last) == {a.request_id for a in body}                    # ... closing every answer
    assert all(not a.payload for a in audio if a.is_last)
    assert len({e.session_id for _, e in events}) == 1
    t_final = next(t for t, e in events if e.WhichOneof("event") == "transcript" and e.transcript.is_final)
    t_last = next(t for t, e in events if e.WhichOneof("event") == "audio" and e.audio.is_last)
    print(f"\n[grpc] {len(finals)} final(s) {[f.text for f in finals]}, first at {t_final - t0:.2f} s, "
          f"first is_last at {t_last - t0:.2f} s (audio 10.00 s); {len(body)} chunks in {len(last)} answer(s)")
    assert await live.wait_idle() == 0


async def test_audiochat_without_valid_jwt_is_unauthenticated(live):
    silence = bytes(FRAME_BYTES * 10)
    for metadata in ((("authorization", f"Bearer {live.jwt(key=live.other_key)}"),),
                     (("authorization", f"Bearer {live.jwt(exp_in=-60)}"),),
                     ()):
        events, code, details = await chat(live, microphone(silence, {"codec": "pcm"}), metadata, budget_s=15)
        assert code == grpc.StatusCode.UNAUTHENTICATED and events == [], (metadata, code, details)
    assert await live.active_sessions() == 0
