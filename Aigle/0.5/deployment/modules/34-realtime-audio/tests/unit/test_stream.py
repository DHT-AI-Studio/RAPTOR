"""RA-1 / RA-2: WS /audio/stream -- pre-upgrade checks, CONFIG, session records, VAD events."""
from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pytest
from app.session import ACTIVE_KEY
from starlette.testclient import WebSocketDenialResponse

from .conftest import (
    PCM_CONFIG,
    AmplitudeVad,
    issue_ticket,
    make_settings,
    silence,
    speech,
)


def collect(ws) -> tuple[list[dict[str, Any]], int]:
    """Read server events until the server closes; returns (events, close code)."""
    events: list[dict[str, Any]] = []
    while True:
        msg = ws.receive()
        if msg["type"] == "websocket.close":
            return events, msg.get("code", 1000)
        if msg.get("text") is not None:
            events.append(json.loads(msg["text"]))


def types(events) -> list[str]:
    return [e["type"] for e in events]


def send_pcm(ws, pcm: np.ndarray, frame_ms: int = 20) -> None:
    step = 16 * frame_ms
    for off in range(0, len(pcm), step):
        ws.send_bytes(pcm[off:off + step].astype("<i2").tobytes())


def hgetall(sync_redis, session_id: str) -> dict[str, str]:
    return {k.decode(): v.decode() for k, v in sync_redis.hgetall(f"rt:session:{session_id}").items()}


def wait_for_slot_release(sync_redis, timeout_s: float = 2.0) -> None:
    """handle()'s `finally: await asyncio.shield(release_slot(...))` runs after the
    close frame reaches the client (and inside its own shielded task on top of that),
    so a synchronous TestClient reading that frame is not proof the server's side has
    actually finished running it yet. Poll instead of asserting immediately after
    collect() returns -- same race test_disconnect_without_end_writes_final_state
    below already waits out."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not sync_redis.zcard(ACTIVE_KEY):
            return
        time.sleep(0.02)


# ---------------------------------------------------------------- pre-upgrade checks

def _denied(client, url: str) -> WebSocketDenialResponse:
    with pytest.raises(WebSocketDenialResponse) as exc, client.websocket_connect(url):
        pass
    return exc.value


def test_missing_or_unknown_ticket_is_401(client):
    for url in ("/audio/stream", "/audio/stream?token=", "/audio/stream?token=nope"):
        denied = _denied(client, url)
        assert denied.status_code == 401
        assert denied.json()["error"] == "invalid_ticket"


def test_ticket_is_single_use(client):
    token = issue_ticket(client)
    with client.websocket_connect(f"/audio/stream?token={token}") as ws:
        ws.send_json(PCM_CONFIG)
        assert ws.receive_json()["type"] == "STATE"
        ws.send_json({"type": "END"})
        collect(ws)
    assert _denied(client, f"/audio/stream?token={token}").status_code == 401


def test_expired_ticket_is_401(client, sync_redis):
    token = issue_ticket(client)
    sync_redis.delete(f"rt:ws_ticket:{token}")          # what the 30s TTL does
    assert _denied(client, f"/audio/stream?token={token}").status_code == 401


def test_max_sessions_is_503_without_burning_the_ticket(make_client, sync_redis):
    client = make_client(make_settings(max_sessions=1))
    first = issue_ticket(client)
    second = issue_ticket(client)
    with client.websocket_connect(f"/audio/stream?token={first}") as ws:
        ws.send_json(PCM_CONFIG)
        ws.receive_json()
        denied = _denied(client, f"/audio/stream?token={second}")
        assert denied.status_code == 503
        assert denied.json() == {"error": "too_many_sessions", "max_sessions": 1}
        ws.send_json({"type": "END"})
        collect(ws)
    wait_for_slot_release(sync_redis)
    assert sync_redis.zcard(ACTIVE_KEY) == 0
    with client.websocket_connect(f"/audio/stream?token={second}") as ws:   # ticket still valid
        ws.send_json({"type": "END"})


def test_stale_slot_from_crashed_instance_is_reclaimed(make_client, sync_redis):
    client = make_client(make_settings(max_sessions=1))
    sync_redis.zadd(ACTIVE_KEY, {"crashed-session": time.time() - 3600})
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client)}") as ws:
        ws.send_json({"type": "END"})
    assert sync_redis.zscore(ACTIVE_KEY, "crashed-session") is None


def test_max_sessions_per_user_is_503_without_burning_the_ticket_or_blocking_others(make_client, sync_redis):
    # VIE01-365 AC: per-user quota, distinct from the service-wide RT_MAX_SESSIONS. max_sessions
    # is left generous here so the per-user cap is the one actually binding.
    client = make_client(make_settings(max_sessions_per_user=1))
    first = issue_ticket(client, "alice")
    second = issue_ticket(client, "alice")
    third = issue_ticket(client, "bob")                   # a different user: unaffected by alice's cap
    with client.websocket_connect(f"/audio/stream?token={first}") as ws:
        ws.send_json(PCM_CONFIG)
        ws.receive_json()
        denied = _denied(client, f"/audio/stream?token={second}")
        assert denied.status_code == 503
        assert denied.json() == {"error": "too_many_sessions_for_user", "max_sessions_per_user": 1}
        with client.websocket_connect(f"/audio/stream?token={third}") as other:
            other.send_json({"type": "END"})
            collect(other)
        ws.send_json({"type": "END"})
        collect(ws)
    wait_for_slot_release(sync_redis)
    assert sync_redis.zcard(ACTIVE_KEY) == 0
    assert sync_redis.zcard("rt:sessions:user:alice") == 0
    with client.websocket_connect(f"/audio/stream?token={second}") as ws:   # ticket still valid
        ws.send_json({"type": "END"})


# ---------------------------------------------------------------- CONFIG

@pytest.mark.parametrize("frame, code", [
    ({"codec": "mp3", "sample_rate": 16000, "language": "zh-TW"}, "UNSUPPORTED_CODEC"),
    ({"codec": "pcm", "sample_rate": 96000, "language": "zh-TW"}, "invalid_config"),
    ({"codec": "pcm", "sample_rate": "16000", "language": "zh-TW"}, "invalid_config"),
    ({"codec": "pcm", "sample_rate": 16000, "language": "Chinese"}, "invalid_config"),
    ({"codec": "pcm", "sample_rate": 16000, "language": ""}, "invalid_config"),   # omitted = RT_STT_LANGUAGE
    ({"type": "END"}, "config_required"),
    ("not json", "invalid_config"),
    (b"\x00\x01", "config_required"),
])
def test_invalid_config_closes_4400(client, sync_redis, frame, code):
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client)}") as ws:
        if isinstance(frame, bytes):
            ws.send_bytes(frame)
        elif isinstance(frame, str):
            ws.send_text(frame)
        else:
            ws.send_json(frame)
        events, close = collect(ws)
    assert close == 4400
    assert events[-1]["type"] == "ERROR" and events[-1]["code"] == code
    wait_for_slot_release(sync_redis)
    assert sync_redis.zcard(ACTIVE_KEY) == 0


def test_config_timeout_closes_4408(make_client):
    client = make_client(make_settings(config_timeout_s=0.2))
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client)}") as ws:
        events, close = collect(ws)
    assert close == 4408 and events[-1]["code"] == "config_timeout"


def test_ready_and_session_record(client, sync_redis):
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client, 'bob')}") as ws:
        ws.send_json({"codec": "pcm", "sample_rate": 16000, "language": "en"})   # "type" optional
        state = ws.receive_json()
        ready = ws.receive_json()
        sid = ready["session_id"]
        assert state == {"type": "STATE", "state": "LISTENING", "seq": 1, "session_id": sid}
        assert ready["type"] == "READY" and ready["state"] == "LISTENING" and ready["seq"] == 2
        assert ready["config"] == {"codec": "pcm", "sample_rate": 16000, "language": "en", "framing": "raw",
                                   "brain": "module15"}
        record = hgetall(sync_redis, sid)
        assert record["user_id"] == "bob" and record["state"] == "LISTENING"
        assert (record["codec"], record["sample_rate"], record["language"]) == ("pcm", "16000", "en")
        assert float(record["started_at"]) > 0
        assert 1790 <= sync_redis.ttl(f"rt:session:{sid}") <= 1800
        assert sync_redis.zscore(ACTIVE_KEY, sid) is not None
        ws.send_json({"type": "END"})
        collect(ws)


# ---------------------------------------------------------------- audio + VAD

def test_utterance_boundary_and_end(client, sync_redis):
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client)}") as ws:
        ws.send_json(PCM_CONFIG)
        ws.receive_json(), ws.receive_json()
        send_pcm(ws, np.concatenate([silence(500), speech(1000), silence(500)]))
        ws.send_text("END")
        events, close = collect(ws)
    assert close == 1000
    assert types(events) == ["SPEECH_START", "UTTERANCE", "STATE", "STATE", "ENDED"]
    utt = events[1]
    assert utt["reason"] == "silence" and utt["index"] == 1
    # speech starts at 500 ms; utterance keeps RT_VAD_SPEECH_PAD_MS (96) of pre-roll, 32 ms windows
    assert 500 - 96 - 32 <= utt["start_ms"] <= 500 - 96 and 1000 <= utt["speech_ms"] <= 1040
    assert [e["state"] for e in events[2:4]] == ["PROCESSING", "LISTENING"]
    ended = events[-1]
    assert ended["utterances"] == 1 and ended["audio_ms"] == 2000 and ended["dropped_chunks"] == 0

    record = hgetall(sync_redis, ended["session_id"])
    assert record["state"] == "ENDED" and record["end_reason"] == "end"
    assert record["utterances"] == "1" and record["audio_ms"] == "2000"
    assert sync_redis.ttl(f"rt:session:{ended['session_id']}") > 0       # kept until its TTL
    wait_for_slot_release(sync_redis)
    assert sync_redis.zcard(ACTIVE_KEY) == 0


def test_end_flushes_pending_utterance(client):
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client)}") as ws:
        ws.send_json(PCM_CONFIG)
        ws.receive_json(), ws.receive_json()
        send_pcm(ws, np.concatenate([silence(200), speech(600)]))
        ws.send_json({"type": "END"})
        events, _ = collect(ws)
    utt = [e for e in events if e["type"] == "UTTERANCE"]
    assert len(utt) == 1 and utt[0]["reason"] == "end"
    assert events[-1]["utterances"] == 1


def test_short_noise_is_discarded(client):
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client)}") as ws:
        ws.send_json(PCM_CONFIG)
        ws.receive_json(), ws.receive_json()
        send_pcm(ws, np.concatenate([silence(300), speech(100), silence(500)]))
        ws.send_json({"type": "END"})
        events, _ = collect(ws)
    assert "UTTERANCE" not in types(events)
    discarded = [e for e in events if e["type"] == "DISCARDED"]
    assert discarded and discarded[0]["reason"] == "too_short"


def test_disconnect_without_end_writes_final_state(client, sync_redis):
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client)}") as ws:
        ws.send_json(PCM_CONFIG)
        ws.receive_json()
        sid = ws.receive_json()["session_id"]
        send_pcm(ws, speech(300))
        ws.close()
        for _ in range(100):      # let the server handle the disconnect before the client tears down
            if hgetall(sync_redis, sid).get("state") == "ENDED" and not sync_redis.zcard(ACTIVE_KEY):
                break
            time.sleep(0.02)
    record = hgetall(sync_redis, sid)
    assert record["state"] == "ENDED" and record["end_reason"] == "disconnect"
    assert sync_redis.zcard(ACTIVE_KEY) == 0


def test_bad_frames_warn_without_closing(client):
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client)}") as ws:
        ws.send_json(PCM_CONFIG)
        ws.receive_json(), ws.receive_json()
        ws.send_json({"type": "HELLO"})
        warning = ws.receive_json()
        assert warning["type"] == "WARNING" and warning["code"] == "unexpected_frame"
        ws.send_json(PCM_CONFIG)          # a second CONFIG is ignored
        assert ws.receive_json()["code"] == "unexpected_frame"
        ws.send_json({"type": "END"})
        events, close = collect(ws)
    assert close == 1000 and events[-1]["type"] == "ENDED"


def test_pcm_odd_byte_frames(client):
    raw = np.concatenate([silence(200), speech(500), silence(400)]).astype("<i2").tobytes()
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client)}") as ws:
        ws.send_json(PCM_CONFIG)
        ws.receive_json(), ws.receive_json()
        for off in range(0, len(raw), 333):                  # odd-sized frames split samples
            ws.send_bytes(raw[off:off + 333])
        ws.send_json({"type": "END"})
        events, _ = collect(ws)
    assert types(events).count("UTTERANCE") == 1
    assert events[-1]["audio_ms"] == 1100


def test_opus_stream(client):
    av = pytest.importorskip("av")
    enc = av.CodecContext.create("libopus", "w")
    enc.sample_rate, enc.layout, enc.format = 48000, "mono", "s16"
    enc.bit_rate = 32000
    pcm16k = np.concatenate([silence(400), speech(800), silence(600)])
    pcm48k = np.repeat(pcm16k, 3)                          # crude upsample; fine for VAD
    packets = []
    for off in range(0, len(pcm48k) - 960 + 1, 960):       # 20 ms frames
        frame = av.AudioFrame.from_ndarray(pcm48k[off:off + 960].reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = 48000
        frame.pts = off
        packets += [bytes(p) for p in enc.encode(frame)]
    packets += [bytes(p) for p in enc.encode(None)]

    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client)}") as ws:
        ws.send_json({"type": "CONFIG", "codec": "opus", "sample_rate": 48000, "language": "en"})
        ws.receive_json(), ws.receive_json()
        for p in packets:
            ws.send_bytes(p)
        ws.send_bytes(b"\xff\xff\xff")                      # corrupt packet -> WARNING, stream continues
        ws.send_json({"type": "END"})
        events, close = collect(ws)
    assert close == 1000
    assert types(events).count("UTTERANCE") == 1
    assert 1500 <= events[-1]["audio_ms"] <= 1900


def test_session_lifetime_closes_4408(make_client, sync_redis):
    client = make_client(make_settings(session_ttl_s=1))
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client)}") as ws:
        ws.send_json(PCM_CONFIG)
        ws.receive_json()
        sid = ws.receive_json()["session_id"]
        events, close = collect(ws)
    assert close == 4408 and events[-1]["code"] == "session_expired"
    assert hgetall(sync_redis, sid).get("end_reason") in (None, "session_expired")   # key may have expired


def test_slow_vad_drops_instead_of_blocking(make_client):
    client = make_client(make_settings(audio_queue_max=2), vad_delay_s=0.02)
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client)}") as ws:
        ws.send_json(PCM_CONFIG)
        ws.receive_json(), ws.receive_json()
        send_pcm(ws, speech(3000), frame_ms=30)
        ws.send_json({"type": "END"})
        events, close = collect(ws)
    assert close == 1000
    assert any(e["type"] == "WARNING" and e["code"] == "QUEUE_OVERFLOW" for e in events)
    assert events[-1]["dropped_chunks"] > 0


def test_sessions_use_separate_vad_instances(make_client):
    client = make_client(make_settings(vad_pool_size=1))
    t1, t2 = issue_ticket(client, "a"), issue_ticket(client, "b")
    with client.websocket_connect(f"/audio/stream?token={t1}") as ws1, \
            client.websocket_connect(f"/audio/stream?token={t2}") as ws2:
        for ws in (ws1, ws2):
            ws.send_json(PCM_CONFIG)
            ws.receive_json(), ws.receive_json()
        send_pcm(ws1, np.concatenate([speech(500), silence(400)]))
        send_pcm(ws2, silence(900))
        for ws in (ws1, ws2):
            ws.send_json({"type": "END"})
        e1, _ = collect(ws1)
        e2, _ = collect(ws2)
    assert types(e1).count("UTTERANCE") == 1 and "UTTERANCE" not in types(e2)
    assert len(AmplitudeVad.instances) == 2


def test_sink_receives_wav_ready_audio(make_client):
    received = []

    class Sink:
        async def handle(self, session_id, config, utterance):
            received.append((session_id, config.language, utterance))

    client = make_client(sink=Sink())
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client)}") as ws:
        ws.send_json(PCM_CONFIG)
        ws.receive_json(), ws.receive_json()
        send_pcm(ws, np.concatenate([speech(700), silence(400)]))
        ws.send_json({"type": "END"})
        collect(ws)
    assert len(received) == 1
    _, language, utt = received[0]
    assert language == "zh-TW"
    wav = utt.wav()
    assert wav[:4] == b"RIFF" and len(wav) == 44 + 2 * len(utt.pcm)


def test_status_endpoint(client):
    body = client.get("/audio/status").json()
    assert body["active_sessions"] == 0 and body["max_sessions"] == 20
