"""RTA-X4 / RTA-X5 / RTA-X3: bounded queues and timeouts, barge-in, numbered frames, PING / PONG
and resuming a dropped session."""
from __future__ import annotations

import asyncio
import json
import struct
import time
from typing import Any

import httpx
import numpy as np
import pytest
from app.config import Settings
from app.llm_bridge import LlmBridge, LlmError, Prefetch
from app.outbox import Outbox, OutboxOverflow
from app.session import ACTIVE_KEY, InvalidTransition, SessionState
from app.tts import TtsService

from .conftest import (
    PCM_CONFIG,
    FakeChat,
    FakeTts,
    FakeWhisper,
    issue_ticket,
    make_settings,
    silence,
    speech,
)
from .test_stream import hgetall

LONG_ANSWER = "今天天氣很好，適合出去走走。|我們可以去公園散步，也可以去海邊看看。"   # 1.65 s of FakeTts audio


class Session:
    """A test WebSocket client that keeps every frame (JSON as dicts, binary as bytes) in order."""

    def __init__(self, ws, framing: str = "raw") -> None:
        self.ws = ws
        self.framing = framing
        self.frames: list[Any] = []
        self.close_code: int | None = None
        self.sent = 0                                        # client seq (framing "seq")

    def start(self, config: dict | None = None) -> dict:
        self.ws.send_json({**(config or PCM_CONFIG), "framing": self.framing})
        return self.until("READY")

    def recv(self) -> Any:
        msg = self.ws.receive()
        if msg["type"] == "websocket.close":
            self.close_code = msg.get("code", 1000)
            return None
        frame = msg["bytes"] if msg.get("bytes") is not None else json.loads(msg["text"])
        self.frames.append(frame)
        return frame

    def until(self, kind: str, **match: Any) -> dict:
        while True:
            frame = self.recv()
            if frame is None:
                raise AssertionError(f"closed ({self.close_code}) before {kind}; got {self.kinds()}")
            if isinstance(frame, dict) and frame["type"] == kind and all(frame.get(k) == v for k, v in match.items()):
                return frame

    def audio_frames(self, n: int) -> None:
        got = 0
        while got < n:
            if isinstance(self.recv(), bytes):
                got += 1

    def drain(self) -> int:
        while self.recv() is not None:
            pass
        return self.close_code

    def send_pcm(self, pcm: np.ndarray, frame_ms: int = 100) -> None:
        step = 16 * frame_ms
        for off in range(0, len(pcm), step):
            data = pcm[off:off + step].astype("<i2").tobytes()
            if self.framing == "seq":
                self.sent += 1
                data = struct.pack("<I", self.sent) + data
            self.ws.send_bytes(data)

    def kinds(self) -> list[str]:
        return ["AUDIO" if isinstance(f, bytes) else f["type"] for f in self.frames]

    def events(self, kind: str) -> list[dict]:
        return [f for f in self.frames if isinstance(f, dict) and f["type"] == kind]

    def last_seq(self) -> int:
        for f in reversed(self.frames):
            return struct.unpack_from("<I", f)[0] if isinstance(f, bytes) else f["seq"]
        return 0


def question() -> np.ndarray:
    return np.concatenate([speech(800), silence(500)])


def voice_client(make_client, chat: FakeChat | None = None, stt: FakeWhisper | None = None, **settings):
    chat = chat or FakeChat(answer=LONG_ANSWER)
    client = make_client(stt=stt or FakeWhisper(), tts=FakeTts(), chat=chat,
                         settings=make_settings(stt_streaming=False, **settings))
    return client, chat


# ---------------------------------------------------------------- RTA-X4 backpressure and timeouts


def test_settings_from_env(monkeypatch):
    for name, value in {"RT_UTTERANCE_QUEUE_MAX": "7", "RT_BARGE_IN_THRESHOLD": "0.85", "RT_TTS_QUEUE_MAX": "3",
                        "RT_LLM_FIRST_TOKEN_TIMEOUT_S": "4", "RT_SESSION_RESUME_WINDOW_S": "90",
                        "RT_PING_INTERVAL_S": "0", "RT_OUTPUT_PACE": "false"}.items():
        monkeypatch.setenv(name, value)
    s = Settings.from_env()
    assert (s.stt_queue_max, s.barge_in_threshold, s.tts_queue_max, s.llm_first_token_timeout_s) == (7, 0.85, 3, 4.0)
    assert (s.resume_window_s, s.ping_interval_s, s.output_pace) == (90.0, 0.0, False)
    monkeypatch.setenv("RT_STT_QUEUE_MAX", "9")                         # the new name wins
    assert Settings.from_env().stt_queue_max == 9
    d = Settings()
    assert (d.audio_queue_max, d.stt_queue_max, d.tts_queue_max, d.tts_max_concurrent) == (200, 10, 5, 2)
    assert (d.llm_first_token_timeout_s, d.barge_in_threshold, d.barge_in_min_ms, d.min_utterance_chars) == (30, 0.7, 300, 2)
    assert d.output_stall_s == 10
    assert (d.resume_window_s, d.ping_interval_s, d.pong_timeout_s) == (60, 30, 10)


def test_full_stt_queue_drops_the_utterance_and_keeps_the_session(make_client):
    client = make_client(stt=FakeWhisper(delay_s=0.4), settings=make_settings(stt_queue_max=1, stt_streaming=False))
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client)}") as ws:
        s = Session(ws)
        s.start()
        s.send_pcm(np.concatenate([np.concatenate([speech(300), silence(400)]) for _ in range(4)]))
        ws.send_json({"type": "END"})
        assert s.drain() == 1000
    errors = s.events("ERROR")
    assert errors and {e["code"] for e in errors} == {"STT_OVERLOADED"} and all(e["fatal"] is False for e in errors)
    finals = {e["utterance"] for e in s.events("TRANSCRIPT_FINAL")}
    assert finals.isdisjoint(e["utterance"] for e in errors) and len(finals) + len(errors) == 4
    assert s.kinds()[-1] == "ENDED"
    assert "rt_stt_overloaded_total " + f"{float(len(errors))}" in client.get("/metrics").text


def test_full_audio_queue_drops_the_oldest_audio(make_client):
    client = make_client(make_settings(audio_queue_max=40), vad_delay_s=0.02)
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client)}") as ws:
        s = Session(ws)
        s.start()
        # far faster than VAD can follow: only about the last 40 chunks (1.2 s) are kept
        s.send_pcm(np.concatenate([silence(3000), speech(600), silence(500)]), frame_ms=30)
        ws.send_json({"type": "END"})
        s.drain()
    warning = s.events("WARNING")[0]
    assert warning["code"] == "QUEUE_OVERFLOW" and warning["dropped_chunks"] > 0
    assert s.events("ENDED")[0]["dropped_chunks"] >= warning["dropped_chunks"]
    assert len(s.events("UTTERANCE")) == 1              # the newest audio (the speech) survived


def _bridge(transport, **kw) -> LlmBridge:
    return LlmBridge("http://chat", kw.pop("timeout_s", 5.0), True, httpx.AsyncClient(transport=httpx.MockTransport(transport)),
                     **kw)


async def _collect(it) -> list[str]:
    return [s async for s in it]


def test_no_answer_text_within_the_first_token_timeout():
    async def slow_first(request):
        async def body():
            yield b"event: start\ndata: {}\n\n"                       # alive, but no text
            await asyncio.sleep(1.0)
            yield b'event: delta\ndata: {"text": "\xe5\xa5\xbd\xe3\x80\x82"}\n\n'
            yield b"event: end\ndata: {}\n\n"

        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body())

    started = time.monotonic()
    with pytest.raises(LlmError) as exc:
        asyncio.run(_collect(_bridge(slow_first, first_token_timeout_s=0.3).sentences("q", "s", "u", "r")))
    assert exc.value.code == "LLM_TIMEOUT" and "answer text" in exc.value.message
    assert time.monotonic() - started < 0.9
    # a generous first-token limit lets the same answer through
    assert asyncio.run(_collect(_bridge(slow_first, first_token_timeout_s=3).sentences("q", "s", "u", "r"))) == ["好。"]


def test_first_token_timeout_does_not_limit_the_rest_of_the_answer():
    async def steady(request):
        async def body():
            for i in range(4):
                yield f'event: delta\ndata: {{"text": "第{i}句。"}}\n\n'.encode()
                await asyncio.sleep(0.15)
            yield b"event: end\ndata: {}\n\n"

        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body())

    out = asyncio.run(_collect(_bridge(steady, first_token_timeout_s=0.3).sentences("q", "s", "u", "r")))
    assert out == ["第0句。", "第1句。", "第2句。", "第3句。"]


def test_sentence_queue_is_bounded_and_a_full_queue_is_not_charged():
    read: list[int] = []

    async def source():
        for i in range(10):
            read.append(i)
            yield f"第{i}句。"

    async def consume():
        out = []
        async for s in Prefetch(source(), 2):
            out.append(s)
            await asyncio.sleep(0.05)
            assert len(read) <= len(out) + 3               # queue of 2, one in hand, one being put
        return out

    assert len(asyncio.run(consume())) == 10

    async def slow_but_steady(request):
        async def body():
            for i in range(6):
                await asyncio.sleep(0.05)
                yield f'event: delta\ndata: {{"text": "第{i}句。"}}\n\n'.encode()
            yield b"event: end\ndata: {}\n\n"

        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body())

    async def speak_slowly():
        out = []
        async for s in Prefetch(_bridge(slow_but_steady, timeout_s=0.6).sentences("q", "s", "u", "r"), 1):
            out.append(s)
            await asyncio.sleep(0.2)                       # 1.2 s of "speaking"; waiting on 15 is 0.3 s
        return out

    assert len(asyncio.run(speak_slowly())) == 6


def test_tts_slot_is_free_once_synthesis_is_done():
    async def scenario():
        svc = TtsService(Settings(tts_max_concurrent=1), loader=lambda _n, _s: FakeTts())
        await svc.start()
        stalled = svc.stream("第一句很長。", "zh-TW")
        await anext(stalled)                               # a paced session holds its stream open
        await asyncio.sleep(0.05)
        other = [p async for p in svc.stream("第二句。", "zh-TW")]
        await stalled.aclose()
        await asyncio.sleep(0.05)                          # worker done-callbacks
        return other, svc.busy, svc.waiting

    other, busy, waiting = asyncio.run(asyncio.wait_for(scenario(), timeout=2))
    assert len(other) == 2 and (busy, waiting) == (0, 0)


def test_metrics_endpoint(make_client):
    client = make_client(stt=FakeWhisper(), tts=FakeTts())
    text = client.get("/metrics").text
    for line in ('rt_sessions{connection="connected"} 0.0', "rt_stt_slots 1.0", "rt_tts_slots 2.0",
                 "rt_audio_queue_chunks 0.0", "rt_output_queue_ms 0.0", "rt_barge_ins_total 0.0",
                 "rt_resumes_total 0.0", "rt_llm_timeouts_total 0.0"):
        assert line in text, line


# ---------------------------------------------------------------- RTA-X5 barge-in


# ~6.9 s: the previous 3.4 s margin above the 2.8 s upper assertion bound (below) still wasn't
# enough -- a loaded CI runner hung this test for 27 minutes (BARGE_IN never fired because the
# whole answer had already played out by the time the barge audio registered as one). Tripling
# the added filler buys much more real-time slack before that can happen again.
BARGE_ANSWER = LONG_ANSWER + "|週末如果下雨，也可以在家看書聽音樂。|記得出門前帶把傘，晚上早點休息喔。" * 3


def test_barge_in_cancels_the_answer_and_the_speech_becomes_the_next_turn(make_client):
    client, chat = voice_client(make_client, chat=FakeChat(answer=BARGE_ANSWER), output_lead_ms=300)
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client)}") as ws:
        s = Session(ws)
        s.start()
        s.send_pcm(question())
        first = s.until("RESPONSE_START")
        time.sleep(1.0)                                    # one second into the answer ...
        s.send_pcm(np.concatenate([speech(600), silence(500)]))   # ... the user speaks over it
        barge = s.until("BARGE_IN")
        assert "RESPONSE_END" not in s.kinds(), f"answer fully sent before the barge-in at {barge['tts_position_ms']} ms"
        end = s.until("RESPONSE_END")
        s.until("TRANSCRIPT_FINAL", utterance=2)
        second = s.until("RESPONSE_START")
        ws.send_json({"type": "END"})
        assert s.drain() == 1000
    assert barge["request_id"] == first["request_id"] == end["request_id"] and end["interrupted"] is True
    assert 900 <= barge["tts_position_ms"] <= 2800 and end["duration_ms"] == barge["tts_position_ms"]
    kinds = s.kinds()
    i_start, i_barge = s.frames.index(first), s.frames.index(barge)
    sent_ms = sum(len(f) for f in s.frames[i_start:i_barge] if isinstance(f, bytes)) / 32
    # position_ms() estimates playback from real wall-clock elapsed since origin; if the
    # server's event loop stalls (a loaded CI runner can genuinely starve it for a while), the
    # estimate jumps ahead once it resumes and legitimately releases a backlog burst past
    # RT_OUTPUT_LEAD_MS to catch up -- not a pacing bug. 1500 ms of slack tolerates that while
    # still catching an actual regression (e.g. pacing disabled outright).
    assert sent_ms <= barge["tts_position_ms"] + 300 + 1500
    assert "AUDIO" not in kinds[i_barge:s.frames.index(second)]       # the rest was never sent
    assert [e["state"] for e in s.events("STATE")][-1] == "LISTENING"
    assert len(chat.requests) == 2 and s.events("ENDED")[0]["barge_ins"] == 1


def test_speech_during_the_answer_without_barge_in_is_discarded(make_client):
    client, chat = voice_client(make_client, barge_in_min_ms=10_000)
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client)}") as ws:
        s = Session(ws)
        s.start()
        s.send_pcm(question())
        s.until("RESPONSE_START")
        s.send_pcm(np.concatenate([speech(500), silence(400)]))    # echo of the answer
        discarded = s.until("DISCARDED")
        ws.send_json({"type": "END"})
        s.drain()
    assert discarded["reason"] == "during_response"
    assert len(chat.requests) == 1 and not s.events("BARGE_IN")
    assert [e["index"] for e in s.events("UTTERANCE")] == [1]


def test_speech_that_began_before_the_answer_is_kept(make_client):
    client, chat = voice_client(make_client, barge_in_min_ms=10_000)
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client)}") as ws:
        s = Session(ws)
        s.start()
        # first sentence, a pause the VAD ends it on, and the start of a second sentence ...
        s.send_pcm(np.concatenate([speech(800), silence(500), speech(300)]))
        s.until("RESPONSE_START")                          # ... the answer to the first one starts
        s.send_pcm(np.concatenate([speech(300), silence(500)]))   # the second sentence ends over it
        s.until("TRANSCRIPT_FINAL", utterance=2)
        ws.send_json({"type": "END"})
        s.drain()
    assert not s.events("DISCARDED") and [e["index"] for e in s.events("UTTERANCE")] == [1, 2]
    assert len(chat.requests) == 2


def test_leaving_after_end_does_not_leave_the_session_waiting(make_client, sync_redis):
    client, _ = voice_client(make_client, chat=FakeChat(answer=LONG_ANSWER, delay_s=0.5), output_queue_ms=500)
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client)}") as ws:
        s = Session(ws)
        sid = s.start()["session_id"]
        s.send_pcm(question())
        ws.send_json({"type": "END"})
        ws.close()                                         # gone before the answer has played
        for _ in range(200):
            if hgetall(sync_redis, sid).get("state") == "ENDED" and not sync_redis.zcard(ACTIVE_KEY):
                break
            time.sleep(0.02)
    assert hgetall(sync_redis, sid)["end_reason"] == "disconnect_after_end"
    assert sync_redis.zcard(ACTIVE_KEY) == 0


def test_a_one_letter_transcript_starts_no_turn(make_client):
    client, chat = voice_client(make_client, stt=FakeWhisper(words=[("嗯", 0, 150)]))
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client)}") as ws:
        s = Session(ws)
        s.start()
        s.send_pcm(question())
        ws.send_json({"type": "END"})
        s.drain()
    assert s.events("TRANSCRIPT_FINAL")[0]["text"] == "嗯" and chat.requests == []


def test_outbox_paces_audio_and_drops_it_on_interrupt():
    sent: list[tuple[float, Any]] = []

    class Recorder:
        async def send_text(self, text):
            sent.append((time.monotonic(), json.loads(text)))

        async def send_bytes(self, data):
            sent.append((time.monotonic(), data))

    async def scenario():
        box = Outbox("s1", lead_ms=200, queue_ms=5000)
        box.attach(Recorder())
        sender = asyncio.create_task(box.run())

        async def produce():
            await box.response_event({"type": "RESPONSE_START"}, "r1")
            for _ in range(100):                            # 2 s of 20 ms frames
                await box.audio(b"\0" * 160, 20, "r1")

        producer = asyncio.create_task(produce())
        await asyncio.sleep(0.5)
        audio_ms = sum(20 for _, f in sent if isinstance(f, bytes))
        position = box.position_ms()
        rid, pos = box.answer_position()
        await box.interrupt()
        producer.cancel()
        n = len(sent)
        await asyncio.sleep(0.2)
        sender.cancel()
        return audio_ms, position, (rid, pos), box.playing, len(sent) - n

    audio_ms, position, (rid, pos), playing, after = asyncio.run(scenario())
    assert 450 <= position <= 750 and rid == "r1" and abs(pos - position) <= 5
    assert position + 140 <= audio_ms <= position + 300                # ~RT_OUTPUT_LEAD_MS ahead
    assert not playing and after == 0


def test_outbox_bounds_both_event_lanes_and_only_a_connected_stall_overflows():
    class Unread:
        async def send_text(self, text):
            pass

        async def send_bytes(self, data):
            pass

    async def scenario():
        box = Outbox("s", lead_ms=0, queue_ms=20, max_frames=2, wait_timeout_s=0.05)
        for _ in range(2):
            box.event({"type": "PING"})
            await box.response_event({"type": "RESPONSE_TEXT"}, "r")
        with pytest.raises(OutboxOverflow, match="control"):
            box.event({"type": "PING"})
        box.event({"type": "ERROR"}, priority=True)
        assert box.queued == 4 and box._control[0].data["type"] == "ERROR"
        await box.interrupt()
        await box.audio(b"audio", 20, "r")
        waiting = asyncio.create_task(box.audio(b"audio", 20, "r"))
        await asyncio.sleep(0.3)
        assert not waiting.done()                          # suspended: waits for RESUME
        box.attach(Unread())                               # connected, but nothing is sent
        with pytest.raises(OutboxOverflow, match="drain"):
            await waiting
        assert box.unsent_ms == 20

    asyncio.run(scenario())


def test_output_space_notification_does_not_swallow_producer_cancellation():
    async def scenario():
        box = Outbox("s", lead_ms=0, queue_ms=20)
        await box.audio(b"old", 20, "old")
        producer = asyncio.create_task(box.audio(b"cancelled", 20, "cancelled"))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        await box.interrupt()
        await asyncio.sleep(0)
        producer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await producer
        assert box.queued == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("target", ["PROCESSING", "RESPONDING"])
def test_invalid_pipeline_transition_closes_connection(make_client, monkeypatch, target):
    original = SessionState.to

    async def invalid(self, new_state, **fields):
        if new_state == target:
            raise InvalidTransition("injected state error")
        await original(self, new_state, **fields)

    monkeypatch.setattr(SessionState, "to", invalid)
    client, _ = voice_client(make_client)
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client)}") as ws:
        s = Session(ws)
        s.start()
        s.send_pcm(question())
        assert s.drain() == 1011
    assert s.events("ERROR")[-1]["code"] == "internal_error"


def test_outbox_numbers_frames_and_keeps_them_for_replay():
    class Recorder:
        def __init__(self):
            self.out: list = []

        async def send_text(self, text):
            self.out.append(json.loads(text))

        async def send_bytes(self, data):
            self.out.append(data)

    async def scenario():
        box = Outbox("s1", pace=False, retain_bytes=420, seq_prefix=True)
        first = Recorder()
        box.attach(first)
        sender = asyncio.create_task(box.run())
        for i in range(5):
            box.event({"type": "E", "i": i})               # 44-45 bytes
            await box.flush()
            await box.audio(bytes(100), 12.5, "r")         # 104 bytes with the seq
            await box.flush()
        box.detach()
        second = Recorder()
        replay = await box.resume(second, 6, lambda r: {"type": "RESUMED", "replayed": r.frames, "gap": r.gap})
        await box.flush()
        third = Recorder()
        gap = await box.resume(third, 0, lambda r: {"type": "RESUMED", "gap": r.gap})
        await box.flush()
        sender.cancel()
        return first.out, second.out, replay, gap

    first, second, replay, gap = asyncio.run(scenario())
    seqs = [f["seq"] if isinstance(f, dict) else struct.unpack_from("<I", f)[0] for f in first]
    assert seqs == list(range(1, 11)) and all(f["session_id"] == "s1" for f in first if isinstance(f, dict))
    assert replay.frames == 4 and not replay.gap                     # 6..10 kept (400 of 420 bytes)
    assert [struct.unpack_from("<I", f)[0] if isinstance(f, bytes) else f["seq"] for f in second[:-1]] == [7, 8, 9, 10]
    assert second[-1] == {"type": "RESUMED", "replayed": 4, "gap": False, "seq": 11, "session_id": "s1"}
    assert gap.gap is True                                           # 1..6 are gone


# ---------------------------------------------------------------- RTA-X3 framing, PING / PONG


def test_every_frame_is_numbered_without_gaps(make_client):
    client, _ = voice_client(make_client, chat=FakeChat(answer="好的。"))
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client)}") as ws:
        s = Session(ws, framing="seq")
        ready = s.start()
        s.send_pcm(question())
        ws.send_json({"type": "END"})
        s.drain()
    seqs = [struct.unpack_from("<I", f)[0] if isinstance(f, bytes) else f["seq"] for f in s.frames]
    assert seqs == list(range(1, len(seqs) + 1)) and "AUDIO" in s.kinds()
    assert {f["session_id"] for f in s.frames if isinstance(f, dict)} == {ready["session_id"]}
    assert ready["config"]["framing"] == "seq"


def test_raw_framing_numbers_binary_frames_too(make_client):
    client, _ = voice_client(make_client, chat=FakeChat(answer="好的。"))
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client)}") as ws:
        s = Session(ws)
        s.start()
        s.send_pcm(question())
        ws.send_json({"type": "END"})
        s.drain()
    for expected, f in enumerate(s.frames, start=1):
        if isinstance(f, dict):
            assert f["seq"] == expected                  # a raw client can count binary frames itself


def test_seq_framed_client_audio_is_deduplicated(make_client):
    client = make_client()
    pcm = np.concatenate([speech(600), silence(500)]).astype("<i2").tobytes()
    frames = [struct.pack("<I", i + 1) + pcm[off:off + 3200] for i, off in enumerate(range(0, len(pcm), 3200))]
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client)}") as ws:
        s = Session(ws, framing="seq")
        s.start()
        for f in frames[:5] + frames[2:]:                 # frames 3-5 arrive twice
            ws.send_bytes(f)
        ws.send_json({"type": "END", "seq": len(frames) + 1})
        s.drain()
    assert s.events("ENDED")[0]["audio_ms"] == 1100 and len(s.events("UTTERANCE")) == 1


def test_ping_pong(make_client):
    client = make_client(make_settings(ping_interval_s=0.15, pong_timeout_s=0.3))
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client)}") as ws:
        s = Session(ws)
        s.start()
        for _ in range(3):
            ping = s.until("PING")
            ws.send_json({"type": "PONG", "ts": ping["ts"]})
        ws.send_json({"type": "PING"})
        s.until("PONG")
        ws.send_json({"type": "END"})
        assert s.drain() == 1000


def test_no_pong_drops_the_connection(make_client, sync_redis):
    client = make_client(make_settings(ping_interval_s=0.1, pong_timeout_s=0.2))
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client)}") as ws:
        s = Session(ws)
        sid = s.start()["session_id"]
        assert s.drain() == 4408
    assert s.events("ERROR")[-1]["code"] == "pong_timeout"
    for _ in range(100):
        if hgetall(sync_redis, sid).get("state") == "ENDED":
            break
        time.sleep(0.02)
    assert hgetall(sync_redis, sid)["end_reason"] == "disconnect"


# ---------------------------------------------------------------- RTA-X3 resume


def _resume(client, sid: str, last_seq: int, user: str = "user-1", query: bool = True):
    url = f"/audio/stream?token={issue_ticket(client, user)}" + (f"&session_id={sid}" if query else "")
    ws = client.websocket_connect(url).__enter__()
    ws.send_json({"type": "RESUME", "session_id": sid, "last_seq": last_seq})
    return ws


def test_suspended_session_returns_vad_model_for_other_sessions(make_client, sync_redis):
    from .conftest import AmplitudeVad

    client = make_client(settings=make_settings(resume_window_s=2))
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client)}") as ws:
        s = Session(ws)
        sid = s.start()["session_id"]
        last = s.last_seq()
        ws.close()
        for _ in range(100):
            if hgetall(sync_redis, sid).get("connection") == "suspended":
                break
            time.sleep(0.01)
        assert len(client.app.state.vad_pool._free) == 1
        with client.websocket_connect(f"/audio/stream?token={issue_ticket(client)}") as other:
            other_session = Session(other)
            other_session.start()
            assert len(AmplitudeVad.instances) == 1
            other.send_json({"type": "END"})
            assert other_session.drain() == 1000
        resumed = _resume(client, sid, last)
        resumed_session = Session(resumed)
        resumed_session.until("RESUMED")
        resumed.send_json({"type": "END"})
        assert resumed_session.drain() == 1000
        resumed.__exit__(None, None, None)


def test_suspended_output_pauses_production_until_the_resume(make_client):
    client, _ = voice_client(make_client, chat=FakeChat(answer="這是很長的一段回答，" * 20),
                             resume_window_s=5, output_lead_ms=0, output_queue_ms=100)
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client)}") as ws:
        s = Session(ws, framing="seq")
        sid = s.start()["session_id"]
        s.send_pcm(question())
        s.until("RESPONSE_START")
        last = s.last_seq()
        ws.close()
        time.sleep(1.3)
        conn = client.app.state.stream_handler.live[sid]
        waiting = conn._turn_task is not None and not conn._turn_task.done()     # not given up
        unsent = conn.outbox.unsent_ms
        resumed = _resume(client, sid, last)
        try:
            s2 = Session(resumed, framing="seq")
            s2.until("RESUMED")
            end = s2.until("RESPONSE_END")
            resumed.send_json({"type": "END"})
            assert s2.drain() == 1000
        finally:
            resumed.__exit__(None, None, None)
    assert waiting and unsent <= 100 + 50 * len("這是很長的一段回答，"), (waiting, unsent)   # not running ahead
    assert not end.get("interrupted") and end["duration_ms"] == 50 * 200 + 200, end       # the whole answer


def test_close_logs_flush_timeout_and_delivers_fatal_error(make_client, caplog):
    from app.stream import Connection
    from starlette.websockets import WebSocketState

    client = make_client()
    conn = Connection(client.app.state.stream_handler, "closing-session")
    events = []

    class Socket:
        application_state = WebSocketState.CONNECTED

        async def send_json(self, event):
            events.append(event)

        async def close(self, code):
            events.append(code)

    async def timeout():
        raise asyncio.TimeoutError

    conn.ws = Socket()
    conn.outbox.flush = timeout
    client.portal.call(conn.close, 1011, "internal_error", "failed")
    assert events == [{"type": "ERROR", "code": "internal_error", "message": "failed"}, 1011]
    assert "output flush timed out" in caplog.text


def test_answer_continues_after_a_dropped_connection_is_resumed(make_client, sync_redis):
    client, chat = voice_client(make_client, resume_window_s=5, output_lead_ms=200)
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client)}") as ws1:
        s1 = Session(ws1, framing="seq")
        sid = s1.start()["session_id"]
        s1.send_pcm(question())
        s1.until("RESPONSE_START")
        s1.audio_frames(5)
        last = s1.last_seq()
        time.sleep(0.3)                                    # frames keep arriving, unread ...
        ws1.close()                                        # ... when the line drops mid-answer
        time.sleep(0.4)
        record = hgetall(sync_redis, sid)
        assert record["connection"] == "suspended" and float(record["resume_until"]) > time.time()
        assert sync_redis.zscore(ACTIVE_KEY, sid) is not None      # the session keeps its slot
        ws2 = _resume(client, sid, last)
        s2 = Session(ws2, framing="seq")
        resumed = s2.until("RESUMED")
        end = s2.until("RESPONSE_END")
        ws2.send_json({"type": "END"})
        assert s2.drain() == 1000
        ws2.__exit__(None, None, None)
    assert resumed["replayed"] > 0 and resumed["gap"] is False and resumed["session_id"] == sid
    frames = s1.frames + s2.frames
    seqs = [struct.unpack_from("<I", f)[0] if isinstance(f, bytes) else f["seq"] for f in frames]
    assert seqs == list(range(1, len(seqs) + 1))                       # nothing lost, nothing twice
    audio = [f for f in frames if isinstance(f, bytes)]
    assert abs(sum(len(f) - 4 for f in audio) / 32 - end["duration_ms"]) <= 1   # the whole answer
    assert len(chat.requests) == 1 and s2.events("ENDED")[0]["resumes"] == 1
    assert hgetall(sync_redis, sid)["state"] == "ENDED" and sync_redis.zcard(ACTIVE_KEY) == 0


def test_an_answer_longer_than_the_output_queue_waits_for_the_resume(make_client):
    # The answer is longer than RT_OUTPUT_QUEUE_MS and the client is away longer than a stall
    # would be tolerated: synthesis waits for the resume instead of giving up.
    client, _ = voice_client(make_client, resume_window_s=5, output_lead_ms=200, output_queue_ms=400)
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client)}") as ws1:
        s1 = Session(ws1, framing="seq")
        sid = s1.start()["session_id"]
        s1.send_pcm(question())
        s1.until("RESPONSE_START")
        s1.audio_frames(2)
        last = s1.last_seq()
        ws1.close()
        time.sleep(2.5)
        ws2 = _resume(client, sid, last)
        s2 = Session(ws2, framing="seq")
        s2.until("RESUMED")
        end = s2.until("RESPONSE_END")
        ws2.send_json({"type": "END"})
        assert s2.drain() == 1000
        ws2.__exit__(None, None, None)
    assert not end.get("interrupted")
    audio = [f for f in s1.frames + s2.frames if isinstance(f, bytes)]
    assert abs(sum(len(f) - 4 for f in audio) / 32 - end["duration_ms"]) <= 1   # the whole answer


def test_resume_frame_on_a_fresh_connection_works_too(make_client):
    client, _ = voice_client(make_client, resume_window_s=5)
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client)}") as ws1:
        s1 = Session(ws1)
        sid = s1.start()["session_id"]
        ws1.close()
        time.sleep(0.2)
        ws2 = _resume(client, sid, s1.last_seq(), query=False)
        s2 = Session(ws2)
        s2.until("RESUMED")
        ws2.send_json({"type": "END"})
        assert s2.drain() == 1000
        ws2.__exit__(None, None, None)


def test_a_newer_connection_takes_the_session_over(make_client):
    client, _ = voice_client(make_client, resume_window_s=5)
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client)}") as ws1:
        s1 = Session(ws1)
        sid = s1.start()["session_id"]
        ws2 = _resume(client, sid, s1.last_seq())
        assert s1.drain() == 4409
        s2 = Session(ws2)
        assert s2.until("RESUMED")["replayed"] == 0
        ws2.send_json({"type": "END"})
        assert s2.drain() == 1000
        ws2.__exit__(None, None, None)


def test_resume_is_refused_for_another_user_and_after_expiry(make_client, sync_redis):
    client, _ = voice_client(make_client, resume_window_s=0.5)
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client)}") as ws1:
        s1 = Session(ws1)
        sid = s1.start()["session_id"]
        ws1.close()
        time.sleep(0.1)
        other = Session(_resume(client, sid, 0, user="mallory"))
        assert other.drain() == 4410 and other.events("ERROR")[0]["code"] == "SESSION_EXPIRED"
        other.ws.__exit__(None, None, None)
        for _ in range(100):
            if hgetall(sync_redis, sid).get("state") == "EXPIRED":
                break
            time.sleep(0.02)
        assert hgetall(sync_redis, sid)["state"] == "EXPIRED"
        late = Session(_resume(client, sid, 0))
        assert late.drain() == 4410 and late.events("ERROR")[0]["code"] == "SESSION_EXPIRED"
        late.ws.__exit__(None, None, None)
        unknown = Session(_resume(client, "no-such-session", 0))
        assert unknown.drain() == 4410
        unknown.ws.__exit__(None, None, None)
    for _ in range(100):
        if not sync_redis.zcard(ACTIVE_KEY):
            break
        time.sleep(0.02)
    assert sync_redis.zcard(ACTIVE_KEY) == 0
    metrics = client.get("/metrics").text
    assert "rt_sessions_expired_total 1.0" in metrics and "rt_resume_rejected_total 3.0" in metrics
