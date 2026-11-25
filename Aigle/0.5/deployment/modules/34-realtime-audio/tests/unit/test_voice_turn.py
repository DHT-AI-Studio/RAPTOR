"""TTS, the Module 15 bridge and spoken voice turns."""
from __future__ import annotations

import asyncio
import json
import time

import httpx
import numpy as np
import pytest
from app.codec import make_encoder
from app.config import Settings
from app.llm_bridge import LlmBridge, LlmError, SentenceSplitter
from app.response import Speaker, speakable
from app.tts import TtsService
from app.tts_backends import (
    REGISTRY,
    BackendUnavailable,
    UnknownBackend,
    backend_class,
    load_backend,
)
from app.tts_backends.breeze import LICENCE, BreezeBackend

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

# ---------------------------------------------------------------- backends and dispatcher


def test_registry_lists_both_backends_and_rejects_unknown_names():
    assert set(REGISTRY) == {"kokoro-82m", "breeze-tts-2"}
    with pytest.raises(UnknownBackend) as exc:
        load_backend("speecht5", Settings())
    assert "available: breeze-tts-2, kokoro-82m" in str(exc.value)


def test_breeze_is_unavailable_until_its_licence_is_accepted():
    with pytest.raises(BackendUnavailable) as exc:
        backend_class("breeze-tts-2").check(Settings())
    assert str(exc.value) == LICENCE and "non-commercial" in LICENCE
    with pytest.raises(BackendUnavailable, match="RT_TTS_BREEZE_URL"):
        backend_class("breeze-tts-2").check(Settings(tts_breeze_enabled=True, tts_breeze_url=""))
    with pytest.raises(BackendUnavailable, match="not reachable"):
        backend_class("breeze-tts-2").check(Settings(tts_breeze_enabled=True, tts_breeze_url="http://127.0.0.1:9"))


@pytest.mark.parametrize("presets", ["{", "{}", "null", "[]", '{"v":null}', '{"v":{}}',
                                      '{"v":{"instruction":12}}', '{"v":{"ref_audio":"a.wav"}}'])
def test_breeze_rejects_invalid_voice_configuration(presets):
    with pytest.raises(BackendUnavailable, match="RT_TTS_BREEZE_VOICES"):
        BreezeBackend(Settings(tts_breeze_voices=presets))


def test_model_status_checks_are_cached_between_polls(monkeypatch):
    calls = []
    monkeypatch.setattr(BreezeBackend, "check", lambda _s: calls.append("check"))

    async def scenario():
        svc = TtsService(Settings())
        svc.backend = FakeTts()
        first = await svc.models()
        assert await svc.models() == first
        assert calls == ["check"]
        svc._models_checked_at -= 61
        await svc.models()
        assert calls == ["check", "check"]

    asyncio.run(scenario())


def test_kokoro_is_unavailable_without_weights(tmp_path):
    with pytest.raises(BackendUnavailable) as exc:
        backend_class("kokoro-82m").check(Settings(tts_kokoro_dir=str(tmp_path)))
    assert "package missing" in str(exc.value) or "no Kokoro weights" in str(exc.value)


def test_unknown_voice_falls_back_to_the_backend_default(caplog):
    fake = FakeTts()
    svc = TtsService(Settings(tts_voice="nope"), loader=lambda _n, _s: fake)
    asyncio.run(svc.start())
    assert svc.voice == "fake-f" and "RT_TTS_VOICE 'nope'" in caplog.text
    svc = TtsService(Settings(tts_voice="fake-m"), loader=lambda _n, _s: fake)
    asyncio.run(svc.start())
    assert svc.voice == "fake-m"
    assert fake.calls[0] == ("您好。", "fake-f", "zh-TW")                  # warm-up


def test_dispatcher_streams_pieces_from_the_worker_thread():
    fake = FakeTts()
    svc = TtsService(Settings(), loader=lambda _n, _s: fake)

    async def scenario():
        await svc.start()
        return [piece async for piece in svc.stream("你好", "zh-TW")]

    pieces = asyncio.run(scenario())
    assert len(pieces) == 2 and sum(len(p) for p in pieces) == 2400       # 2 chars x 50 ms x 24 kHz
    assert pieces[0].dtype == np.float32


def test_dispatcher_raises_backend_errors():
    svc = TtsService(Settings(), loader=lambda _n, _s: FakeTts())

    async def scenario():
        await svc.start()
        svc.backend.fail = True
        return [p async for p in svc.stream("x", None)]

    with pytest.raises(RuntimeError, match="synthesis exploded"):
        asyncio.run(scenario())


def test_models_reports_loaded_and_unavailable_backends(tmp_path):
    svc = TtsService(Settings(tts_kokoro_dir=str(tmp_path)), loader=lambda _n, _s: FakeTts())

    async def scenario():
        await svc.start()
        return await svc.models()

    models = {m["name"]: m for m in asyncio.run(scenario())}
    assert models["kokoro-82m"]["status"] == "loaded" and models["kokoro-82m"]["active"] is True
    assert [v["id"] for v in models["kokoro-82m"]["voices"]] == ["fake-f", "fake-m"]
    assert models["breeze-tts-2"] == {"name": "breeze-tts-2", "active": False, "status": "unavailable",
                                      "reason": LICENCE}


# ---------------------------------------------------------------- speaking a response


async def _sentences(*texts, fail_after: int | None = None):
    for i, t in enumerate(texts):
        if fail_after is not None and i == fail_after:
            raise LlmError("LLM_TIMEOUT", "too slow")
        yield t


class Lane:
    """Records what a Speaker queues (the Outbox in the service)."""

    def __init__(self) -> None:
        self.out: list = []

    async def response_event(self, event, request_id):
        self.out.append(event)

    async def audio(self, data, audio_ms, request_id):
        self.out.append(data)


def _speak(codec: str, rate: int, texts, **kw):
    svc = TtsService(Settings(), loader=lambda _n, _s: FakeTts())
    lane = Lane()

    async def scenario():
        await svc.start()
        speaker = Speaker(svc, make_encoder(codec, rate), lane, "zh-TW", 200)
        return await speaker.speak(_sentences(*texts, **kw), "req-1")

    return asyncio.run(scenario()), lane.out


def test_response_frames_are_framed_by_start_and_end():
    result, out = _speak("pcm", 16000, ["你好。", "今天天氣很好！"])
    kinds = ["bytes" if isinstance(o, bytes) else o["type"] for o in out]
    assert kinds[0] == "RESPONSE_TEXT" and kinds[1] == "RESPONSE_START" and kinds[2] == "bytes"
    assert kinds[-1] == "RESPONSE_END" and kinds[-2] == "bytes"
    assert out[1] == {"type": "RESPONSE_START", "request_id": "req-1",
                      "output": {"codec": "pcm", "sample_rate": 16000, "frame_ms": None}}
    assert [o["text"] for o in out if isinstance(o, dict) and o["type"] == "RESPONSE_TEXT"] == ["你好。", "今天天氣很好！"]
    frames = [o for o in out if isinstance(o, bytes)]
    assert max(len(f) for f in frames) <= 4096
    pcm = np.frombuffer(b"".join(frames), dtype="<i2")
    speech_ms = 50 * (3 + 7)
    assert abs(len(pcm) / 16 - (speech_ms + 200)) <= 2                  # + 200 ms silence padding
    assert np.all(pcm[-16 * 150:] == 0)
    assert out[-1] == {"type": "RESPONSE_END", "request_id": "req-1", "duration_ms": speech_ms + 200, "sentences": 2}
    assert result["sentences"] == 2 and result["started"] is True


def test_speakable_strips_screen_markup():
    assert speakable("1.**便利商店**（如 7-11、全家）：") == "1.便利商店（如 7-11、全家）："
    assert speakable("## 步驟\n*   檢查 `ping` 結果\n> 注意 https://example.com/a?b=1 😀") == "步驟 檢查 ping 結果 注意"
    assert speakable("3 * 4 = 12，a*b") == "3 * 4 = 12，a*b"          # arithmetic stays
    _, out = _speak("pcm", 16000, ["**", "好的。"])                    # nothing speakable: skipped
    assert [o["text"] for o in out if isinstance(o, dict) and o["type"] == "RESPONSE_TEXT"] == ["好的。"]


def test_telephony_response_is_20ms_g711_frames():
    _, out = _speak("pcma", 8000, ["請稍候。"])
    frames = [o for o in out if isinstance(o, bytes)]
    assert {len(f) for f in frames} == {160}
    assert len(frames) == (50 * 4 + 200) // 20


def test_llm_failure_mid_response_still_ends_it():
    lane = Lane()

    async def scenario():
        svc = TtsService(Settings(), loader=lambda _n, _s: FakeTts())
        await svc.start()
        speaker = Speaker(svc, make_encoder("pcm", 16000), lane, "zh-TW")
        await speaker.speak(_sentences("第一句。", "第二句。", fail_after=1), "r")

    with pytest.raises(LlmError):
        asyncio.run(scenario())
    assert lane.out[-1]["type"] == "RESPONSE_END" and lane.out[-1]["sentences"] == 1


# ---------------------------------------------------------------- sentences and the Module 15 bridge


def test_sentence_splitter():
    sp = SentenceSplitter()
    assert sp.feed("你好") == []
    assert sp.feed("。今天") == ["你好。"]
    assert sp.feed("天氣很好！明天呢？") == ["今天天氣很好！", "明天呢？"]
    assert sp.feed("版本 3.5 已經發布. Next") == ["版本 3.5 已經發布."]       # a decimal is not an end
    assert sp.feed("嗯。") == ["Next嗯。"]
    assert sp.flush() == []
    sp = SentenceSplitter()
    assert sp.feed("好。") == []                                                  # 1 letter: joins the next
    assert sp.feed("我們走吧。") == ["好。我們走吧。"]
    long = "這是一段很長的回答，" * 7                                             # 70 chars, no terminator
    assert sp.feed(long + "還沒結束") == [long]                                      # cut at its last comma
    assert sp.flush() == ["還沒結束"]


def _bridge(chat: FakeChat, timeout_s: float = 2.0, stream: bool = True) -> LlmBridge:
    return LlmBridge("http://chat:8021", timeout_s, stream, httpx.AsyncClient(transport=httpx.MockTransport(chat)))


async def _collect(bridge: LlmBridge):
    return [s async for s in bridge.sentences("今天天氣如何", "sess-1", "user-1", "req-1")]


def test_bridge_streams_sentences_from_sse():
    chat = FakeChat(answer="今天|天氣很好。我們|去散步吧！")
    assert asyncio.run(_collect(_bridge(chat))) == ["今天天氣很好。", "我們去散步吧！"]
    req = chat.requests[0]
    assert req["url"] == "http://chat:8021/chat" and req["request_id"] == "req-1"
    assert req["body"] == {"message": "今天天氣如何", "session_id": "sess-1", "user_id": "user-1", "stream": True}


def test_bridge_accepts_a_plain_json_answer():
    chat = FakeChat(answer="你好。很高興見到你", stream=False)
    assert asyncio.run(_collect(_bridge(chat, stream=False))) == ["你好。", "很高興見到你"]
    assert chat.requests[0]["body"]["stream"] is False


@pytest.mark.parametrize(("chat", "code"), [
    (FakeChat(status=500), "LLM_UNAVAILABLE"),
    (FakeChat(delay_s=1.0), "LLM_TIMEOUT"),
])
def test_bridge_errors(chat, code):
    with pytest.raises(LlmError) as exc:
        asyncio.run(_collect(_bridge(chat, timeout_s=0.3)))
    assert exc.value.code == code


def test_bridge_error_event_and_unreachable_service():
    async def error_stream(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              content=b'event: start\ndata: {}\n\nevent: error\ndata: {"message": "graph exploded"}\n\n')

    bridge = LlmBridge("http://chat", 2, True, httpx.AsyncClient(transport=httpx.MockTransport(error_stream)))
    with pytest.raises(LlmError, match="graph exploded"):
        asyncio.run(_collect(bridge))

    async def refuse(request):
        raise httpx.ConnectError("connection refused")

    bridge = LlmBridge("http://chat", 2, True, httpx.AsyncClient(transport=httpx.MockTransport(refuse)))
    with pytest.raises(LlmError) as exc:
        asyncio.run(_collect(bridge))
    assert exc.value.code == "LLM_UNAVAILABLE"


@pytest.mark.parametrize("tail", [b'event: delta\ndata: {"text":"lost"}',
                                  b'event: delta\ndata: {"text":"lost"}\n',
                                  b'event: delta\ndata: {"text":"lost"}\n\n',
                                  b'event: end\ndata: {}\n', b''])
def test_bridge_rejects_truncated_sse(tail):
    async def truncated(_request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              content=b'event: start\ndata: {}\n\n' + tail)

    bridge = LlmBridge("http://chat", 2, True, httpx.AsyncClient(transport=httpx.MockTransport(truncated)))
    with pytest.raises(LlmError) as exc:
        asyncio.run(_collect(bridge))
    assert exc.value.code == "LLM_UNAVAILABLE"


def test_time_spent_speaking_is_not_charged_to_the_llm_timeout():
    async def slow_but_steady(request):
        async def body():
            yield b"event: start\ndata: {}\n\n"
            for i in range(3):
                await asyncio.sleep(0.1)
                yield f'event: delta\ndata: {{"text": "第{i}句。"}}\n\n'.encode()
            yield b"event: end\ndata: {}\n\n"

        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body())

    bridge = LlmBridge("http://chat", 0.6, True, httpx.AsyncClient(transport=httpx.MockTransport(slow_but_steady)))

    async def consume_slowly():
        out = []
        async for s in bridge.sentences("q", "s", "u", "r"):
            out.append(s)
            await asyncio.sleep(0.3)              # "speaking": 0.9 s in total, longer than the timeout
        return out

    assert asyncio.run(consume_slowly()) == ["第0句。", "第1句。", "第2句。"]


def test_sse_stall_after_start_times_out():
    async def stall(request):
        async def body():
            yield b"event: start\ndata: {}\n\n"
            await asyncio.sleep(2)
            yield b"event: end\ndata: {}\n\n"

        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body())

    bridge = LlmBridge("http://chat", 0.3, True, httpx.AsyncClient(transport=httpx.MockTransport(stall)))
    started = time.monotonic()
    with pytest.raises(LlmError) as exc:
        asyncio.run(_collect(bridge))
    assert exc.value.code == "LLM_TIMEOUT" and time.monotonic() - started < 1.5


# ---------------------------------------------------------------- a whole voice turn


def _turn(client, config=None, audio=None):
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client, sub='user-42')}") as ws:
        ws.send_json(config or PCM_CONFIG)
        ready = ws.receive_json()
        while ready["type"] != "READY":
            ready = ws.receive_json()
        pcm = audio if audio is not None else np.concatenate([speech(800), silence(500)])
        for off in range(0, len(pcm), 1600):
            ws.send_bytes(pcm[off:off + 1600].astype("<i2").tobytes())
        ws.send_json({"type": "END"})
        out = []
        while True:
            msg = ws.receive()
            if msg["type"] == "websocket.close":
                return ready, out, msg.get("code", 1000)
            out.append(msg["bytes"] if msg.get("bytes") is not None else json.loads(msg["text"]))


def _kinds(out):
    return ["AUDIO" if isinstance(o, bytes) else o["type"] for o in out]


def test_voice_turn_end_to_end(make_client):
    chat = FakeChat(answer="你好，|我是語音助理。有什麼|可以幫你？")
    tts = FakeTts()
    client = make_client(stt=FakeWhisper(), tts=tts, chat=chat, settings=make_settings(stt_streaming=False))
    ready, out, code = _turn(client)
    assert code == 1000
    assert ready["tts"] == {"model": "kokoro-82m", "voice": "fake-f", "sample_rate": 24000}
    assert ready["output"] == {"codec": "pcm", "sample_rate": 16000, "frame_ms": None}
    kinds = _kinds(out)
    final = kinds.index("TRANSCRIPT_FINAL")
    start = kinds.index("RESPONSE_START")
    assert final < start < kinds.index("AUDIO") < kinds.index("RESPONSE_END") < kinds.index("ENDED")
    states = [o["state"] for o in out if isinstance(o, dict) and o["type"] == "STATE"]
    # A slow-to-queue next sentence can let the outbox drain and fire on_idle before it's ready,
    # a benign RESPONDING->PROCESSING->RESPONDING blip between sentences of the SAME answer, not
    # a second turn -- assert the overall shape (starts processing, ends listening), not the
    # exact sequence of PROCESSING/RESPONDING flips in between.
    assert states[0] == "PROCESSING" and states[-1] == "LISTENING"
    assert set(states[1:-1]) <= {"PROCESSING", "RESPONDING"}
    assert [o["text"] for o in out if isinstance(o, dict) and o["type"] == "RESPONSE_TEXT"] == \
        ["你好，我是語音助理。", "有什麼可以幫你？"]
    req = chat.requests[0]
    assert req["body"]["message"] == "你好嗎" and req["body"]["user_id"] == "user-42"
    assert req["body"]["session_id"] == ready["session_id"]
    assert out[start]["request_id"] == req["request_id"]
    assert [c[0] for c in tts.calls] == ["你好，我是語音助理。", "有什麼可以幫你？"]
    assert out[-1]["responses"] == 1


def test_llm_timeout_speaks_the_canned_answer_and_keeps_the_session(make_client):
    chat = FakeChat(delay_s=1.0)
    client = make_client(stt=FakeWhisper(), tts=FakeTts(), chat=chat,
                         settings=make_settings(stt_streaming=False, llm_timeout_s=0.2))
    _, out, code = _turn(client, audio=np.concatenate([speech(800), silence(500), speech(800), silence(500)]))
    assert code == 1000
    errors = [o for o in out if isinstance(o, dict) and o["type"] == "ERROR"]
    assert [e["code"] for e in errors] == ["LLM_TIMEOUT", "LLM_TIMEOUT"] and all(e["fatal"] is False for e in errors)
    spoken = [o for o in out if isinstance(o, dict) and o["type"] == "RESPONSE_TEXT"]
    assert [o["text"] for o in spoken] == ["系統暫時無法回應，請稍後再試。"] * 2
    assert all(o["fallback"] is True for o in out if isinstance(o, dict) and o["type"].startswith("RESPONSE_"))
    assert _kinds(out)[-1] == "ENDED"


def test_mid_answer_timeout_gives_fallback_its_own_request_id(make_client):
    async def partial_answer(_request):
        async def body():
            yield 'event: delta\ndata: {"text":"第一句。"}\n\n'.encode()
            await asyncio.sleep(2)
            yield b'event: end\ndata: {}\n\n'

        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body())

    client = make_client(stt=FakeWhisper(), tts=FakeTts(), chat=partial_answer,
                         settings=make_settings(stt_streaming=False, llm_timeout_s=0.1))
    _, out, code = _turn(client)
    assert code == 1000
    starts = [e for e in out if isinstance(e, dict) and e["type"] == "RESPONSE_START"]
    ends = [e for e in out if isinstance(e, dict) and e["type"] == "RESPONSE_END"]
    assert len(starts) == len(ends) == 2
    ids = [e["request_id"] for e in starts]
    assert len(set(ids)) == 2 and [e["request_id"] for e in ends] == ids
    error = next(e for e in out if isinstance(e, dict) and e["type"] == "ERROR")
    assert error["request_id"] == ids[0] and starts[1]["fallback"] is True

    async def cancel_primary_only():
        from app.outbox import Outbox

        sent = []

        class Transport:
            async def send_text(self, text):
                sent.append(json.loads(text))

            async def send_bytes(self, data):
                sent.append(data)

        box = Outbox("s", pace=False)
        for i, rid in enumerate(ids):
            await box.response_event(starts[i], rid)
            await box.audio(f"audio-{i}".encode(), 20, rid)
            await box.response_event(ends[i], rid)
        await box.drop_request(ids[0])
        box.attach(Transport())
        while box.queued:
            await box._step()
        assert [e["request_id"] for e in sent if isinstance(e, dict)] == [ids[1], ids[1]]
        assert [e for e in sent if isinstance(e, bytes)] == [b"audio-1"]

    asyncio.run(cancel_primary_only())


def test_g711_session_gets_g711_frames_back(make_client):
    from app.codec import g711_encode

    client = make_client(stt=FakeWhisper(), tts=FakeTts(), chat=FakeChat(answer="好的。"),
                         settings=make_settings(stt_streaming=False))
    audio8k = np.concatenate([(np.sin(2 * np.pi * 220 * np.arange(6400) / 8000) * 0.3 * 32767).astype(np.int16),
                              np.zeros(4000, dtype=np.int16)])
    encoded = g711_encode(audio8k, "pcmu")
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client)}") as ws:
        ws.send_json({"type": "CONFIG", "codec": "pcmu", "sample_rate": 8000, "language": "zh-TW"})
        for i in range(0, len(encoded), 160):
            ws.send_bytes(encoded[i:i + 160])
        ws.send_json({"type": "END"})
        frames = []
        while True:
            msg = ws.receive()
            if msg["type"] == "websocket.close":
                break
            if msg.get("bytes") is not None:
                frames.append(msg["bytes"])
    assert frames and {len(f) for f in frames} == {160}


def test_empty_transcript_is_not_a_turn(make_client):
    chat = FakeChat()
    client = make_client(stt=FakeWhisper(words=[]), tts=FakeTts(), chat=chat,
                         settings=make_settings(stt_streaming=False))
    _, out, _ = _turn(client)
    assert chat.requests == [] and "RESPONSE_START" not in _kinds(out)


def test_tts_models_endpoint(make_client, tmp_path):
    assert make_client().get("/tts/models").json() == {"enabled": False, "models": []}
    body = make_client(tts=FakeTts(), settings=make_settings(tts_kokoro_dir=str(tmp_path))).get("/tts/models").json()
    status = {m["name"]: m["status"] for m in body["models"]}
    assert body["enabled"] is True and status == {"kokoro-82m": "loaded", "breeze-tts-2": "unavailable"}


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("RT_TTS_MODEL", "breeze-tts-2")
    monkeypatch.setenv("RT_TTS_BREEZE_ENABLED", "true")
    monkeypatch.setenv("RT_CHAT_STREAM", "false")
    monkeypatch.setenv("RT_LLM_TIMEOUT_S", "12")
    s = Settings.from_env()
    assert (s.tts_model, s.tts_breeze_enabled, s.chat_stream, s.llm_timeout_s) == ("breeze-tts-2", True, False, 12.0)
