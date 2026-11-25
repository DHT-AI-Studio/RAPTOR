"""RA-3 / RTA-X2: final and rolling-window transcripts (app/stt.py) and their WebSocket events."""
from __future__ import annotations

import asyncio
import re
import sys
import time
from typing import Any

import numpy as np
import pytest
from app.stt import (
    HFWhisperBackend,
    RollingParams,
    SttOptions,
    SttService,
    SttSession,
    Transcript,
    Word,
    _hf_chunks_to_segments,
    _hf_model_id,
    lcp,
    norm,
    resolve_model,
    wants_traditional,
    whisper_language,
)

from .conftest import (
    PCM_CONFIG,
    FakeWhisper,
    issue_ticket,
    make_settings,
    silence,
    speech,
)
from .test_stream import collect, types

# ---------------------------------------------------------------- helpers


class Scripted:
    """Backend returning one scripted word list per call (times relative to the window)."""

    name, device, compute_type = "scripted", "cpu", "int8"

    def __init__(self, provisional: list[list[tuple[str, int, int]]], final: list[tuple[str, int, int]]):
        self.provisional = list(provisional)
        self.final = final
        self.calls: list[dict[str, Any]] = []

    def transcribe(self, audio, language, *, beam_size, prompt=None, on_words=None, max_new_tokens=None,
                   fallback=True):
        self.calls.append({"samples": len(audio), "prompt": prompt, "beam_size": beam_size})
        spec = self.final if beam_size > 1 else self.provisional.pop(0)
        words = tuple(Word(t, s, e, 0.8) for t, s, e in spec)
        if on_words and words:
            on_words(list(words), "zh")
        return Transcript("".join(w.text for w in words), words, "zh", 0.99)


def make_service(backend, **opts) -> SttService:
    svc = SttService(lambda _name: backend, SttOptions(zh_convert="", **opts))
    svc.final = svc.provisional = backend
    return svc


def test_partial_preserves_mixed_language_word_spacing():
    async def scenario():
        backend = Scripted([], [("我要", 0, 100), (" call", 100, 200), (" 你", 200, 300)])
        sent = []

        async def send(event):
            sent.append(event)

        session = SttSession(make_service(backend), "zh-TW", send, RollingParams())
        await session.finalize(1, np.zeros(16000, dtype=np.int16), 1000)
        partials = [e["text"] for e in sent if e["type"] == "TRANSCRIPT_PARTIAL"]
        assert partials == ["我要", " call", " 你"]
        assert "".join(partials) == sent[-1]["text"] == "我要 call 你"

    asyncio.run(scenario())


async def run_rolling(session: SttSession, index: int, speech_ms: int, audio_ms: int) -> None:
    session.speech_progress(index, speech_ms, lambda: np.zeros(audio_ms * 16, dtype=np.int16))
    snap, session._snapshot = session._snapshot, None
    if snap is not None:
        await session._decode_window(*snap)


# ---------------------------------------------------------------- languages and text


@pytest.mark.parametrize(("tag", "detected", "expected"), [
    ("zh-TW", "zh", True), ("zh-Hant", "zh", True), ("zh-HK", "zh", True), ("zh", "zh", True),
    ("zh-CN", "zh", False), ("zh-Hans-TW", "zh", False), ("en-US", "en", False),
    (None, "zh", True), (None, "en", False),
])
def test_wants_traditional(tag, detected, expected):
    assert wants_traditional(tag, detected) is expected


def test_whisper_language_uses_primary_subtag():
    assert whisper_language("zh-TW") == "zh"
    assert whisper_language("en-US") == "en"
    assert whisper_language(None) is None


def test_zh_conversion_keeps_the_speakers_words():
    pytest.importorskip("opencc")
    default = SttService(lambda _n: None, SttOptions())
    assert default.convert("确认图表中的数据，我们用软件打印") == "確認圖表中的數據，我們用軟件打印"
    phrases = SttService(lambda _n: None, SttOptions(zh_convert="s2twp"))     # opt-in vocabulary swap
    assert phrases.convert("我们用软件打印") == "我們用軟體列印"


def test_words_are_converted_as_one_text():
    pytest.importorskip("opencc")

    async def send(_event):
        pass

    session = SttSession(SttService(lambda _n: None, SttOptions()), "zh-TW", send, RollingParams())
    words = [Word("我", 0, 1, 1), Word("的头", 1, 2, 1), Word("发", 2, 3, 1)]
    assert [w.text for w in session._post_words(words, "zh")] == ["我", "的頭", "髮"]
    simplified = SttSession(SttService(lambda _n: None, SttOptions()), "zh-CN", send, RollingParams())
    assert simplified._post_words(words, "zh") == words


def test_resolve_model_prefers_local_directories(tmp_path):
    (tmp_path / "faster-whisper-large-v3").mkdir()
    (tmp_path / "custom").mkdir()
    assert resolve_model("large-v3", str(tmp_path)) == str(tmp_path / "faster-whisper-large-v3")
    assert resolve_model("custom", str(tmp_path)) == str(tmp_path / "custom")
    assert resolve_model(str(tmp_path / "custom"), "") == str(tmp_path / "custom")
    assert resolve_model("medium", str(tmp_path)) == "medium"          # left to the HF cache


def test_hf_model_id_prefixes_a_bare_size_name():
    assert _hf_model_id("large-v3") == "openai/whisper-large-v3"
    assert _hf_model_id("tiny") == "openai/whisper-tiny"


def test_hf_model_id_leaves_a_repo_id_or_path_alone(tmp_path):
    assert _hf_model_id("openai/whisper-base") == "openai/whisper-base"
    local = tmp_path / "my-finetune"
    local.mkdir()
    assert _hf_model_id(str(local)) == str(local)


def test_hf_chunks_to_segments_splits_on_pause_gap():
    chunks = [
        {"text": " hello", "timestamp": (0.0, 0.4)},
        {"text": " world", "timestamp": (0.5, 0.9)},
        # 1.5s gap here, well past the 0.8s threshold
        {"text": " again", "timestamp": (2.4, 2.8)},
    ]
    segments = _hf_chunks_to_segments(chunks)
    assert [[w.text for w in seg] for seg in segments] == [[" hello", " world"], [" again"]]
    assert segments[0][0].start_ms == 0 and segments[0][1].end_ms == 900


def test_hf_chunks_to_segments_splits_on_sentence_end_punctuation():
    chunks = [
        {"text": " Hi there.", "timestamp": (0.0, 0.5)},
        {"text": " Bye", "timestamp": (0.5, 0.8)},  # no gap, but the prior chunk ended a sentence
    ]
    segments = _hf_chunks_to_segments(chunks)
    assert [[w.text for w in seg] for seg in segments] == [[" Hi there."], [" Bye"]]


def test_hf_chunks_to_segments_skips_empty_chunks_and_fills_missing_timestamps():
    chunks = [
        {"text": " hello", "timestamp": (0.0, 0.4)},
        {"text": "   ", "timestamp": (0.4, 0.5)},         # whitespace-only: dropped
        {"text": " world", "timestamp": (None, None)},     # missing: falls back to last_end
    ]
    segments = _hf_chunks_to_segments(chunks)
    assert [w.text for seg in segments for w in seg] == [" hello", " world"]
    assert segments[0][1].start_ms == 400 and segments[0][1].end_ms == 400


def test_cuda_available_uses_torch_on_arm64(monkeypatch):
    import app.stt as stt_mod

    monkeypatch.setattr(stt_mod, "_IS_ARM64", True)

    class FakeTorchCuda:
        @staticmethod
        def is_available():
            return True

    class FakeTorch:
        cuda = FakeTorchCuda()

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    assert stt_mod.cuda_available() is True


def test_cuda_available_false_on_arm64_without_a_cuda_runtime(monkeypatch):
    import app.stt as stt_mod

    monkeypatch.setattr(stt_mod, "_IS_ARM64", True)
    # A None entry in sys.modules makes `import torch` raise ImportError --
    # standard Python import-system behaviour, not a monkeypatch trick.
    monkeypatch.setitem(sys.modules, "torch", None)
    assert stt_mod.cuda_available() is False


def test_hf_whisper_backend_pins_num_beams_to_1_regardless_of_beam_size(monkeypatch):
    """transformers==4.51.3's word-timestamp extraction crashes with num_beams > 1
    on a long-form decode (confirmed live on GX10) -- beam_size must never reach
    generate_kwargs["num_beams"] as anything but 1."""

    class FakeTorch:
        float16 = "float16"
        float32 = "float32"

    captured: dict[str, Any] = {}

    class FakePipe:
        tokenizer = None

        def __call__(self, inputs, *, return_timestamps, return_language, generate_kwargs):
            captured["generate_kwargs"] = generate_kwargs
            return {"chunks": [], "language": "zh"}

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    import types

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.pipeline = lambda *a, **k: FakePipe()
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    backend = HFWhisperBackend("openai/whisper-large-v3", "cuda", "float16")
    backend.transcribe(np.zeros(1600, dtype=np.float32), "zh", beam_size=5)
    assert captured["generate_kwargs"]["num_beams"] == 1


def test_norm_and_lcp():
    assert norm("Hello, 世界！") == "hello世界"
    assert lcp(norm("今天，好"), norm("今天氣")) == 2


# ---------------------------------------------------------------- rolling decodes (RTA-X2)


def _words(*texts: str, step: int = 200) -> list[tuple[str, int, int]]:
    return [(t, i * step, (i + 1) * step) for i, t in enumerate(texts)]


def _session(backend, sent: list) -> SttSession:
    async def send(event):
        sent.append(event)

    return SttSession(make_service(backend), "zh-TW", send, RollingParams(stride_s=0.5, first_decode_ms=300))


def test_rolling_promotes_what_two_decodes_agree_on():
    sent: list[dict] = []
    backend = Scripted(
        provisional=[
            _words("今", "天"),
            _words("今", "天", "天", "器"),
            _words("今", "天", "天", "氣", "很"),
        ],
        final=_words("今", "天", "天", "氣", "很", "好"),
    )
    session = _session(backend, sent)

    async def scenario():
        await run_rolling(session, 1, 200, 400)          # before RT_STT_FIRST_DECODE_MS: no decode
        assert backend.calls == []
        await run_rolling(session, 1, 300, 500)
        await run_rolling(session, 1, 500, 700)          # before the next stride: skipped
        await run_rolling(session, 1, 800, 1000)
        await run_rolling(session, 1, 1300, 1500)
        session.utterance_closed(1)
        await run_rolling(session, 1, 1800, 2000)        # closed: nothing more
        await session.finalize(1, np.zeros(16 * 1200, dtype=np.int16), 1200)

    asyncio.run(scenario())
    assert len(backend.calls) == 4                       # 3 rolling + 1 final
    assert [c["samples"] for c in backend.calls[:3]] == [8000, 16000, 24000]   # the whole utterance
    kinds = [e["type"] for e in sent]
    assert kinds[:5] == ["TRANSCRIPT_PROVISIONAL", "TRANSCRIPT_STABLE", "TRANSCRIPT_PROVISIONAL",
                         "TRANSCRIPT_STABLE", "TRANSCRIPT_PROVISIONAL"]
    assert sent[0]["text"] == "今天" and sent[0]["stable_text"] == "" and sent[0]["unstable_text"] == "今天"
    assert sent[1] == {"type": "TRANSCRIPT_STABLE", "utterance": 1, "text": "今天", "stable_text": "今天",
                       "word_index": 0, "words": 2}
    assert sent[2]["text"] == "今天天器" and sent[2]["unstable_text"] == "天器"
    # the repeated 天 is kept: agreement is by position, not by deduplicating text
    assert sent[3]["text"] == "天" and sent[3]["stable_text"] == "今天天" and sent[3]["word_index"] == 2
    assert sent[4]["stable_text"] == "今天天" and sent[4]["unstable_text"] == "氣很"
    assert "TRANSCRIPT_CORRECTION" not in kinds
    final = sent[-1]
    assert final["type"] == "TRANSCRIPT_FINAL" and final["text"] == "今天天氣很好"
    assert kinds.count("TRANSCRIPT_PARTIAL") == 6


def test_word_inserted_before_the_agreed_text_is_not_lost():
    """Regression: stitching windows by timestamps dropped 嗎 from 有空嗎我們 in real audio."""
    sent: list[dict] = []
    backend = Scripted(
        provisional=[
            _words("有空", "我們"),
            _words("有空", "嗎", "我們", "可以"),
            _words("有空", "嗎", "我們", "可以", "一起"),
        ],
        final=_words("有空", "嗎", "？", "我們", "可以", "一起", "去"),
    )
    session = _session(backend, sent)

    async def scenario():
        for ms in (300, 800, 1300):
            await run_rolling(session, 1, ms, ms)
        session.utterance_closed(1)
        await session.finalize(1, np.zeros(16 * 1400, dtype=np.int16), 1400)

    asyncio.run(scenario())
    stable = [e["stable_text"] for e in sent if e["type"] == "TRANSCRIPT_STABLE"]
    assert stable == ["有空", "有空嗎我們可以"]
    assert not [e for e in sent if e["type"] == "TRANSCRIPT_CORRECTION"]


def test_agreement_survives_different_word_splits():
    sent: list[dict] = []
    backend = Scripted(provisional=[_words("我們", "去"), _words("我", "們", "去", "吃")], final=[])
    session = _session(backend, sent)

    async def scenario():
        await run_rolling(session, 1, 300, 300)
        await run_rolling(session, 1, 800, 800)

    asyncio.run(scenario())
    stable = [e for e in sent if e["type"] == "TRANSCRIPT_STABLE"]
    assert stable[0]["stable_text"] == "我們去" and stable[0]["words"] == 3


def test_disagreeing_hypothesis_promotes_nothing_until_it_agrees_again():
    sent: list[dict] = []
    backend = Scripted(
        provisional=[_words("我", "要", "去"), _words("我", "要", "去", "吃"), _words("我", "藥", "去", "吃", "飯")],
        final=[],
    )
    session = _session(backend, sent)

    async def scenario():
        for ms in (300, 800, 1300):
            await run_rolling(session, 1, ms, ms)

    asyncio.run(scenario())
    stable = [e["stable_text"] for e in sent if e["type"] == "TRANSCRIPT_STABLE"]
    assert stable == ["我要去"]
    last = sent[-1]
    # the latest hypothesis contradicts the stable text: it is sent whole as unstable
    assert last["text"] == "我藥去吃飯" and last["stable_text"] == "我要去" and last["unstable_text"] == "我藥去吃飯"


def test_correction_when_final_contradicts_stable_text():
    sent: list[dict] = []

    async def scenario(final):
        sent.clear()
        backend = Scripted(provisional=[_words("我", "要", "去")] * 2, final=final)
        session = _session(backend, sent)
        await run_rolling(session, 1, 300, 600)
        await run_rolling(session, 1, 800, 1100)
        session.utterance_closed(1)
        await session.finalize(1, np.zeros(16 * 800, dtype=np.int16), 800)

    asyncio.run(scenario(_words("我", "要", "吃", "飯")))
    assert [e["stable_text"] for e in sent if e["type"] == "TRANSCRIPT_STABLE"] == ["我要"]
    assert not [e for e in sent if e["type"] == "TRANSCRIPT_CORRECTION"]     # 我要 is a prefix of 我要吃飯

    asyncio.run(scenario(_words("我", "藥", "去")))
    assert [e for e in sent if e["type"] == "TRANSCRIPT_CORRECTION"] == [
        {"type": "TRANSCRIPT_CORRECTION", "utterance": 1, "start": 1, "end": 2, "old_text": "要",
         "new_text": "藥", "stable_text": "我藥"}]
    assert [e["type"] for e in sent][-1] == "TRANSCRIPT_FINAL"

    # an inserted word: the replacement runs to where the stable text ends in the final text
    async def inserted():
        sent.clear()
        backend = Scripted(provisional=[_words("有空", "我們", "去", "吃")] * 2,
                           final=_words("有空", "嗎", "，", "我們", "去", "吃", "飯"))
        session = _session(backend, sent)
        await run_rolling(session, 1, 300, 600)
        await run_rolling(session, 1, 800, 1100)
        session.utterance_closed(1)
        await session.finalize(1, np.zeros(16 * 1400, dtype=np.int16), 1400)

    asyncio.run(inserted())
    corr = next(e for e in sent if e["type"] == "TRANSCRIPT_CORRECTION")
    assert corr["old_text"] == "我們去" and corr["new_text"] == "嗎，我們去" and corr["stable_text"] == "有空嗎，我們去"


def test_discarded_utterance_resets_rolling_state():
    sent: list[dict] = []
    backend = Scripted(provisional=[_words("嗯"), _words("好")], final=[])
    session = _session(backend, sent)

    async def scenario():
        await run_rolling(session, 1, 300, 400)
        session.utterance_discarded()                     # too short: index 1 is reused
        await run_rolling(session, 1, 300, 400)

    asyncio.run(scenario())
    assert [e["text"] for e in sent] == ["嗯", "好"]
    assert len(backend.calls) == 2


def test_provisional_decode_skips_when_gpu_slots_are_busy():
    backend = FakeWhisper(delay_s=0.2)
    svc = make_service(backend, max_concurrent=1)
    pcm = np.zeros(16000, dtype=np.int16)

    async def scenario():
        final = asyncio.create_task(svc.transcribe(pcm, "zh"))
        await asyncio.sleep(0.05)
        skipped = await svc.transcribe_provisional(pcm, "zh")
        await final
        ran = await svc.transcribe_provisional(pcm, "zh")
        return skipped, ran

    skipped, ran = asyncio.run(scenario())
    assert skipped is None and svc.skipped_provisional == 1
    assert isinstance(ran, Transcript)
    assert [c["beam_size"] for c in backend.calls] == [5, 1]
    assert backend.calls[0]["dtype"] == "float32"
    # both bounded by the audio length; only the final decode may re-decode at higher temperature
    assert [c["max_new_tokens"] for c in backend.calls] == [32, 32]
    assert [c["fallback"] for c in backend.calls] == [True, False]


def test_final_decodes_are_bounded_by_the_semaphore():
    backend = FakeWhisper(delay_s=0.15)
    svc = make_service(backend, max_concurrent=1)
    pcm = np.zeros(1600, dtype=np.int16)

    async def scenario():
        t0 = time.monotonic()
        await asyncio.gather(svc.transcribe(pcm, "zh"), svc.transcribe(pcm, "zh"))
        return time.monotonic() - t0

    assert asyncio.run(scenario()) >= 0.3


# ---------------------------------------------------------------- over the WebSocket


def _stream(client, config: dict, audio: np.ndarray, pace_s: float = 0.0) -> list[dict]:
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client)}") as ws:
        ws.send_json(config)
        ready = ws.receive_json()
        while ready["type"] == "STATE":
            ready = ws.receive_json()
        step = 16 * 100
        for off in range(0, len(audio), step):
            ws.send_bytes(audio[off:off + step].astype("<i2").tobytes())
            if pace_s:
                time.sleep(pace_s)
        ws.send_json({"type": "END"})
        events, code = collect(ws)
    assert ready["type"] == "READY" and code == 1000, (ready, events)
    return [ready, *events]


def test_stream_sends_provisional_partial_and_final(make_client):
    stt = FakeWhisper()
    client = make_client(stt=stt)
    events = _stream(client, PCM_CONFIG, np.concatenate([speech(1500), silence(500)]), pace_s=0.03)
    kinds = types(events)
    assert events[0]["stt"]["language"] == "zh-TW" and events[0]["stt"]["streaming"] is True
    assert "TRANSCRIPT_PROVISIONAL" in kinds
    final_at = kinds.index("TRANSCRIPT_FINAL")
    assert max(i for i, k in enumerate(kinds) if k == "TRANSCRIPT_PROVISIONAL") < final_at
    assert kinds.index("UTTERANCE") < kinds.index("TRANSCRIPT_PARTIAL") < final_at < kinds.index("ENDED")
    final = events[final_at]
    assert final["text"] == "你好嗎" and final["utterance"] == 1 and final["language"] == "zh"
    assert final["duration_ms"] > 1400 and final["first_word_ms"] is not None
    partials = [e for e in events if e["type"] == "TRANSCRIPT_PARTIAL"]
    assert [p["text"] for p in partials] == ["你", "好", "嗎"] and [p["word_index"] for p in partials] == [0, 1, 2]
    assert stt.calls[-1]["beam_size"] == 5 and stt.calls[-1]["language"] == "zh"
    metrics = client.get("/metrics").text
    assert "rt_stt_finalize_latency_seconds_count 1.0" in metrics
    latency_sum = float(re.search(r"rt_stt_finalize_latency_seconds_sum (\S+)", metrics).group(1))
    assert latency_sum > 0                                             # a real, non-zero latency was observed


def test_language_null_auto_detects_and_missing_uses_default(make_client):
    stt = FakeWhisper(language="ja")
    client = make_client(stt=stt, settings=make_settings(stt_streaming=False))
    audio = np.concatenate([speech(600), silence(400)])
    auto = _stream(client, {**PCM_CONFIG, "language": None}, audio)
    assert auto[0]["config"]["language"] is None
    assert stt.calls[-1]["language"] is None
    assert next(e for e in auto if e["type"] == "TRANSCRIPT_FINAL")["language"] == "ja"
    assert "TRANSCRIPT_PROVISIONAL" not in types(auto)              # RT_STT_STREAMING=false

    default = _stream(client, {k: v for k, v in PCM_CONFIG.items() if k != "language"}, audio)
    assert default[0]["config"]["language"] == "zh-TW"
    assert stt.calls[-1]["language"] == "zh"


def test_rt_stt_language_auto_makes_detection_the_default(make_client):
    stt = FakeWhisper()
    client = make_client(stt=stt, settings=make_settings(stt_language="auto", stt_streaming=False))
    events = _stream(client, {k: v for k, v in PCM_CONFIG.items() if k != "language"},
                     np.concatenate([speech(600), silence(400)]))
    assert events[0]["config"]["language"] is None and stt.calls[-1]["language"] is None


def test_stt_failure_warns_and_keeps_the_session(make_client):
    client = make_client(stt=FakeWhisper(fail=True), settings=make_settings(stt_streaming=False))
    events = _stream(client, PCM_CONFIG, np.concatenate([speech(600), silence(400), speech(600), silence(400)]))
    warnings = [e for e in events if e["type"] == "WARNING"]
    assert [w["code"] for w in warnings] == ["stt_failed", "stt_failed"]
    assert types(events)[-1] == "ENDED"
    assert "rt_stt_failed_total 2.0" in client.get("/metrics").text
    # a failed finalize() never completed -- no latency observation for it
    assert "rt_stt_finalize_latency_seconds_count 0.0" in client.get("/metrics").text


def test_without_stt_only_vad_events_are_sent(client):
    events = _stream(client, PCM_CONFIG, np.concatenate([speech(600), silence(400)]))
    assert events[0]["stt"] is None
    assert not [k for k in types(events) if k.startswith("TRANSCRIPT_")]


def test_status_reports_stt_model(make_client):
    assert make_client().get("/audio/status").json()["stt"] == {"enabled": False}
    body = make_client(stt=FakeWhisper()).get("/audio/status").json()["stt"]
    assert body["loaded"] is True and body["model"] == "fake-whisper" and body["max_concurrent"] == 1


def test_invalid_language_is_rejected(make_client):
    client = make_client(stt=FakeWhisper())
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client)}") as ws:
        ws.send_json({**PCM_CONFIG, "language": 42})
        events, code = collect(ws)
    assert code == 4400 and events[-1]["code"] == "invalid_config"


def test_utterance_dump_sink_still_runs_with_stt(make_client):
    received = []

    class Sink:
        async def handle(self, session_id, config, utterance):
            received.append(utterance.index)

    client = make_client(stt=FakeWhisper(), sink=Sink(), settings=make_settings(stt_streaming=False))
    _stream(client, PCM_CONFIG, np.concatenate([speech(600), silence(400)]))
    assert received == [1]
