"""RA-2: Segmenter timing rules, re-windowing and the model pool (no torch needed)."""
from __future__ import annotations

import numpy as np
from app.vad import (
    WINDOW_SAMPLES,
    Discarded,
    Segmenter,
    SpeechStart,
    Utterance,
    VadModelPool,
    VadParams,
)

from .conftest import AmplitudeVad


class ScriptedProb:
    """Returns probabilities from a per-window script (1 speech, 0 silence)."""

    def __init__(self, script: list[float]) -> None:
        self.script = list(script)

    def __call__(self, _window) -> float:
        return self.script.pop(0) if self.script else 0.0


def windows(n: int) -> np.ndarray:
    return np.zeros(n * WINDOW_SAMPLES, dtype=np.int16)


DEFAULT_VAD_PARAMS = VadParams()


def run(script: list[float], params: VadParams = DEFAULT_VAD_PARAMS, chunk: int = 480, flush: bool = False):
    seg = Segmenter(ScriptedProb(script), params)
    audio = windows(len(script))
    events = []
    for off in range(0, len(audio), chunk):
        events += seg.feed(audio[off:off + chunk])
    if flush:
        events += seg.flush()
    return seg, events


def of(events, kind):
    return [e for e in events if isinstance(e, kind)]


def test_rewindows_transport_chunks_without_losing_samples():
    seg, _ = run([0.0] * 20, chunk=480)       # 30 ms chunks -> 32 ms windows
    assert seg.processed_ms == 20 * 32
    seg = Segmenter(ScriptedProb([]), VadParams())
    seg.feed(np.zeros(1000, dtype=np.int16))
    assert seg.processed_ms == 32              # 1000 = 512 + 488 pending


def test_silence_of_300ms_ends_utterance():
    # 20 speech windows (640 ms), then 10 silent windows (320 ms >= 300 ms)
    _, events = run([1] * 20 + [0] * 10 + [0] * 5, VadParams(speech_pad_ms=0))
    utt = of(events, Utterance)
    assert len(utt) == 1 and utt[0].reason == "silence"
    assert utt[0].start_ms == 0 and utt[0].end_ms == 640 and utt[0].speech_ms == 640
    assert len(utt[0].pcm) == 20 * WINDOW_SAMPLES
    assert of(events, SpeechStart)[0].start_ms == 0


def test_shorter_gap_does_not_split():
    # 9 silent windows = 288 ms < 300 ms
    _, events = run([1] * 10 + [0] * 9 + [1] * 10 + [0] * 12, VadParams(speech_pad_ms=0))
    utt = of(events, Utterance)
    assert len(utt) == 1 and utt[0].speech_ms == 29 * 32


def test_threshold_is_inclusive_and_configurable():
    _, events = run([0.5] * 10 + [0.49] * 10, VadParams(speech_pad_ms=0))
    assert of(events, Utterance)[0].speech_ms == 320
    _, events = run([0.5] * 10 + [0] * 10, VadParams(threshold=0.6))
    assert not of(events, SpeechStart)


def test_min_speech_200ms():
    # 6 windows = 192 ms -> discarded; 7 windows = 224 ms -> kept
    _, events = run([1] * 6 + [0] * 12, VadParams(speech_pad_ms=0))
    assert not of(events, Utterance)
    assert of(events, Discarded)[0].speech_ms == 192
    _, events = run([1] * 7 + [0] * 12, VadParams(speech_pad_ms=0))
    assert of(events, Utterance)[0].speech_ms == 224


def test_max_utterance_flushes_and_continues():
    # 12 s of continuous speech with a 10 s cap
    n = int(12000 / 32)
    _, events = run([1] * n, VadParams(speech_pad_ms=0), flush=True)
    utt = of(events, Utterance)
    assert [u.reason for u in utt] == ["max_length", "end"]
    assert 9968 <= utt[0].duration_ms <= 10000
    assert utt[1].start_ms == utt[0].end_ms
    assert sum(len(u.pcm) for u in utt) == n * WINDOW_SAMPLES
    assert [u.index for u in utt] == [1, 2]


def test_flush_on_end():
    seg, events = run([0] * 5 + [1] * 10)
    assert not of(events, Utterance) and seg.triggered
    tail = seg.flush()
    assert len(tail) == 1 and tail[0].reason == "end"
    assert seg.flush() == []
    _, events = run([1] * 3, flush=True)
    assert of(events, Discarded) and not of(events, Utterance)


def test_speech_padding_before_and_after():
    params = VadParams(speech_pad_ms=96)      # 3 windows
    _, events = run([0] * 10 + [1] * 10 + [0] * 12, params)
    utt = of(events, Utterance)[0]
    assert utt.start_ms == 10 * 32 - 96
    assert utt.end_ms == 20 * 32 + 96
    assert utt.speech_ms == 320
    assert len(utt.pcm) == (3 + 10 + 3) * WINDOW_SAMPLES


def test_second_utterance_keeps_its_own_preroll():
    params = VadParams(speech_pad_ms=64)
    _, events = run([1] * 10 + [0] * 10 + [1] * 10 + [0] * 10, params)
    first, second = of(events, Utterance)
    assert second.start_ms == 20 * 32 - 64
    assert second.index == 2 and first.end_ms <= second.start_ms


def test_amplitude_model_on_real_signal():
    from .conftest import silence, speech

    seg = Segmenter(AmplitudeVad(), VadParams())
    events = seg.feed(np.concatenate([silence(400), speech(800), silence(500)]))
    utt = of(events, Utterance)
    assert len(utt) == 1 and 780 <= utt[0].speech_ms <= 850


async def test_pool_preloads_resets_and_caps():
    AmplitudeVad.instances = []
    pool = VadModelPool(AmplitudeVad, size=2)
    await pool.start()
    assert len(AmplitudeVad.instances) == 2
    a, b, c = await pool.acquire(), await pool.acquire(), await pool.acquire()
    assert len({id(a), id(b), id(c)}) == 3          # third loaded on demand
    assert a.resets == b.resets == c.resets == 1
    for m in (a, b, c):
        pool.release(m)
    assert len(pool._free) == 2
