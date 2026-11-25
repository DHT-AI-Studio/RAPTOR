"""RA-2: silero-VAD speech boundary detection.

Frames vs. model windows
  Audio arrives in RT_VAD_CHUNK_MS transport chunks (default 30 ms = 480 samples at 16 kHz).
  Silero VAD v5+/v6 at 16 kHz only accepts 512-sample (32 ms) windows, so the Segmenter
  re-windows the stream; every timing rule below therefore has 32 ms resolution.

Rules (all in audio time, not wall-clock)
  * a window is speech when its probability >= RT_VAD_THRESHOLD (no hysteresis: the same
    threshold ends speech, unlike silero's VADIterator which uses threshold - 0.15)
  * an utterance ends after >= RT_VAD_MIN_SILENCE_MS of consecutive non-speech windows
  * utterances whose speech span is < RT_VAD_MIN_SPEECH_MS are discarded as noise
  * a buffered utterance reaching RT_MAX_UTTERANCE_S is emitted immediately (reason
    ``max_length``) and the buffer is cleared; speech that continues starts a new utterance
  * END flushes a pending utterance (reason ``end``) if it meets the minimum length
  * RT_VAD_SPEECH_PAD_MS of audio before the first / after the last speech window is kept
  * barge-in (RTA-X5): ``loud_ms`` is the run of consecutive windows with probability above
    RT_BARGE_IN_THRESHOLD (0.7), whether or not an utterance is open; the session decides what
    a long enough run means while an answer is playing

Output: 16 kHz mono int16 PCM per utterance (see Utterance.wav() for a WAV wrapper that
07's /v1/audio/transcriptions accepts).
"""
from __future__ import annotations

import asyncio
import io
import logging
import wave
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 512                  # silero-vad window at 16 kHz
WINDOW_MS = WINDOW_SAMPLES * 1000 / SAMPLE_RATE

ProbFn = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class VadParams:
    threshold: float = 0.5
    min_silence_ms: int = 300
    min_speech_ms: int = 200
    max_utterance_s: float = 10.0
    speech_pad_ms: int = 96
    barge_in_threshold: float = 0.7


@dataclass
class SpeechStart:
    start_ms: int


@dataclass
class Utterance:
    index: int
    pcm: np.ndarray = field(repr=False)
    start_ms: int
    end_ms: int
    speech_ms: int
    reason: str                       # silence | max_length | end

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def wav(self) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(self.pcm.astype("<i2").tobytes())
        return buf.getvalue()


@dataclass
class Discarded:
    start_ms: int
    end_ms: int
    speech_ms: int
    reason: str = "too_short"


Event = SpeechStart | Utterance | Discarded


def _ms(samples: int) -> int:
    return round(samples * 1000 / SAMPLE_RATE)


def _samples(ms: float) -> int:
    return round(ms * SAMPLE_RATE / 1000)


class Segmenter:
    """Per-session VAD state machine; ``prob_fn`` maps a float32 window in [-1, 1] to P(speech)."""

    def __init__(self, prob_fn: ProbFn, params: VadParams) -> None:
        self._prob = prob_fn
        self.p = params
        self._min_silence = _samples(params.min_silence_ms)
        self._min_speech = _samples(params.min_speech_ms)
        self._max_len = _samples(params.max_utterance_s * 1000)
        self._pad = _samples(params.speech_pad_ms)
        pad_windows = -(-self._pad // WINDOW_SAMPLES)
        self._pending = np.zeros(0, dtype=np.int16)
        self._pos = 0                                   # samples already windowed
        # (start sample, window) pairs kept before speech starts; maxlen 0 disables pre-roll
        self._preroll: deque[tuple[int, np.ndarray]] = deque(maxlen=pad_windows)
        self._buf: deque[np.ndarray] = deque()          # int16 windows of the current utterance
        self._triggered = False
        self._start = 0
        self._speech_start = 0
        self._last_speech_end = 0
        self._silence = 0
        self._count = 0
        self._loud = 0                                  # samples of the current run above barge_in_threshold

    @property
    def triggered(self) -> bool:
        return self._triggered

    def replace_model(self, prob_fn: ProbFn) -> None:
        """Resume with a fresh model while preserving buffered audio and boundaries."""
        self._prob = prob_fn

    @property
    def processed_ms(self) -> int:
        return _ms(self._pos)

    # Utterance in progress, for the rolling-window STT (RTA-X2).  Read these only between
    # feed() calls -- feed() runs in a worker thread.

    @property
    def next_index(self) -> int:
        """Index the utterance in progress will get if it is emitted."""
        return self._count + 1

    @property
    def speech_elapsed_ms(self) -> int:
        """Audio time since speech started (0 when no utterance is open)."""
        return _ms(self._pos - self._speech_start) if self._triggered else 0

    @property
    def loud_ms(self) -> int:
        """Audio time of the current run of windows above the barge-in threshold."""
        return _ms(self._loud)

    def snapshot(self) -> np.ndarray:
        """Copy of the audio buffered for the utterance in progress (with its pre-roll)."""
        return np.concatenate(list(self._buf)) if self._triggered and self._buf else np.zeros(0, dtype=np.int16)

    def feed(self, pcm: np.ndarray) -> list[Event]:
        data = np.concatenate([self._pending, pcm.astype(np.int16, copy=False)]) if self._pending.size else pcm
        n = (len(data) // WINDOW_SAMPLES) * WINDOW_SAMPLES
        self._pending = data[n:].copy()
        events: list[Event] = []
        for off in range(0, n, WINDOW_SAMPLES):
            events.extend(self._window(data[off:off + WINDOW_SAMPLES]))
        return events

    def flush(self) -> list[Event]:
        """END of stream: emit (or discard) the utterance in progress."""
        if not self._triggered:
            return []
        return [self._finalize("end")]

    # ----- internals -----

    def _window(self, win: np.ndarray) -> list[Event]:
        start, end = self._pos, self._pos + WINDOW_SAMPLES
        self._pos = end
        prob = float(self._prob(win.astype(np.float32) / 32768.0))
        speech = prob >= self.p.threshold
        self._loud = self._loud + WINDOW_SAMPLES if prob > self.p.barge_in_threshold else 0
        events: list[Event] = []
        if self._triggered and end - self._start > self._max_len:
            # This window would overflow the buffer: emit what we have and start over.
            events.append(self._finalize("max_length"))
        if not self._triggered:
            if not speech:
                if self._preroll.maxlen:
                    self._preroll.append((start, win))
                return events
            self._triggered = True
            self._buf = deque(w for _, w in self._preroll)
            self._start = self._preroll[0][0] if self._preroll else start
            self._preroll.clear()
            self._buf.append(win)
            self._speech_start, self._last_speech_end, self._silence = start, end, 0
            return events + [SpeechStart(start_ms=_ms(start))]

        self._buf.append(win)
        if speech:
            self._last_speech_end, self._silence = end, 0
        else:
            self._silence += WINDOW_SAMPLES
        if self._silence >= self._min_silence:
            events.append(self._finalize("silence"))
        return events

    def _finalize(self, reason: str) -> Event:
        audio = np.concatenate(list(self._buf)) if self._buf else np.zeros(0, dtype=np.int16)
        buffered_end = self._start + len(audio)
        if reason == "max_length":
            end = buffered_end
        else:
            end = min(self._last_speech_end + self._pad, buffered_end)
        speech = self._last_speech_end - self._speech_start
        tail = audio[end - self._start:]
        audio = audio[: end - self._start]

        # Trailing non-speech windows can serve as pre-roll for the next utterance.
        self._preroll.clear()
        if self._preroll.maxlen and len(tail) >= WINDOW_SAMPLES:
            n_tail = len(tail) // WINDOW_SAMPLES
            for i in range(max(0, n_tail - self._preroll.maxlen), n_tail):
                self._preroll.append((end + i * WINDOW_SAMPLES, tail[i * WINDOW_SAMPLES:(i + 1) * WINDOW_SAMPLES]))
        self._buf = deque()
        self._triggered = False
        self._silence = 0

        if speech < self._min_speech:
            return Discarded(start_ms=_ms(self._start), end_ms=_ms(end), speech_ms=_ms(speech))
        self._count += 1
        return Utterance(index=self._count, pcm=audio, start_ms=_ms(self._start), end_ms=_ms(end),
                         speech_ms=_ms(speech), reason=reason)


# ---------------------------------------------------------------- silero model


def silero_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("silero-vad")
    except PackageNotFoundError:
        return None


def load_silero():
    """Load one silero-VAD (TorchScript) instance from the pip package's bundled weights."""
    import torch
    from silero_vad import load_silero_vad

    torch.set_num_threads(1)
    model = load_silero_vad(onnx=False)
    model.eval()
    return model


class SileroProb:
    """Callable adapter: float32 window -> speech probability, keeping the model's RNN state."""

    def __init__(self, model) -> None:
        import torch

        self._torch = torch
        self.model = model

    def reset(self) -> None:
        self.model.reset_states()

    def __call__(self, window: np.ndarray) -> float:
        with self._torch.no_grad():
            return float(self.model(self._torch.from_numpy(window), SAMPLE_RATE).item())


class VadModelPool:
    """Silero keeps per-stream RNN state, so every session needs its own instance.
    ``size`` instances are loaded at startup (which also proves the model loads); more are
    loaded on demand and returned to the pool up to ``size``."""

    def __init__(self, loader: Callable[[], ProbFn], size: int) -> None:
        self._loader = loader
        self._size = max(1, size)
        self._free: list[ProbFn] = []

    async def start(self) -> None:
        for _ in range(self._size):
            self._free.append(await asyncio.to_thread(self._loader))
        logger.info("silero-VAD %s loaded (%d instances)", silero_version() or "", self._size)

    async def acquire(self) -> ProbFn:
        model = self._free.pop() if self._free else await asyncio.to_thread(self._loader)
        reset = getattr(model, "reset", None)
        if reset:
            reset()
        return model

    def release(self, model: ProbFn) -> None:
        if len(self._free) < self._size:
            self._free.append(model)


def silero_loader() -> ProbFn:
    return SileroProb(load_silero())
