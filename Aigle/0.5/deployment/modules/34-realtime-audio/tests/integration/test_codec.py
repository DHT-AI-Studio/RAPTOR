"""RA-10 codec path: the fixtures (encoded by libavcodec, see tests/fixtures/make_fixtures.py) through
the decoders a session uses -- app.audio.make_decoder(codec, rate).decode() per frame, then
flush() -- must come out as 16 kHz mono int16 PCM matching the 16 kHz original."""
from __future__ import annotations

import av
import numpy as np
import pytest
from app.audio import make_decoder
from app.codec import TARGET_RATE

from .conftest import FIXTURES, frames, read_wav, read_wav_payload

pytestmark = pytest.mark.integration

WAVE_FORMAT_MULAW = 7
ENVELOPE_MS = 20


def envelope(pcm: np.ndarray, rate: int = TARGET_RATE) -> np.ndarray:
    n = rate * ENVELOPE_MS // 1000
    usable = len(pcm) - len(pcm) % n
    return np.sqrt(np.mean(pcm[:usable].astype(np.float64).reshape(-1, n) ** 2, axis=1))


def best_correlation(a: np.ndarray, b: np.ndarray, max_lag: int = 3) -> float:
    """Correlation of two loudness envelopes, allowing a codec delay of up to ``max_lag`` frames."""
    best = -1.0
    for lag in range(-max_lag, max_lag + 1):
        x, y = (a[lag:], b) if lag >= 0 else (a, b[-lag:])
        n = min(len(x), len(y))
        best = max(best, float(np.corrcoef(x[:n], y[:n])[0, 1]))
    return best


def level_db(pcm: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)) + 1e-9)


def decode_all(codec: str, rate: int, packets: list[bytes]) -> np.ndarray:
    decoder = make_decoder(codec, rate)
    out = [decoder.decode(p) for p in packets] + [decoder.flush()]
    return np.concatenate(out)


def assert_is_the_reference(decoded: np.ndarray, *, tolerance_ms: float, min_corr: float = 0.95) -> None:
    reference, rate = read_wav("zh_tw_5s.wav")
    assert rate == TARGET_RATE
    assert decoded.dtype == np.int16 and decoded.ndim == 1
    assert abs(len(decoded) - len(reference)) <= tolerance_ms * TARGET_RATE / 1000, \
        f"{len(decoded)} samples for {len(reference)} ({len(decoded) / TARGET_RATE:.3f} s)"
    corr = best_correlation(envelope(decoded), envelope(reference))
    assert corr >= min_corr, f"loudness envelope correlation {corr:.3f}"
    assert abs(level_db(decoded) - level_db(reference)) <= 3.0


def test_g711_pcmu_fixture_decodes_to_16k_pcm():
    data, fmt, rate = read_wav_payload("zh_tw_5s_pcmu.wav")
    assert (fmt, rate) == (WAVE_FORMAT_MULAW, 8000) and len(data) == 5 * 8000
    decoded = decode_all("pcmu", 8000, frames(data, 160))          # 20 ms telephony frames
    assert_is_the_reference(decoded, tolerance_ms=10)


def test_opus_48k_fixture_decodes_to_16k_pcm():
    with av.open(str(FIXTURES / "zh_tw_5s_opus48k.ogg")) as container:
        stream = container.streams.audio[0]
        assert stream.codec_context.name == "opus" and stream.sample_rate == 48000
        packets = [bytes(p) for p in container.demux(stream) if p.size]
    assert len(packets) >= 240                                       # 20 ms packets, as a client sends them
    decoded = decode_all("opus", 48000, packets)
    # raw packets: the decoder neither drops Opus pre-skip (6.5 ms) nor the last frame's padding
    assert_is_the_reference(decoded, tolerance_ms=40, min_corr=0.9)


def av_resample(pcm: np.ndarray, src: int, dst: int) -> np.ndarray:
    """libswresample (PyAV), independent of app.codec.Resampler."""
    resampler = av.AudioResampler(format="s16", layout="mono", rate=dst)
    frame = av.AudioFrame.from_ndarray(pcm.astype(np.int16).reshape(1, -1), format="s16", layout="mono")
    frame.sample_rate = src
    return np.concatenate([f.to_ndarray().reshape(-1) for f in [*resampler.resample(frame), *resampler.resample(None)]])


def av_encode(pcm: np.ndarray, rate: int, codec_name: str) -> bytes:
    """libavcodec's encoder, independent of app.codec."""
    codec = av.CodecContext.create(codec_name, "w")
    codec.sample_rate, codec.layout, codec.format = rate, "mono", "s16"
    frame = av.AudioFrame.from_ndarray(pcm.astype(np.int16).reshape(1, -1), format="s16", layout="mono")
    frame.sample_rate = rate
    return b"".join(bytes(p) for p in [*codec.encode(frame), *codec.encode(None)])


def test_g711_pcma_decodes_to_16k_pcm():
    reference, _ = read_wav("zh_tw_5s.wav")
    encoded = av_encode(av_resample(reference, TARGET_RATE, 8000), 8000, "pcm_alaw")
    decoded = decode_all("pcma", 8000, frames(encoded, 160))
    assert_is_the_reference(decoded, tolerance_ms=10)


@pytest.mark.parametrize("rate", [8000, 22050, 44100, 48000])
def test_pcm_at_any_rate_resamples_to_16k(rate):
    reference, _ = read_wav("zh_tw_5s.wav")
    pcm = av_resample(reference, TARGET_RATE, rate)
    decoded = decode_all("pcm", rate, frames(pcm.astype("<i2").tobytes(), rate // 50 * 2 + 1))   # odd sizes too
    assert_is_the_reference(decoded, tolerance_ms=10)
