"""Decode audio frames into 16 kHz mono int16 PCM."""
from __future__ import annotations

import numpy as np

from app.codec import G711_RATE, TARGET_RATE, Resampler, g711_decode

PCM_RATE_RANGE = (8000, 48000)
OPUS_RATES = {8000, 12000, 16000, 24000, 48000}
G711_RATES = {G711_RATE}
CODECS = ("pcm", "opus", "pcmu", "pcma")


class DecodeError(Exception):
    pass


class _ResamplingDecoder:
    def flush(self) -> np.ndarray:
        return self._resample.flush()


class PcmDecoder(_ResamplingDecoder):
    def __init__(self, sample_rate: int = TARGET_RATE) -> None:
        self._carry = b""
        self._resample = Resampler(sample_rate, TARGET_RATE)

    def decode(self, data: bytes) -> np.ndarray:
        data = self._carry + data
        usable = len(data) - (len(data) % 2)
        self._carry = data[usable:]
        return self._resample(np.frombuffer(data[:usable], dtype="<i2").astype(np.int16, copy=False))

class G711Decoder(_ResamplingDecoder):
    def __init__(self, law: str) -> None:
        self._law = law
        self._resample = Resampler(G711_RATE, TARGET_RATE)

    def decode(self, data: bytes) -> np.ndarray:
        return self._resample(g711_decode(data, self._law))

class OpusDecoder:
    def __init__(self, sample_rate: int) -> None:
        import av  # PyAV ships libopus; imported lazily so PCM-only deployments/tests do not need it

        self._av = av
        self._ctx = av.CodecContext.create("libopus", "r")
        self._ctx.sample_rate = sample_rate
        self._ctx.layout = "mono"
        self._resampler = av.AudioResampler(format="s16", layout="mono", rate=TARGET_RATE)

    def decode(self, data: bytes) -> np.ndarray:
        try:
            frames = self._ctx.decode(self._av.Packet(data))
        except Exception as exc:  # av.error.* -- a corrupt packet
            raise DecodeError(f"invalid opus packet: {exc}") from exc
        out: list[np.ndarray] = []
        for frame in frames:
            for resampled in self._resampler.resample(frame):
                out.append(resampled.to_ndarray().reshape(-1).astype(np.int16, copy=False))
        return np.concatenate(out) if out else np.zeros(0, dtype=np.int16)

    def flush(self) -> np.ndarray:
        out = [f.to_ndarray().reshape(-1).astype(np.int16, copy=False) for f in self._resampler.resample(None)]
        return np.concatenate(out) if out else np.zeros(0, dtype=np.int16)


def make_decoder(codec: str, sample_rate: int):
    if codec == "pcm":
        return PcmDecoder(sample_rate)
    if codec == "opus":
        return OpusDecoder(sample_rate)
    if codec in ("pcmu", "pcma"):
        return G711Decoder(codec)
    raise DecodeError(f"unsupported codec {codec!r}")
