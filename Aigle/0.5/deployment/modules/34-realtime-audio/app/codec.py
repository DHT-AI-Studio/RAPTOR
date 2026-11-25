"""G.711 tables, streaming resampling and output packet encoders."""
from __future__ import annotations

import numpy as np

TARGET_RATE = 16000
G711_RATE = 8000
FRAME_MS = 20
PCM_CHUNK_BYTES = 4096


class EncodeError(ValueError):
    pass

# ---------------------------------------------------------------- G.711 tables

_SEG_UEND = np.array([0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF], dtype=np.int32)
_SEG_AEND = np.array([0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF], dtype=np.int32)


def _ulaw_decode_table() -> np.ndarray:
    u = ~np.arange(256, dtype=np.int32) & 0xFF
    exponent = (u >> 4) & 0x07
    magnitude = ((((u & 0x0F) << 3) + 0x84) << exponent) - 0x84
    return np.where(u & 0x80, -magnitude, magnitude).astype(np.int16)


def _alaw_decode_table() -> np.ndarray:
    a = np.arange(256, dtype=np.int32) ^ 0x55
    seg = (a & 0x70) >> 4
    t = (a & 0x0F) << 4
    t = np.where(seg == 0, t + 8, t + 0x108)
    t = np.where(seg > 1, t << np.maximum(seg - 1, 0), t)
    return np.where(a & 0x80, t, -t).astype(np.int16)


def _ulaw_encode_table() -> np.ndarray:
    pcm = np.arange(-32768, 32768, dtype=np.int32) >> 2                  # 14-bit
    mask = np.where(pcm < 0, 0x7F, 0xFF)
    mag = np.minimum(np.abs(pcm), 8159) + (0x84 >> 2)
    seg = np.searchsorted(_SEG_UEND, mag, side="left")
    uval = (seg << 4) | ((mag >> (np.minimum(seg, 7) + 1)) & 0x0F)
    out = np.where(seg >= 8, 0x7F ^ mask, uval ^ mask)
    return out.astype(np.uint8)


def _alaw_encode_table() -> np.ndarray:
    pcm = np.arange(-32768, 32768, dtype=np.int32) >> 3                  # 13-bit
    mask = np.where(pcm >= 0, 0xD5, 0x55)
    mag = np.where(pcm >= 0, pcm, -pcm - 1)
    seg = np.searchsorted(_SEG_AEND, mag, side="left")
    quant = np.where(seg < 2, mag >> 1, mag >> np.minimum(seg, 7)) & 0x0F
    aval = (seg << 4) | quant
    out = np.where(seg >= 8, 0x7F ^ mask, aval ^ mask)
    return out.astype(np.uint8)


ULAW_DECODE = _ulaw_decode_table()
ALAW_DECODE = _alaw_decode_table()
ULAW_ENCODE = _ulaw_encode_table()
ALAW_ENCODE = _alaw_encode_table()
SILENCE = {"pcmu": 0xFF, "pcma": 0xD5}                                      # encoded 0


def g711_decode(data: bytes, law: str) -> np.ndarray:
    table = ULAW_DECODE if law == "pcmu" else ALAW_DECODE
    return table[np.frombuffer(data, dtype=np.uint8)]


def g711_encode(pcm: np.ndarray, law: str) -> bytes:
    table = ULAW_ENCODE if law == "pcmu" else ALAW_ENCODE
    return table[pcm.astype(np.int32) + 32768].tobytes()


# ---------------------------------------------------------------- resampling


class Resampler:
    """Streaming mono int16 resampler; identity when the rates match."""

    def __init__(self, src_rate: int, dst_rate: int) -> None:
        self.src_rate = src_rate
        self.dst_rate = dst_rate
        self._r = None
        if src_rate != dst_rate:
            import av  # PyAV ships libswresample; lazy so PCM-16k-only paths never need it

            self._av = av
            self._r = av.AudioResampler(format="s16", layout="mono", rate=dst_rate)

    def __call__(self, pcm: np.ndarray) -> np.ndarray:
        if self._r is None or not len(pcm):
            return pcm.astype(np.int16, copy=False)
        frame = self._av.AudioFrame.from_ndarray(pcm.astype(np.int16, copy=False).reshape(1, -1),
                                                 format="s16", layout="mono")
        frame.sample_rate = self.src_rate
        return self._collect(self._r.resample(frame))

    def flush(self) -> np.ndarray:
        if self._r is None:
            return np.zeros(0, dtype=np.int16)
        return self._collect(self._r.resample(None))

    @staticmethod
    def _collect(frames) -> np.ndarray:
        out = [f.to_ndarray().reshape(-1).astype(np.int16, copy=False) for f in frames]
        return np.concatenate(out) if out else np.zeros(0, dtype=np.int16)


# ---------------------------------------------------------------- encoders


class Encoder:
    """Turns PCM at any rate into the session's binary audio frames."""

    codec = "pcm"
    sample_rate = TARGET_RATE

    def __init__(self) -> None:
        self._resampler: Resampler | None = None

    def info(self) -> dict:
        return {"codec": self.codec, "sample_rate": self.sample_rate, "frame_ms": self.frame_ms}

    frame_ms: int | None = None

    def _resample(self, pcm: np.ndarray, rate: int) -> np.ndarray:
        if self._resampler is None or self._resampler.src_rate != rate:
            tail = self._resampler.flush() if self._resampler is not None else None
            self._resampler = Resampler(rate, self.sample_rate)
            if tail is not None and len(tail):
                return np.concatenate([tail, self._resampler(pcm)])
        return self._resampler(pcm)

    def encode(self, pcm: np.ndarray, rate: int) -> list[bytes]:
        return self._frames(self._resample(pcm, rate))

    def flush(self) -> list[bytes]:
        """End of a response: everything still buffered, the last frame padded if needed."""
        tail = self._resampler.flush() if self._resampler is not None else np.zeros(0, dtype=np.int16)
        self._resampler = None
        return self._frames(tail, final=True)

    def _frames(self, pcm: np.ndarray, final: bool = False) -> list[bytes]:
        raise NotImplementedError

    def duration_ms(self, frame: bytes) -> float:
        """Playback time of one frame returned by encode() / flush()."""
        return len(frame) * 1000 / (2 * self.sample_rate)


class PcmEncoder(Encoder):
    codec = "pcm"
    sample_rate = TARGET_RATE

    def _frames(self, pcm, final=False):
        data = pcm.astype("<i2").tobytes()
        return [data[i:i + PCM_CHUNK_BYTES] for i in range(0, len(data), PCM_CHUNK_BYTES)]


class _Framed(Encoder):
    """Encoders that need whole 20 ms frames: keeps the remainder until the next call."""

    frame_ms = FRAME_MS

    def __init__(self) -> None:
        super().__init__()
        self._pending = np.zeros(0, dtype=np.int16)

    @property
    def frame_samples(self) -> int:
        return self.sample_rate * FRAME_MS // 1000

    def _frames(self, pcm, final=False):
        data = np.concatenate([self._pending, pcm]) if self._pending.size else pcm
        n = self.frame_samples
        if final and len(data) % n:
            data = np.concatenate([data, np.zeros(n - len(data) % n, dtype=np.int16)])
        whole = len(data) - len(data) % n
        self._pending = data[whole:].copy()
        return [packet for i in range(0, whole, n) for packet in self._encode_frame(data[i:i + n])]

    def _encode_frame(self, frame: np.ndarray) -> list[bytes]:
        raise NotImplementedError

    def duration_ms(self, frame: bytes) -> float:
        return float(FRAME_MS)


class G711Encoder(_Framed):
    sample_rate = G711_RATE

    def __init__(self, law: str) -> None:
        super().__init__()
        self.codec = law

    def _encode_frame(self, frame):
        return [g711_encode(frame, self.codec)]


class OpusEncoder(_Framed):
    codec = "opus"

    def __init__(self, sample_rate: int) -> None:
        super().__init__()
        import av

        self._av = av
        self.sample_rate = sample_rate
        self._ctx = av.CodecContext.create("libopus", "w")
        self._ctx.sample_rate, self._ctx.layout, self._ctx.format = sample_rate, "mono", "s16"
        self._pts = 0

    def _encode_frame(self, frame):
        f = self._av.AudioFrame.from_ndarray(frame.reshape(1, -1), format="s16", layout="mono")
        f.sample_rate, f.pts = self.sample_rate, self._pts
        self._pts += len(frame)
        return [bytes(p) for p in self._ctx.encode(f)]


def make_encoder(codec: str, sample_rate: int) -> Encoder:
    if codec in ("pcmu", "pcma"):
        return G711Encoder(codec)
    if codec == "opus":
        return OpusEncoder(sample_rate)
    if codec == "pcm":
        return PcmEncoder()
    raise EncodeError(f"unsupported codec {codec!r}")
