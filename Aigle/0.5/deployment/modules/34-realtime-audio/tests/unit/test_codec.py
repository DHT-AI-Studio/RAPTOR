"""RA-8 / RTA-X11: G.711 tables, streaming resampling, decoders, encoders and CONFIG codecs."""
from __future__ import annotations

import warnings

import numpy as np
import pytest
from app.audio import make_decoder
from app.codec import (
    ALAW_DECODE,
    ALAW_ENCODE,
    ULAW_DECODE,
    ULAW_ENCODE,
    EncodeError,
    G711Encoder,
    PcmEncoder,
    Resampler,
    g711_decode,
    g711_encode,
    make_encoder,
)
from app.protocol import ProtocolError, parse_config

from .conftest import bare, issue_ticket, make_settings
from .test_stream import collect, types


def tone(rate: int, ms: int, freq: float = 440.0, amp: float = 0.4) -> np.ndarray:
    t = np.arange(rate * ms // 1000) / rate
    return (np.sin(2 * np.pi * freq * t) * amp * 32767).astype(np.int16)


def snr_db(reference: np.ndarray, test: np.ndarray, max_lag: int = 64) -> float:
    """Best SNR over small lags (resamplers delay the signal a little); edges excluded."""
    ref = reference.astype(np.float64)
    out = test.astype(np.float64)
    best = -np.inf
    for lag in range(max_lag + 1):
        n = min(len(ref), len(out) - lag) - 2 * max_lag
        if n <= 0:
            break
        r, o = ref[max_lag:max_lag + n], out[lag + max_lag:lag + max_lag + n]
        noise = np.sum((r - o) ** 2)
        best = max(best, 10 * np.log10(np.sum(r ** 2) / max(noise, 1e-9)))
    return best


def dominant_hz(pcm: np.ndarray, rate: int) -> float:
    spectrum = np.abs(np.fft.rfft(pcm.astype(np.float64) * np.hanning(len(pcm))))
    return float(np.fft.rfftfreq(len(pcm), 1 / rate)[np.argmax(spectrum)])


# ---------------------------------------------------------------- G.711


def test_g711_reference_vectors():
    # values from CPython's audioop (Python 3.12), so they hold where audioop no longer exists
    assert ULAW_DECODE[[0x00, 0x7F, 0x80, 0xFF, 0x0F, 0x70]].tolist() == [-32124, 0, 32124, 0, -16764, -120]
    assert ALAW_DECODE[[0xD5, 0x55, 0x2A, 0xAA, 0x80, 0x00]].tolist() == [8, -8, -32256, 32256, 5504, -5504]
    enc = lambda table, x: int(table[x + 32768])
    assert [enc(ULAW_ENCODE, x) for x in (0, -1, 32767, -32768, 100, -100)] == [0xFF, 0x7E, 0x80, 0x00, 0xF2, 0x72]
    assert [enc(ALAW_ENCODE, x) for x in (0, -1, 32767, -32768, 100, -100)] == [0xD5, 0x55, 0xAA, 0x2A, 0xD3, 0x53]


def test_g711_matches_audioop_bit_for_bit():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        audioop = pytest.importorskip("audioop")          # removed in Python 3.13
    every = np.arange(-32768, 32768, dtype=np.int16).astype("<i2").tobytes()
    codes = bytes(range(256))
    assert g711_encode(np.arange(-32768, 32768), "pcmu") == audioop.lin2ulaw(every, 2)
    assert g711_encode(np.arange(-32768, 32768), "pcma") == audioop.lin2alaw(every, 2)
    assert g711_decode(codes, "pcmu").astype("<i2").tobytes() == audioop.ulaw2lin(codes, 2)
    assert g711_decode(codes, "pcma").astype("<i2").tobytes() == audioop.alaw2lin(codes, 2)


@pytest.mark.parametrize("law", ["pcmu", "pcma"])
def test_g711_roundtrip_quantisation_noise(law):
    pcm = tone(8000, 1000)
    back = g711_decode(g711_encode(pcm, law), law)
    assert snr_db(pcm, back, max_lag=0) > 33                 # G.711 is ~38 dB for a full-scale tone


# ---------------------------------------------------------------- resampling and decoders


def test_streaming_resampler_matches_one_shot():
    pytest.importorskip("av")
    pcm = tone(8000, 1000)
    framed = Resampler(8000, 16000)
    out = np.concatenate([framed(pcm[i:i + 160]) for i in range(0, len(pcm), 160)] + [framed.flush()])
    whole = Resampler(8000, 16000)
    ref = np.concatenate([whole(pcm), whole.flush()])
    assert len(out) == len(ref) == 16000
    assert np.max(np.abs(out.astype(np.int32) - ref)) <= 1                     # no seams at frame edges
    assert Resampler(16000, 16000)(pcm) is not None and Resampler(16000, 16000).flush().size == 0


@pytest.mark.parametrize("law", ["pcmu", "pcma"])
def test_g711_8k_wav_to_16k_pcm(law):
    pytest.importorskip("av")
    dec = make_decoder(law, 8000)
    encoded = g711_encode(tone(8000, 1000), law)
    out = np.concatenate([dec.decode(encoded[i:i + 160]) for i in range(0, len(encoded), 160)] + [dec.flush()])
    assert out.dtype == np.int16 and len(out) == 16000
    assert snr_db(tone(16000, 1000), out) > 25
    assert abs(dominant_hz(out, 16000) - 440) < 5


@pytest.mark.parametrize("rate", [8000, 22050, 44100, 48000])
def test_pcm_any_rate_to_16k(rate):
    pytest.importorskip("av")
    dec = make_decoder("pcm", rate)
    data = tone(rate, 1000).astype("<i2").tobytes()
    step = rate // 50 * 2 + 1                                       # odd byte counts across frames
    out = np.concatenate([dec.decode(data[i:i + step]) for i in range(0, len(data), step)] + [dec.flush()])
    assert abs(len(out) - 16000) <= 2
    assert snr_db(tone(16000, 1000), out) > 25


def test_opus_48k_to_16k():
    av = pytest.importorskip("av")
    enc = av.CodecContext.create("libopus", "w")
    enc.sample_rate, enc.layout, enc.format = 48000, "mono", "s16"
    pcm = tone(48000, 1000)
    packets = []
    for off in range(0, len(pcm), 960):
        frame = av.AudioFrame.from_ndarray(pcm[off:off + 960].reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate, frame.pts = 48000, off
        packets += [bytes(p) for p in enc.encode(frame)]
    dec = make_decoder("opus", 48000)
    out = np.concatenate([dec.decode(p) for p in packets] + [dec.flush()])
    assert abs(len(out) - 16000) <= 640
    assert abs(dominant_hz(out, 16000) - 440) < 5


# ---------------------------------------------------------------- encoders (server -> client)


@pytest.mark.parametrize("law", ["pcmu", "pcma"])
def test_tts_24k_straight_to_8k_g711_frames(law):
    pytest.importorskip("av")
    enc = make_encoder(law, 8000)
    assert isinstance(enc, G711Encoder) and enc.info() == {"codec": law, "sample_rate": 8000, "frame_ms": 20}
    tts = tone(24000, 1010)                                         # not a whole number of frames
    frames = enc.encode(tts[:12000], 24000) + enc.encode(tts[12000:], 24000) + enc.flush()
    assert {len(f) for f in frames} == {160}                         # 20 ms frames only
    assert len(frames) == 51                                         # 1010 ms padded to 1020
    out = g711_decode(b"".join(frames), law)
    assert snr_db(tone(8000, 1010), out) > 25
    assert enc.flush() == []


def test_pcm_encoder_resamples_to_16k_in_4096_byte_chunks():
    pytest.importorskip("av")
    enc = PcmEncoder()
    chunks = enc.encode(tone(24000, 500), 24000) + enc.flush()
    assert max(len(c) for c in chunks) <= 4096
    out = np.frombuffer(b"".join(chunks), dtype="<i2")
    assert abs(len(out) - 8000) <= 2 and abs(dominant_hz(out, 16000) - 440) < 5


def test_opus_encoder_packets_decode():
    pytest.importorskip("av")
    enc = make_encoder("opus", 48000)
    packets = enc.encode(tone(24000, 1000), 24000) + enc.flush()
    assert packets and all(packets)
    dec = make_decoder("opus", 48000)
    out = np.concatenate([dec.decode(p) for p in packets] + [dec.flush()])
    assert abs(dominant_hz(out, 16000) - 440) < 5


def test_opus_preserves_packet_boundaries_when_encode_returns_multiple_packets():
    enc = make_encoder("opus", 48000)

    class BufferedEncoder:
        def __init__(self):
            self.batches = iter([[], [b"packet-1", b"packet-2"], [b"packet-3"]])

        def encode(self, frame):
            return next(self.batches)

    enc._ctx = BufferedEncoder()
    assert enc.encode(tone(48000, 60), 48000) == [b"packet-1", b"packet-2", b"packet-3"]


def test_output_encoder_rejects_unknown_codec():
    with pytest.raises(EncodeError, match="unsupported codec"):
        make_encoder("opuz", 48000)


# ---------------------------------------------------------------- CONFIG


def _cfg(**body) -> dict:
    import json

    return {"text": json.dumps({"type": "CONFIG", "language": "zh-TW", **body})}


def test_config_codecs_and_rates():
    assert parse_config(_cfg(codec="pcma", sample_rate=8000)).codec == "pcma"
    assert parse_config(_cfg(codec="pcmu", sample_rate=8000)).codec == "pcmu"
    assert parse_config(_cfg(codec="pcm", sample_rate=44100)).sample_rate == 44100
    for body, code in [
        ({"codec": "g729", "sample_rate": 8000}, "UNSUPPORTED_CODEC"),
        ({"codec": "pcma", "sample_rate": 16000}, "invalid_config"),
        ({"codec": "pcm", "sample_rate": 96000}, "invalid_config"),
        ({"sample_rate": 16000}, "invalid_config"),
    ]:
        with pytest.raises(ProtocolError) as exc:
            parse_config(_cfg(**body))
        assert exc.value.code == code


def test_config_without_resampling_accepts_16k_pcm_only():
    assert parse_config(_cfg(codec="pcm", sample_rate=16000), resample=False).codec == "pcm"
    for body in ({"codec": "pcm", "sample_rate": 8000}, {"codec": "opus", "sample_rate": 48000},
                 {"codec": "pcma", "sample_rate": 8000}):
        with pytest.raises(ProtocolError) as exc:
            parse_config(_cfg(**body), resample=False)
        assert exc.value.code == "UNSUPPORTED_CODEC"


def test_unsupported_codec_closes_the_stream(make_client):
    client = make_client(settings=make_settings(codec_resample=False))
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client)}") as ws:
        ws.send_json({"type": "CONFIG", "codec": "pcma", "sample_rate": 8000, "language": "zh-TW"})
        events, code = collect(ws)
    assert code == 4400 and bare(events[-1]) == {"type": "ERROR", "code": "UNSUPPORTED_CODEC",
                                           "message": "this server only accepts 16 kHz pcm (RT_CODEC_RESAMPLE=false)"}


@pytest.mark.parametrize("law", ["pcmu", "pcma"])
def test_g711_telephony_stream_is_segmented(client, law):
    pytest.importorskip("av")
    audio = np.concatenate([tone(8000, 700, amp=0.3), np.zeros(4000, dtype=np.int16)])
    encoded = g711_encode(audio, law)
    with client.websocket_connect(f"/audio/stream?token={issue_ticket(client)}") as ws:
        ws.send_json({"type": "CONFIG", "codec": law, "sample_rate": 8000, "language": "zh-TW"})
        for i in range(0, len(encoded), 160):                       # 20 ms RTP-sized frames
            ws.send_bytes(encoded[i:i + 160])
        ws.send_json({"type": "END"})
        events, code = collect(ws)
    assert code == 1000
    utt = next(e for e in events if e["type"] == "UTTERANCE")
    assert 600 <= utt["speech_ms"] <= 800
    assert types(events)[-1] == "ENDED" and events[-1]["audio_ms"] >= 1190
