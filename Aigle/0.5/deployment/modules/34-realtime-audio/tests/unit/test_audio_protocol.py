"""Frame decoding and CONFIG / END parsing."""
from __future__ import annotations

import numpy as np
import pytest
from app.audio import DecodeError, PcmDecoder, make_decoder
from app.protocol import ProtocolError, is_end, parse_config


def test_pcm_decoder_carries_odd_byte():
    dec = PcmDecoder()
    samples = np.array([1, -2, 300, -32768, 32767], dtype="<i2").tobytes()
    out = np.concatenate([dec.decode(samples[:3]), dec.decode(samples[3:7]), dec.decode(samples[7:])])
    assert out.tolist() == [1, -2, 300, -32768, 32767]


def test_opus_decoder_roundtrip_and_corrupt_packet():
    av = pytest.importorskip("av")
    enc = av.CodecContext.create("libopus", "w")
    enc.sample_rate, enc.layout, enc.format = 16000, "mono", "s16"
    t = np.arange(16000) / 16000
    pcm = (np.sin(2 * np.pi * 300 * t) * 8000).astype(np.int16)
    packets = []
    for off in range(0, len(pcm), 320):
        frame = av.AudioFrame.from_ndarray(pcm[off:off + 320].reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate, frame.pts = 16000, off
        packets += [bytes(p) for p in enc.encode(frame)]
    dec = make_decoder("opus", 16000)
    out = np.concatenate([dec.decode(p) for p in packets])
    assert out.dtype == np.int16 and abs(len(out) - 16000) <= 640
    assert np.sqrt(np.mean(out.astype(np.float64) ** 2)) > 2000
    with pytest.raises(DecodeError):
        dec.decode(b"\xff\xff\xff")


def test_make_decoder_rejects_unknown_codec():
    with pytest.raises(DecodeError):
        make_decoder("mp3", 16000)


def test_parse_config():
    cfg = parse_config({"text": '{"type":"CONFIG","codec":"opus","sample_rate":48000,"language":"zh-Hant-TW"}'})
    assert (cfg.codec, cfg.sample_rate, cfg.language) == ("opus", 48000, "zh-Hant-TW")
    for text, code in [
        ('{"codec":"opus","sample_rate":44100,"language":"en"}', "invalid_config"),
        ('{"codec":"pcm","sample_rate":true,"language":"en"}', "invalid_config"),
        ('["CONFIG"]', "invalid_config"),
        ('{"type":"AUDIO"}', "config_required"),
    ]:
        with pytest.raises(ProtocolError) as exc:
            parse_config({"text": text})
        assert exc.value.code == code
    with pytest.raises(ProtocolError):
        parse_config({"bytes": b"\x00"})


def test_is_end():
    assert is_end("END") and is_end(' {"type": "END"} ')
    assert not is_end('{"type":"CONFIG"}') and not is_end("end") and not is_end("[]")
