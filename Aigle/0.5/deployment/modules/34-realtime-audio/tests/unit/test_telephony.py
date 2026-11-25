"""Codec round-trip and quality gates reject missing, duplicated or failed evidence."""
import copy

import numpy as np
import pytest
import soundfile as sf
from benchmarks.telephony import capacity_gate, degrade, evaluate, mos_gate


@pytest.mark.parametrize("codec,subtype", [("PCMU", "ULAW"), ("PCMA", "ALAW")])
def test_real_8khz_companded_wav(tmp_path, codec, subtype):
    source, output = tmp_path / "wide.wav", tmp_path / "phone.wav"
    tone = np.sin(np.arange(16000) * 2 * np.pi * 440 / 16000) * .3
    sf.write(source, tone, 16000)
    details = degrade(source, output, codec)
    info = sf.info(output)
    assert info.samplerate == 8000 and info.subtype == subtype
    decoded, _ = sf.read(output)
    assert len(decoded) == 8000 and np.std(decoded) > .1
    assert details["codec"] == codec
    assert details["source_sha256"] != details["audio_sha256"]


def ratings():
    return [{"sample_id": f"tts-{sample}", "evaluator_id": f"rater-{rater}",
             "evaluator_region": "TW", "score": 4, "codec": "PCMU", "sample_rate": 8000}
            for sample in range(10) for rater in range(3)]


def evidence():
    return {"narrowband": {"cer": .14, "samples": [
        {"sha256": f"{i:064x}", "consented": True, "codec": "PCMU", "sample_rate": 8000,
         "locale": "zh-TW", "speaker": f"speaker-{i % 2}", "code_switching": True}
        for i in range(20)]}, "wideband": {"cer": .1}, "ratings": ratings(),
        "encoded_tts_sample_ids": [f"tts-{i}" for i in range(10)],
        "rta_x8_slos_ms": {"stt": 100, "tts": 200},
        "capacity_runs": [{"mode": "external-brain", "concurrent_calls": 20,
            "gpu_count": 1, "failures": 0, "stage_latency_ms": {"stt": [50] * 20, "tts": [150] * 20}}]}


def test_gate_requires_all_three_types_of_evidence():
    assert evaluate(evidence())["status"] == "pass"
    assert evaluate({})["status"] == "incomplete_or_fail"
    for key in ["wideband", "ratings", "rta_x8_slos_ms", "capacity_runs"]:
        missing = evidence()
        missing.pop(key)
        assert evaluate(missing)["status"] != "pass"


def test_quality_failures_never_pass():
    bad = evidence()
    bad["narrowband"]["cer"] = .16
    assert evaluate(bad)["stt"]["status"] == "fail"
    for r in bad["ratings"]:
        r["score"] = 3
    assert evaluate(bad)["tts_mos"]["status"] == "fail"
    with pytest.raises(ValueError, match="duplicate"):
        mos_gate(ratings() + [copy.copy(ratings()[0])], {f"tts-{i}" for i in range(10)})


def test_pending_tts_evidence_never_crashes_even_with_unmatched_ratings():
    """A rating for a sample_id not yet in encoded_tts_sample_ids (an in-progress
    evidence set -- real timeline: ratings arrive before all 10 sentences are
    encoded) used to make mos_gate() raise before the <10 pending check ever ran."""
    partial = evidence()
    partial["encoded_tts_sample_ids"] = []
    result = evaluate(partial)
    assert result["tts_mos"] == {"status": "pending", "reason": "need ten encoded TTS sentences"}
    assert result["status"] != "pass"


def test_capacity_gpu_counts_require_measurements_not_linear_extrapolation():
    result = capacity_gate(evidence()["capacity_runs"], {"stt": 100, "tts": 200})
    assert result["measured_gpu_requirements"] == {"20": 1, "50": None, "100": None}
    failed = evidence()["capacity_runs"]
    failed[0]["failures"] = 1
    assert capacity_gate(failed, {"stt": 100})["status"] != "pass"
    failed[0]["failures"] = 0
    failed[0]["stage_latency_ms"]["stt"] = [200] * 20
    assert capacity_gate(failed, {"stt": 100})["status"] != "pass"
