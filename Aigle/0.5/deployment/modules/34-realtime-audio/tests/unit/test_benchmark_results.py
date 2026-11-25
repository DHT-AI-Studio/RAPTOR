"""MOS exports merge to one rating per evaluator and clip; 8 kHz fixtures load as 16 kHz;
the capacity runner keeps what each call measured."""
import numpy as np
import soundfile as sf
from benchmarks.model_benchmark import load_audio
from benchmarks.mos_results import merge, summarize


def rating(system, condition, sentence, rater, score):
    rate, codec = (8000, "PCMA") if condition == "telephony" else (24000, None)
    return {"sample_id": f"{system}:{condition}:{sentence}", "clip_id": f"{system}-{condition}-{sentence}",
            "system": system, "condition": condition, "sentence_id": sentence, "evaluator_id": rater,
            "evaluator_region": "TW", "sample_rate": rate, "codec": codec, "score": score}


def test_overlapping_exports_keep_the_latest_rating_per_evaluator_and_clip():
    first = {"exported_at": "2026-10-28T10:00:00Z", "ratings": [rating("a", "telephony", "tts-01", "r1", 2)]}
    later = {"exported_at": "2026-10-28T11:00:00Z", "ratings": [rating("a", "telephony", "tts-01", "r1", 4),
                                                                rating("a", "telephony", "tts-01", "r2", 5)]}
    merged = merge([later, first])
    assert sorted((r["evaluator_id"], r["score"]) for r in merged) == [("r1", 4), ("r2", 5)]


def test_summary_reports_mos_per_condition_and_the_telephony_gate_per_system():
    ratings = [rating(system, condition, f"tts-{s:02d}", f"r{r}", score)
               for system, score in (("good", 4), ("poor", 3))
               for condition in ("wideband", "telephony")
               for s in range(10) for r in range(3)]
    report = summarize(ratings)
    assert report["evaluators"] == 3 and report["ratings"] == 120
    assert report["mos"]["good/telephony"] == {"mos": 4, "sd": 0, "ratings": 30}
    assert report["telephony_gate"]["good"]["status"] == "pass"
    assert report["telephony_gate"]["poor"]["status"] == "fail"
    assert len(report["telephony_gate"]["poor"]["per_sentence"]) == 10


def test_two_raters_per_sentence_leave_the_gate_pending():
    ratings = [rating("a", "telephony", f"tts-{s:02d}", f"r{r}", 5) for s in range(10) for r in range(2)]
    assert summarize(ratings)["telephony_gate"]["a"]["status"] == "pending"


def test_g711_fixture_is_upsampled_to_16khz(tmp_path):
    path = tmp_path / "phone.wav"
    sf.write(path, np.sin(np.arange(8000) * 2 * np.pi * 440 / 8000) * .3, 8000, subtype="ALAW")
    audio = load_audio(path)
    assert audio.dtype == np.float32 and len(audio) == 16000 and np.std(audio) > .1


def test_capacity_run_keeps_stage_latencies_counters_and_failure_details():
    import asyncio
    from contextlib import asynccontextmanager

    from benchmarks.capacity import run

    class Session:
        def __init__(self, call_id):
            self.call_id = call_id

        async def turn(self, sample):
            if self.call_id == 2:
                raise RuntimeError("call 2: speech start 0, utterances 0, finals 0")
            return {"final": 400.0, "first_audio": 300.0, "no_provisional": 1}

    @asynccontextmanager
    async def driver(sample, call_id, config):
        yield Session(call_id)

    config = {"slos_ms": {"final": 1500, "first_audio": 500}, "gpu_count": 1, "reference_gpu": "test"}
    result = asyncio.run(run(driver, [{"id": "a"}], 3, 2, config))
    assert result["stage_latency_ms"] == {"final": [400.0] * 4, "first_audio": [300.0] * 4}
    assert result["counters"] == {"no_provisional": 4}
    assert result["failures"] == 1 and result["errors"][0]["detail"].startswith("call 2:")
