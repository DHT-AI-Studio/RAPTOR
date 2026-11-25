"""Known edit distances and missing-evidence behavior, without model downloads."""
import subprocess
import threading
from types import SimpleNamespace

import pytest
from benchmarks.metrics import corpus_rates, edit_distance, percentile, text_normalize
from benchmarks.model_benchmark import gpu_peak
from benchmarks.stream_eval import load_samples, score


def test_stream_score_skips_empty_references_and_handles_an_empty_corpus(tmp_path):
    def result(reference):
        return {"reference": reference, "hypothesis": "你", "per_utterance": [], "responses": [],
                "first_provisional_ms": None, "corrections": 0}

    results = [result(""), result(None), result("，！"), result("你好")]
    summary = score(results)
    assert [r["cer"] for r in results] == [None, None, None, 0.5]
    assert (summary["cer"], summary["reference_chars"], summary["char_errors"]) == (0.5, 2, 1)
    assert score([])["cer"] is None
    assert score([result("")])["cer"] is None
    path = tmp_path / "metadata.json"
    path.write_text('[{"id":"empty","audio":"a.wav","transcript":""},{"id":"missing","audio":"b.wav"}]')
    assert [s["reference"] for s in load_samples(path)] == ["", ""]


def test_edit_distance_counts_insert_delete_substitution():
    assert edit_distance("今天下雨", "今天晴天") == 2
    assert edit_distance(["今天", "下雨"], ["今天", "下", "小雨"]) == 2
    assert edit_distance("", "abc") == 3


def test_corpus_rates_weight_by_reference_length():
    rates = corpus_rates([
        {"reference": "你好", "hypothesis": "你",
         "reference_words": ["你好"], "hypothesis_words": ["你"]},
        {"reference": "今天天氣很好", "hypothesis": "今天天氣很好",
         "reference_words": ["今天", "天氣", "很好"], "hypothesis_words": ["今天", "天氣", "很好"]},
    ])
    assert rates["cer"] == 1 / 8
    assert rates["wer"] == 1 / 4
    assert text_normalize("ＡＢＣ，早安！") == "abc早安"


def test_percentiles_and_missing_evidence():
    assert percentile([1, 2, 3, 4], .5) == 2.5
    assert percentile([1, 2, 3, 4], .95) == pytest.approx(3.85)
    with pytest.raises(ValueError):
        percentile([], .95)
    with pytest.raises(ValueError):
        corpus_rates([])


def test_vram_sampling_includes_baseline_and_peak(monkeypatch):
    observed = threading.Event()
    values = iter([100, 240, 130])
    def read(*args, **kwargs):
        value = next(values, 130)
        if value == 240:
            observed.set()
        return SimpleNamespace(stdout=str(value))
    monkeypatch.setattr("benchmarks.model_benchmark.subprocess.run", read)
    with gpu_peak(0) as report:
        assert observed.wait(1)
    assert report["device_baseline_mib"] == 100
    assert report["device_peak_mib"] == 240
    assert report["incremental_peak_mib"] == 140


def test_failed_vram_monitor_cannot_publish_a_valid_measurement(monkeypatch):
    observed = threading.Event()
    calls = 0
    def read(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 1:
            observed.set()
            raise subprocess.CalledProcessError(1, "nvidia-smi")
        return SimpleNamespace(stdout="100")
    monkeypatch.setattr("benchmarks.model_benchmark.subprocess.run", read)
    with pytest.raises(RuntimeError, match="measurement is invalid"), gpu_peak(0):
        assert observed.wait(1)
