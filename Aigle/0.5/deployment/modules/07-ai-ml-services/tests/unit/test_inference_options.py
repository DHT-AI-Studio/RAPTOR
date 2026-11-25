"""M07-FIX-02/03: shared text-generation option normalization, stop strings, usage."""
import pytest
from src.inference.exceptions import ConflictingParametersError, InvalidParameterError
from src.inference.options import (
    StopStringFilter,
    is_greedy,
    normalize_generation_options,
    truncate_at_stop,
    usage_from_metadata,
)

# ----- max_tokens / max_new_tokens / max_length -----

def test_max_new_tokens_is_synonym_of_max_tokens():
    assert normalize_generation_options({"max_new_tokens": 64}) == {"max_tokens": 64}
    assert normalize_generation_options({"max_tokens": 64, "max_new_tokens": 64}) == {"max_tokens": 64}


def test_differing_synonyms_conflict():
    with pytest.raises(ConflictingParametersError) as ei:
        normalize_generation_options({"max_tokens": 64, "max_new_tokens": 32})
    assert ei.value.code == "conflicting_parameters"


@pytest.mark.parametrize("value", [0, -1, "10", 1.5, True])
def test_max_tokens_must_be_positive_int(value):
    with pytest.raises(InvalidParameterError) as ei:
        normalize_generation_options({"max_tokens": value})
    assert ei.value.code == "invalid_parameter"


def test_integral_float_accepted():
    assert normalize_generation_options({"max_tokens": 16.0}) == {"max_tokens": 16}


def test_max_length_kept_with_legacy_meaning():
    assert normalize_generation_options({"max_length": 300, "temperature": 0.3}) == {
        "max_length": 300, "temperature": 0.3,
    }


@pytest.mark.parametrize("key", ["max_tokens", "max_new_tokens"])
def test_max_length_mixed_with_new_limit_conflicts(key):
    with pytest.raises(ConflictingParametersError):
        normalize_generation_options({key: 10, "max_length": 10})


def test_none_values_are_treated_as_absent():
    assert normalize_generation_options({"max_tokens": None, "stop": None, "temperature": None}) == {}


# ----- stop / stop_sequences -----

def test_stop_string_becomes_list_and_sequences_alias():
    assert normalize_generation_options({"stop": "\n"}) == {"stop": ["\n"]}
    assert normalize_generation_options({"stop_sequences": ["END", "。"]}) == {"stop": ["END", "。"]}


def test_stop_same_value_accepted_and_different_conflicts():
    assert normalize_generation_options({"stop": "A", "stop_sequences": ["A"]}) == {"stop": ["A"]}
    with pytest.raises(ConflictingParametersError):
        normalize_generation_options({"stop": ["A"], "stop_sequences": ["B"]})


def test_empty_stop_list_means_no_extra_stop():
    assert normalize_generation_options({"stop_sequences": []}) == {"stop": []}


@pytest.mark.parametrize("value", [[""], [1], {"a": 1}, 5])
def test_invalid_stop_rejected(value):
    with pytest.raises(InvalidParameterError):
        normalize_generation_options({"stop": value})


# ----- temperature / greedy -----

@pytest.mark.parametrize("t", [0, 0.0, 1, 2])
def test_temperature_range_accepted(t):
    assert normalize_generation_options({"temperature": t})["temperature"] == float(t)


@pytest.mark.parametrize("t", [-0.1, 2.01, "0.5", True])
def test_temperature_out_of_range_rejected(t):
    with pytest.raises(InvalidParameterError):
        normalize_generation_options({"temperature": t})


def test_temperature_zero_is_greedy_and_conflicts_with_do_sample_true():
    opts = normalize_generation_options({"temperature": 0})
    assert is_greedy(opts)
    with pytest.raises(ConflictingParametersError):
        normalize_generation_options({"temperature": 0, "do_sample": True})
    assert is_greedy(normalize_generation_options({"temperature": 0.7, "do_sample": False}))
    assert not is_greedy(normalize_generation_options({"temperature": 0.7}))


def test_runtime_specific_options_pass_through():
    opts = normalize_generation_options({"num_ctx": 4096, "think": False, "keep_alive": "5m", "repeat_penalty": 1.1})
    assert opts == {"num_ctx": 4096, "think": False, "keep_alive": "5m", "repeat_penalty": 1.1}


def test_input_not_mutated():
    src = {"max_new_tokens": 5, "stop_sequences": "x"}
    normalize_generation_options(src)
    assert src == {"max_new_tokens": 5, "stop_sequences": "x"}


# ----- stop truncation -----

def test_truncate_at_earliest_stop():
    assert truncate_at_stop("hello END world ###", ["###", "END"]) == ("hello ", True)
    assert truncate_at_stop("你好。再見", ["。"]) == ("你好", True)
    assert truncate_at_stop("no stop here", ["END"]) == ("no stop here", False)
    assert truncate_at_stop("END at start", ["END"]) == ("", True)


def _run_filter(stops, chunks):
    f = StopStringFilter(stops)
    out = [f.feed(c) for c in chunks]
    out.append(f.flush())
    return "".join(out), out, f.stopped


def test_filter_holds_back_stop_prefix_across_chunks():
    text, pieces, stopped = _run_filter(["END"], ["abc E", "N", "D tail"])
    assert text == "abc " and stopped
    # the "E" / "EN" prefix must never be emitted before we know it is a stop string
    assert all("E" not in p for p in pieces)


def test_filter_releases_false_prefix():
    text, _, stopped = _run_filter(["END"], ["abc E", "X", " more"])
    assert text == "abc EX more" and not stopped


def test_filter_multichar_chinese_and_multiple_stops():
    text, _, stopped = _run_filter(["。", "\n\n"], ["第一句", "話\n", "\n第二"])
    assert text == "第一句話" and stopped


def test_filter_stop_at_chunk_boundary_and_flush_without_stop():
    assert _run_filter(["</s>"], ["x</", "s>", "y"])[0] == "x"
    assert _run_filter(["</s>"], ["x<", "/"])[0] == "x</"  # stream ended: held text is flushed
    assert _run_filter([], ["a", "b"])[0] == "ab"


# ----- usage -----

def test_usage_requires_real_counts():
    assert usage_from_metadata({"prompt_tokens": 10, "completion_tokens": 5}) == {
        "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
    }
    assert usage_from_metadata({"prompt_eval_count": 10, "eval_count": 5}) is None
    assert usage_from_metadata({"prompt_tokens": 10}) is None
    assert usage_from_metadata({"prompt_tokens": True, "completion_tokens": 1}) is None
    assert usage_from_metadata(None) is None
