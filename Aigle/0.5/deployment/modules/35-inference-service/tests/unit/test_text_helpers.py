"""Pure post-processing helpers used by the text tasks."""
from __future__ import annotations

from app.services.text_inference import (
    count_words,
    parse_detected_language,
    split_chunks,
    split_sentences,
    sum_usage,
    truncate_words,
)


def test_count_words_mixes_cjk_and_latin():
    assert count_words("今天天氣很好") == 6
    assert count_words("GPT 模型 is great!") == 5
    assert count_words("，。！") == 0


def test_truncate_words_keeps_punctuation_inside():
    assert truncate_words("a, b, c, d", 2) == "a, b"
    assert truncate_words("一二三四", 2) == "一二"
    assert truncate_words("short", 5) == "short"


def test_split_sentences_does_not_break_decimals():
    assert split_sentences("Version 3.5 is out. 很好。Yes!") == ["Version 3.5 is out.", "很好。", "Yes!"]


def test_parse_detected_language():
    assert parse_detected_language("DETECTED_LANGUAGE: zh-TW\n\n你好") == ("zh-TW", "你好")
    assert parse_detected_language("`DETECTED_LANGUAGE：en`\nHello") == ("en", "Hello")
    assert parse_detected_language("Hello\nDETECTED_LANGUAGE: en") == (None, "Hello\nDETECTED_LANGUAGE: en")
    assert parse_detected_language("Hello") == (None, "Hello")


def test_split_chunks_respects_limit_and_order():
    text = "\n\n".join(f"para {i} " + "x" * 300 for i in range(10))
    chunks = split_chunks(text, 1000)
    assert len(chunks) > 1
    assert all(len(c) <= 1000 for c in chunks)
    assert "".join(chunks).replace("\n", "") == text.replace("\n", "")

    no_breaks = "y" * 2500
    assert [len(c) for c in split_chunks(no_breaks, 1000)] == [1000, 1000, 500]
    assert split_chunks("short", 1000) == ["short"]


def test_sum_usage():
    u = {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    assert sum_usage([u, u]) == {"prompt_tokens": 2, "completion_tokens": 4, "total_tokens": 6}
    assert sum_usage([u, None]) is None
