"""Corpus-weighted edit rates, latency percentiles and evidence-only quality gates."""
from __future__ import annotations

import math
import unicodedata
from collections.abc import Sequence


def text_normalize(text: str) -> str:
    """NFKC/lowercase normalization, preserving letters/numbers and word spaces."""
    value = unicodedata.normalize("NFKC", text).lower()
    return "".join(c for c in value if c.isspace() or unicodedata.category(c)[0] in {"L", "N"})


def edit_distance(reference: Sequence, hypothesis: Sequence) -> int:
    """Levenshtein distance using one row of memory."""
    row = list(range(len(hypothesis) + 1))
    for index, expected in enumerate(reference, 1):
        next_row = [index]
        for column, actual in enumerate(hypothesis, 1):
            next_row.append(min(next_row[-1] + 1, row[column] + 1,
                                row[column - 1] + (expected != actual)))
        row = next_row
    return row[-1]


def percentile(values: Sequence[float], fraction: float) -> float:
    """Linear-interpolated percentile; refuse missing or non-finite evidence."""
    if not values or not 0 <= fraction <= 1 or not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError("percentile requires finite, nonnegative observations")
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def corpus_rates(samples: Sequence[dict]) -> dict:
    """Weight CER/WER by reference length, not by the number of clips."""
    chars = words = char_errors = word_errors = 0
    for sample in samples:
        ref, hyp = text_normalize(sample["reference"]), text_normalize(sample["hypothesis"])
        ref_chars = "".join(ref.split())
        hyp_chars = "".join(hyp.split())
        ref_words, hyp_words = sample["reference_words"], sample["hypothesis_words"]
        if not ref_chars or not ref_words:
            raise ValueError("reference transcript and word tokens must not be empty")
        chars += len(ref_chars)
        words += len(ref_words)
        char_errors += edit_distance(ref_chars, hyp_chars)
        word_errors += edit_distance(ref_words, hyp_words)
    if not chars or not words:
        raise ValueError("missing corpus observations")
    return {"cer": char_errors / chars, "wer": word_errors / words,
            "reference_chars": chars, "reference_words": words,
            "char_errors": char_errors, "word_errors": word_errors}
