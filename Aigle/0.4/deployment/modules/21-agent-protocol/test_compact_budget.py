"""
Unit tests for the token budget trimming logic in pipeline.py (MV-12).

Run from the 21-agent-protocol/ directory:
    python -m pytest test_compact_budget.py -v

Tests the trimming algorithm in isolation — no A2A agents or LLM needed.
"""
import json


# ── Extracted trimming logic (mirrors pipeline.py Step 5 exactly) ─────────────

def apply_budget_trim(top_chunks: list[dict], budget_tokens: int) -> list[dict]:
    """Return trimmed chunk list if over budget; otherwise return original."""
    chunk_tokens = sum(len(json.dumps(c, default=str)) // 4 for c in top_chunks)
    if chunk_tokens <= budget_tokens or len(top_chunks) <= 1:
        return top_chunks
    kept: list[dict] = []
    used = 0
    for chunk in top_chunks:
        t = len(json.dumps(chunk, default=str)) // 4
        if used + t > budget_tokens:
            break
        kept.append(chunk)
        used += t
    return kept if kept else top_chunks


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_no_trim_when_under_budget():
    chunks = [{"content": "short", "doc_id": "d1", "score": 0.9}]
    result = apply_budget_trim(chunks, budget_tokens=80000)
    assert result == chunks


def test_trims_chunks_when_over_budget():
    # Each chunk ~250 tokens (1000 chars); budget = 400 tokens
    chunks = [{"content": "x" * 1000, "doc_id": f"d{i}", "score": 0.9} for i in range(5)]
    result = apply_budget_trim(chunks, budget_tokens=400)
    assert len(result) < len(chunks)


def test_never_drops_all_chunks():
    # Even if first chunk exceeds budget, must keep at least 1
    chunks = [
        {"content": "x" * 400000, "doc_id": "big", "score": 1.0},
        {"content": "small", "doc_id": "s1", "score": 0.5},
    ]
    result = apply_budget_trim(chunks, budget_tokens=1)
    # kept stays empty → falls back to original
    assert len(result) >= 1


def test_single_chunk_never_trimmed():
    # len(top_chunks) <= 1 → skip trim entirely
    chunks = [{"content": "x" * 400000, "doc_id": "only"}]
    result = apply_budget_trim(chunks, budget_tokens=1)
    assert result == chunks


def test_preserves_order_of_reranked_chunks():
    # Higher-ranked chunks (earlier in list) must be kept first
    chunks = [
        {"content": "x" * 200, "doc_id": f"d{i}", "score": 1.0 - i * 0.1}
        for i in range(10)
    ]
    result = apply_budget_trim(chunks, budget_tokens=200)
    # Whatever is kept, it must be a prefix of the original list
    assert chunks[:len(result)] == result


def test_exact_budget_boundary():
    # json.dumps adds key-name overhead: {"content": "a"*400, "doc_id": "exact"} = 434 chars → 108 tokens.
    # budget=110 lets the first chunk in (108 ≤ 110) but blocks the second (108+108=216 > 110).
    one_chunk = {"content": "a" * 400, "doc_id": "exact"}
    second = {"content": "b" * 400, "doc_id": "second"}
    result = apply_budget_trim([one_chunk, second], budget_tokens=110)
    assert one_chunk in result
    assert second not in result


def test_no_trim_for_single_chunk_regardless_of_size():
    chunks = [{"content": "z" * 2000000}]
    result = apply_budget_trim(chunks, budget_tokens=0)
    assert result == chunks
