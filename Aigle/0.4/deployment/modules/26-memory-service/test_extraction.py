"""
MV-10: User Preference / Entity Extraction — unit tests.

Run:
    conda run -n CIE python -m pytest test_extraction.py -v
"""
import asyncio
import json
import os
import sys

os.environ.setdefault("MEM_REDIS_HOST", "localhost")
os.environ.setdefault("MEM_STORAGE_ROOT", "/tmp/mv10_test")
os.environ.setdefault("MEM_EXTRACTION_MODEL", "qwen2.5:7b")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

import pytest
import pytest_asyncio
from fakeredis import FakeAsyncRedis

from services.extractor import (
    ExtractionOp,
    _format_turn,
    _item_text,
    _parse_ops,
    extract_and_store,
)
from services.long_term_memory import LongTermMemoryService, FactAddRequest
from services.session_memory import SessionMemoryService, TurnAppendRequest


# ── Pure-function tests (no I/O) ──────────────────────────────────────────────

def test_format_turn():
    turn = {"user_message": "Hello", "assistant_response": "Hi there"}
    out = _format_turn(turn)
    assert "Hello" in out
    assert "Hi there" in out


def test_parse_ops_add_preference():
    raw = '[{"op": "ADD", "type": "preference", "text": "user prefers dark mode"}]'
    ops = _parse_ops(raw)
    assert len(ops) == 1
    assert ops[0].op == "ADD"
    assert ops[0].type == "preference"
    assert "dark mode" in ops[0].text


def test_parse_ops_add_entity():
    raw = '[{"op": "ADD", "type": "entity", "name": "Project Alpha", "context": "user main project"}]'
    ops = _parse_ops(raw)
    assert len(ops) == 1
    assert ops[0].op == "ADD"
    assert ops[0].name == "Project Alpha"
    assert _item_text(ops[0]) == "Project Alpha: user main project"


def test_parse_ops_add_fact():
    raw = '[{"op": "ADD", "type": "fact", "text": "deadline is 2026-09-01"}]'
    ops = _parse_ops(raw)
    assert ops[0].type == "fact"
    assert "2026-09-01" in ops[0].text


def test_parse_ops_delete():
    raw = '[{"op": "DELETE", "id": "42"}]'
    ops = _parse_ops(raw)
    assert ops[0].op == "DELETE"
    assert ops[0].id == "42"


def test_parse_ops_update():
    raw = '[{"op": "UPDATE", "id": "7", "text": "user prefers Chinese now"}]'
    ops = _parse_ops(raw)
    assert ops[0].op == "UPDATE"
    assert ops[0].id == "7"
    assert "Chinese" in ops[0].text


def test_parse_ops_strips_markdown_fences():
    raw = '```json\n[{"op": "ADD", "type": "fact", "text": "foo"}]\n```'
    ops = _parse_ops(raw)
    assert len(ops) == 1


def test_parse_ops_empty_array():
    assert _parse_ops("[]") == []


def test_parse_ops_invalid_json_returns_empty():
    assert _parse_ops("not json at all") == []


def test_parse_ops_unknown_op_skipped():
    raw = '[{"op": "MERGE", "text": "whatever"}, {"op": "ADD", "type": "fact", "text": "ok"}]'
    ops = _parse_ops(raw)
    assert len(ops) == 1
    assert ops[0].op == "ADD"


def test_item_text_preference():
    op = ExtractionOp(op="ADD", type="preference", text="user likes Python")
    assert _item_text(op) == "user likes Python"


def test_item_text_entity_both_fields():
    op = ExtractionOp(op="ADD", type="entity", name="Alice", context="user's manager")
    assert _item_text(op) == "Alice: user's manager"


def test_item_text_entity_name_only():
    op = ExtractionOp(op="ADD", type="entity", name="Alice")
    assert _item_text(op) == "Alice"


# ── Integration tests (mocked LLM, real MemVID) ───────────────────────────────

@pytest_asyncio.fixture
async def lt_svc(tmp_path):
    yield LongTermMemoryService(storage_root=str(tmp_path))


@pytest.mark.asyncio
async def test_extract_add_preference(lt_svc, tmp_path):
    async def fake_llm(turn_text, existing):
        return '[{"op": "ADD", "type": "preference", "text": "user prefers Traditional Chinese"}]'

    turn = {"user_message": "請用繁體中文", "assistant_response": "好的"}
    stored = await extract_and_store("u1", "sess1", turn, lt_svc, _llm_fn=fake_llm)
    assert stored == 1

    facts = await lt_svc.get_facts("u1")
    assert any("Traditional Chinese" in f.get("text", "") for f in facts)


@pytest.mark.asyncio
async def test_extract_add_entity(lt_svc, tmp_path):
    async def fake_llm(turn_text, existing):
        return '[{"op": "ADD", "type": "entity", "name": "Project Alpha", "context": "user main project"}]'

    turn = {"user_message": "Project Alpha is my main project", "assistant_response": "Got it"}
    stored = await extract_and_store("u2", "sess2", turn, lt_svc, _llm_fn=fake_llm)
    assert stored == 1


@pytest.mark.asyncio
async def test_extract_add_fact(lt_svc, tmp_path):
    async def fake_llm(turn_text, existing):
        return '[{"op": "ADD", "type": "fact", "text": "deadline is 2026-09-01"}]'

    turn = {"user_message": "My deadline is September 1st 2026", "assistant_response": "Noted"}
    stored = await extract_and_store("u3", "sess3", turn, lt_svc, _llm_fn=fake_llm)
    assert stored == 1
    facts = await lt_svc.get_facts("u3")
    assert any("2026-09-01" in f.get("text", "") for f in facts)


@pytest.mark.asyncio
async def test_extract_delete_removes_fact(lt_svc, tmp_path):
    # First add a fact manually
    resp = await lt_svc.add_fact("u4", FactAddRequest(text="user prefers English", frame_type="preference"))
    frame_id = resp.frame_id

    async def fake_llm(turn_text, existing):
        return f'[{{"op": "DELETE", "id": "{frame_id}"}}]'

    turn = {"user_message": "Actually use Chinese", "assistant_response": "OK"}
    stored = await extract_and_store("u4", "sess4", turn, lt_svc, _llm_fn=fake_llm)
    assert stored == 1

    # The deleted fact should not appear in search
    from services.long_term_memory import SearchRequest
    hits = await lt_svc.search("u4", SearchRequest(query="English preference", top_k=5))
    assert not any("prefers English" in h.text for h in hits)


@pytest.mark.asyncio
async def test_extract_update_replaces_fact(lt_svc, tmp_path):
    resp = await lt_svc.add_fact("u5", FactAddRequest(text="user prefers English", frame_type="preference"))
    frame_id = resp.frame_id

    async def fake_llm(turn_text, existing):
        return f'[{{"op": "UPDATE", "id": "{frame_id}", "type": "preference", "text": "user prefers Traditional Chinese"}}]'

    turn = {"user_message": "Switch to Chinese please", "assistant_response": "OK"}
    stored = await extract_and_store("u5", "sess5", turn, lt_svc, _llm_fn=fake_llm)
    assert stored == 1

    facts = await lt_svc.get_facts("u5")
    assert any("Traditional Chinese" in f.get("text", "") for f in facts)


@pytest.mark.asyncio
async def test_extract_returns_zero_on_llm_failure(lt_svc):
    async def failing_llm(turn_text, existing):
        raise ConnectionError("Ollama not reachable")

    turn = {"user_message": "hello", "assistant_response": "hi"}
    stored = await extract_and_store("u6", "sess6", turn, lt_svc, _llm_fn=failing_llm)
    assert stored == 0


@pytest.mark.asyncio
async def test_extract_empty_array_stores_nothing(lt_svc):
    async def fake_llm(turn_text, existing):
        return "[]"

    turn = {"user_message": "What is 2+2?", "assistant_response": "4"}
    stored = await extract_and_store("u7", "sess7", turn, lt_svc, _llm_fn=fake_llm)
    assert stored == 0


@pytest.mark.asyncio
async def test_extract_passes_existing_facts_to_llm(lt_svc, tmp_path):
    # Pre-seed a fact
    await lt_svc.add_fact("u8", FactAddRequest(text="user likes Python", frame_type="preference"))

    received_existing: list = []

    async def fake_llm(turn_text, existing):
        received_existing.extend(existing)
        return "[]"

    turn = {"user_message": "I'm a Python dev", "assistant_response": "Great"}
    await extract_and_store("u8", "sess8", turn, lt_svc, _llm_fn=fake_llm)

    # LLM should have received the existing fact
    assert any("Python" in item.get("text", "") for item in received_existing)


@pytest.mark.asyncio
async def test_extract_multiple_ops_in_one_turn(lt_svc, tmp_path):
    async def fake_llm(turn_text, existing):
        return json.dumps([
            {"op": "ADD", "type": "preference", "text": "user prefers Python"},
            {"op": "ADD", "type": "fact", "text": "user works at ACME Corp"},
        ])

    turn = {"user_message": "I work at ACME and use Python", "assistant_response": "Noted"}
    stored = await extract_and_store("u9", "sess9", turn, lt_svc, _llm_fn=fake_llm)
    assert stored == 2


# ── Dedup check tests (embedding shortlist + LLM judgment) ───────────────────

@pytest.mark.asyncio
async def test_dedup_skips_verbatim_duplicate(lt_svc, tmp_path):
    existing = await lt_svc.add_fact(
        "u10", FactAddRequest(text="用戶的車型是 Toyota Camry 2022", frame_type="fact")
    )

    async def fake_llm(turn_text, existing_hits):
        return '[{"op": "ADD", "type": "fact", "text": "用戶的車型是 Toyota Camry 2022"}]'

    async def fake_dedup_llm(new_text, candidates):
        assert candidates  # embedding shortlist must have found the verbatim match
        return f"DUPLICATE:{candidates[0]['frame_id']}"

    turn = {"user_message": "我的車是 Toyota Camry 2022", "assistant_response": "了解"}
    stored = await extract_and_store(
        "u10", "sess10", turn, lt_svc, _llm_fn=fake_llm, _dedup_llm_fn=fake_dedup_llm
    )
    assert stored == 0

    facts = await lt_svc.get_facts("u10")
    assert len(facts) == 1
    assert facts[0]["frame_id"] == existing.frame_id


@pytest.mark.asyncio
async def test_dedup_converts_contradiction_to_update(lt_svc, tmp_path):
    old = await lt_svc.add_fact(
        "u11", FactAddRequest(text="user prefers English", frame_type="preference")
    )

    async def fake_llm(turn_text, existing_hits):
        return '[{"op": "ADD", "type": "preference", "text": "user prefers Traditional Chinese"}]'

    async def fake_dedup_llm(new_text, candidates):
        return f"UPDATE:{candidates[0]['frame_id']}"

    turn = {"user_message": "Switch to Chinese please", "assistant_response": "OK"}
    stored = await extract_and_store(
        "u11", "sess11", turn, lt_svc, _llm_fn=fake_llm, _dedup_llm_fn=fake_dedup_llm
    )
    assert stored == 1

    facts = await lt_svc.get_facts("u11")
    assert len(facts) == 1
    assert facts[0]["frame_id"] != old.frame_id
    assert "Traditional Chinese" in facts[0]["text"]


@pytest.mark.asyncio
async def test_dedup_skips_llm_call_when_no_candidates(lt_svc, tmp_path):
    dedup_calls: list = []

    async def fake_llm(turn_text, existing_hits):
        return '[{"op": "ADD", "type": "fact", "text": "user works at ACME Corp"}]'

    async def fake_dedup_llm(new_text, candidates):
        dedup_calls.append((new_text, candidates))
        return "NEW"

    turn = {"user_message": "I work at ACME", "assistant_response": "Noted"}
    stored = await extract_and_store(
        "u12", "sess12", turn, lt_svc, _llm_fn=fake_llm, _dedup_llm_fn=fake_dedup_llm
    )
    assert stored == 1
    assert dedup_calls == []  # no existing facts -> shortlist empty -> dedup LLM never called


@pytest.mark.asyncio
async def test_dedup_fails_open_on_llm_error(lt_svc, tmp_path):
    await lt_svc.add_fact(
        "u13", FactAddRequest(text="user prefers English", frame_type="preference")
    )

    async def fake_llm(turn_text, existing_hits):
        return '[{"op": "ADD", "type": "preference", "text": "user prefers French"}]'

    async def failing_dedup_llm(new_text, candidates):
        raise ConnectionError("Module 07 unreachable")

    turn = {"user_message": "I prefer French", "assistant_response": "OK"}
    stored = await extract_and_store(
        "u13", "sess13", turn, lt_svc, _llm_fn=fake_llm, _dedup_llm_fn=failing_dedup_llm
    )
    # Dedup check failed -> fails open to NEW -> ADD still proceeds
    assert stored == 1
    facts = await lt_svc.get_facts("u13")
    assert len(facts) == 2


@pytest.mark.asyncio
async def test_session_append_triggers_extraction(tmp_path):
    """append_turn fires background extraction task without blocking response."""
    redis = FakeAsyncRedis(decode_responses=True)
    extracted: list[dict] = []

    async def fake_llm(turn_text, existing):
        extracted.append({"turn": turn_text})
        return "[]"

    svc = SessionMemoryService(redis=redis, storage_root=str(tmp_path))

    # Monkey-patch _trigger_extraction to use our fake LLM
    async def patched_trigger(user_id, session_id, turn):
        from services.long_term_memory import LongTermMemoryService
        lt = LongTermMemoryService(storage_root=str(tmp_path))
        from services.extractor import extract_and_store
        await extract_and_store(user_id, session_id, turn, lt, _llm_fn=fake_llm)

    svc._trigger_extraction = patched_trigger

    await svc.append_turn("u_trigger", "s1", TurnAppendRequest(
        user_message="I prefer English", assistant_response="OK"
    ))
    # Allow background task to run
    await asyncio.sleep(0.1)
    assert len(extracted) == 1

    await redis.aclose()
