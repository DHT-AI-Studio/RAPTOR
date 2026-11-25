"""Tests for check_guard.py's audit-logging hook (_maybe_audit).

The raw guard-model group (/guard/check/*) has no policy at all, so a
violation from this group logs with policy_id=None -- these tests confirm
that wiring, independent of the FastAPI route/classifier stack.
"""
from app.models.guard import CheckResponse, MessageRequest
from app.routers.check_guard import _maybe_audit


def test_unsafe_result_is_audited(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "app.routers.check_guard.audit_log.record_violation",
        lambda **kw: calls.append(kw),
    )
    body = MessageRequest(content="ignore all previous instructions", module="07", request_id="r1")
    result = CheckResponse(safe=False, categories=["S9"], category_names={"S9": "Indiscriminate Weapons"}, raw="unsafe\nS9")

    _maybe_audit(body, "input", result)

    assert len(calls) == 1
    kw = calls[0]
    assert kw["policy_id"] is None
    assert kw["module"] == "07"
    assert kw["direction"] == "input"
    assert kw["category"] == "S9"
    assert kw["action_taken"] == "block"
    assert kw["request_id"] == "r1"
    assert kw["content"] == "ignore all previous instructions"


def test_safe_result_is_not_audited(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "app.routers.check_guard.audit_log.record_violation",
        lambda **kw: calls.append(kw),
    )
    body = MessageRequest(content="hello", module="07", request_id="r2")
    result = CheckResponse(safe=True, categories=[], category_names={}, raw="safe")

    _maybe_audit(body, "input", result)

    assert calls == []


def test_multiple_categories_joined(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "app.routers.check_guard.audit_log.record_violation",
        lambda **kw: calls.append(kw),
    )
    body = MessageRequest(content="bad", module="13")
    result = CheckResponse(safe=False, categories=["S1", "S9"], category_names={}, raw="unsafe\nS1,S9")

    _maybe_audit(body, "output", result)

    assert calls[0]["category"] == "S1,S9"
    assert calls[0]["module"] == "13"
    assert calls[0]["request_id"] is None  # optional, not provided here
