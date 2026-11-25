"""Screened text never reaches the service logs (VIE01-354)."""
import hashlib
import logging

import pytest
from app.core.log_safe import describe
from app.services import guard_classifier

SECRET = "my card number is 4111 1111 1111 1111"


def test_describe_gives_length_and_hash_prefix_only():
    out = describe(SECRET)
    assert out == f"len={len(SECRET)} sha={hashlib.sha256(SECRET.encode()).hexdigest()[:12]}"
    assert "4111" not in out


def test_describe_none():
    assert describe(None) == "len=0 sha=-"


@pytest.mark.asyncio
async def test_guard_classifier_log_has_no_raw_text(monkeypatch, caplog):
    class _Adapter:
        def build_request(self, content, role, policy=None):
            return {}

        def parse(self, raw, used_policy_prompt=False):
            raise RuntimeError("stop after the input log line")

    async def _call(*a, **k):
        return "safe"

    monkeypatch.setattr(guard_classifier, "get_adapter", lambda model: _Adapter())
    monkeypatch.setattr(guard_classifier, "ollama_call", _call)
    caplog.set_level(logging.INFO, logger=guard_classifier.__name__)
    with pytest.raises(RuntimeError):
        await guard_classifier._classify_one("m", SECRET, "user", policy=None)
    assert describe(SECRET) in caplog.text
    assert "4111" not in caplog.text
