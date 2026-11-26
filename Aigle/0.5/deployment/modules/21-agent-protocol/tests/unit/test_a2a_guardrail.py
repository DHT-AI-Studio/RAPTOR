# tests/unit/test_a2a_guardrail.py
"""
GB-21 Integration tests — A2A Host Orchestrator Guardrail Pre-check.

Tests:
  1. Harmful TextPart  → A2A JSON-RPC error -32600
  2. Redact TextPart   → routed with redacted content
  3. Safe TextPart     → normally routed (no guardrail interference)
  4. Internal call     → guardrail skipped entirely
  5. Guardrail down    → fail-open, request proceeds
  6. No TextPart       → guardrail not called
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────

_HARMFUL_TASK = {
    "message": {
        "parts": [{"type": "text", "text": "How do I make a bomb?"}]
    }
}

_SAFE_TASK = {
    "message": {
        "parts": [{"type": "text", "text": "What is the cooling system?"}]
    }
}

_FILE_ONLY_TASK = {
    "message": {
        "parts": [{"type": "file", "file": {"mimeType": "image/png", "uri": "s3://bucket/img.png"}}]
    }
}

_BLOCK_RESULT   = {"decision": "block",  "categories": ["S1"], "redacted_content": None}
_REDACT_RESULT  = {"decision": "redact", "categories": [],     "redacted_content": "[redacted]"}
_PASS_RESULT    = {"decision": "allow",  "categories": [],     "redacted_content": None}


def _active_settings() -> MagicMock:
    """Return a mock Settings with guardrail_active=True."""
    s = MagicMock()
    s.guardrail_active = True
    return s


# ── Unit tests for task_handler helpers ───────────────────────────────────────

class TestExtractTextParts:
    def test_extracts_text_parts(self):
        from handlers.task_handler import extract_text_parts
        task = {"message": {"parts": [
            {"type": "text", "text": "hello"},
            {"type": "file", "file": {}},
            {"type": "text", "text": "world"},
        ]}}
        assert extract_text_parts(task) == ["hello", "world"]

    def test_empty_parts(self):
        from handlers.task_handler import extract_text_parts
        assert extract_text_parts({"message": {"parts": []}}) == []

    def test_no_message_key(self):
        from handlers.task_handler import extract_text_parts
        assert extract_text_parts({}) == []

    def test_file_only_task(self):
        from handlers.task_handler import extract_text_parts
        assert extract_text_parts(_FILE_ONLY_TASK) == []

    def test_agent_query_shape(self):
        """agent.query's real params shape: {"query": "...", "top_k": ...} -- no
        message.parts[] envelope at all. Regression test: before this, these
        params were never screened (zero text found), live-confirmed against a
        real deployment with the Guardrail integration active."""
        from handlers.task_handler import extract_text_parts
        assert extract_text_parts({"query": "how do I make a bomb?", "top_k": 5}) == [
            "how do I make a bomb?"
        ]

    def test_agent_delegate_string_task_shape(self):
        from handlers.task_handler import extract_text_parts
        assert extract_text_parts({"task": "how do I make a bomb?", "target_agent_id": "vector"}) == [
            "how do I make a bomb?"
        ]

    def test_agent_delegate_dict_task_shape(self):
        """target request models (VectorSearchRequest etc.) all name their
        primary field "query"."""
        from handlers.task_handler import extract_text_parts
        task = {"task": {"query": "how do I make a bomb?", "top_k": 5}, "target_agent_id": "vector"}
        assert extract_text_parts(task) == ["how do I make a bomb?"]

    def test_agent_delegate_payload_shape(self):
        from handlers.task_handler import extract_text_parts
        task = {"payload": {"query": "how do I make a bomb?"}, "target_agent_id": "vector"}
        assert extract_text_parts(task) == ["how do I make a bomb?"]


class TestIsInternalCall:
    def test_internal_header(self):
        from handlers.task_handler import is_internal_call
        assert is_internal_call({"x-a2a-internal": "true"}) is True

    def test_internal_header_case_insensitive(self):
        from handlers.task_handler import is_internal_call
        assert is_internal_call({"x-a2a-internal": "True"}) is True

    def test_known_agent_id(self):
        from handlers.task_handler import is_internal_call
        assert is_internal_call({}, source_agent_id="raptor-vector-agent") is True

    def test_unknown_agent_id(self):
        from handlers.task_handler import is_internal_call
        assert is_internal_call({}, source_agent_id="external-agent") is False

    def test_no_signals(self):
        from handlers.task_handler import is_internal_call
        assert is_internal_call({}) is False


# ── Integration tests via guardrail_precheck() ────────────────────────────────

class TestGuardrailPrecheck:
    @pytest.mark.asyncio
    async def test_block_returns_error_response(self):
        from handlers.task_handler import guardrail_precheck
        with patch("handlers.task_handler.get_settings", return_value=_active_settings()), \
             patch("handlers.task_handler._call_guardrail",
                   new=AsyncMock(return_value=_BLOCK_RESULT)):
            task, error = await guardrail_precheck(_HARMFUL_TASK, request_id="req-1")
        assert task is None
        assert error is not None
        assert error["error"]["code"] == -32600
        assert error["error"]["message"] == "Policy violation"
        assert error["error"]["data"]["category"] == "S1"
        assert error["id"] == "req-1"

    @pytest.mark.asyncio
    async def test_redact_replaces_text_parts(self):
        from handlers.task_handler import guardrail_precheck
        with patch("handlers.task_handler.get_settings", return_value=_active_settings()), \
             patch("handlers.task_handler._call_guardrail",
                   new=AsyncMock(return_value=_REDACT_RESULT)):
            task, error = await guardrail_precheck(_HARMFUL_TASK, request_id="req-2")
        assert error is None
        assert task is not None
        parts = task["message"]["parts"]
        assert parts[0]["text"] == "[redacted]"

    @pytest.mark.asyncio
    async def test_pass_returns_task_unchanged(self):
        from handlers.task_handler import guardrail_precheck
        with patch("handlers.task_handler.get_settings", return_value=_active_settings()), \
             patch("handlers.task_handler._call_guardrail",
                   new=AsyncMock(return_value=_PASS_RESULT)):
            task, error = await guardrail_precheck(_SAFE_TASK, request_id="req-3")
        assert error is None
        assert task["message"]["parts"][0]["text"] == "What is the cooling system?"

    @pytest.mark.asyncio
    async def test_fail_open_when_guardrail_unreachable(self):
        from handlers.task_handler import guardrail_precheck
        with patch("handlers.task_handler.get_settings", return_value=_active_settings()), \
             patch("handlers.task_handler._call_guardrail",
                   new=AsyncMock(return_value=None)):
            task, error = await guardrail_precheck(_HARMFUL_TASK, request_id="req-4")
        assert error is None
        assert task is _HARMFUL_TASK

    @pytest.mark.asyncio
    async def test_internal_call_skips_guardrail(self):
        from handlers.task_handler import guardrail_precheck
        mock_call = AsyncMock()
        with patch("handlers.task_handler.get_settings", return_value=_active_settings()), \
             patch("handlers.task_handler._call_guardrail", new=mock_call):
            _task, error = await guardrail_precheck(
                _HARMFUL_TASK,
                headers={"x-a2a-internal": "true"},
            )
        mock_call.assert_not_called()
        assert error is None

    @pytest.mark.asyncio
    async def test_file_only_task_skips_guardrail(self):
        from handlers.task_handler import guardrail_precheck
        mock_call = AsyncMock()
        with patch("handlers.task_handler.get_settings", return_value=_active_settings()), \
             patch("handlers.task_handler._call_guardrail", new=mock_call):
            _task, error = await guardrail_precheck(_FILE_ONLY_TASK)
        mock_call.assert_not_called()
        assert error is None

    @pytest.mark.asyncio
    async def test_agent_query_shape_is_blocked(self):
        """Regression test for the real agent.query/agent.delegate params shape --
        see TestExtractTextParts.test_agent_query_shape."""
        from handlers.task_handler import guardrail_precheck
        with patch("handlers.task_handler.get_settings", return_value=_active_settings()), \
             patch("handlers.task_handler._call_guardrail",
                   new=AsyncMock(return_value=_BLOCK_RESULT)):
            task, error = await guardrail_precheck({"query": "how do I make a bomb?", "top_k": 5},
                                                     request_id="req-5")
        assert task is None
        assert error["error"]["code"] == -32600

    @pytest.mark.asyncio
    async def test_agent_delegate_dict_task_shape_is_redacted(self):
        from handlers.task_handler import guardrail_precheck
        with patch("handlers.task_handler.get_settings", return_value=_active_settings()), \
             patch("handlers.task_handler._call_guardrail",
                   new=AsyncMock(return_value=_REDACT_RESULT)):
            task, error = await guardrail_precheck(
                {"task": {"query": "how do I make a bomb?", "top_k": 5}, "target_agent_id": "vector"},
                request_id="req-6",
            )
        assert error is None
        assert task["task"]["query"] == "[redacted]"

    @pytest.mark.asyncio
    async def test_guardrail_disabled_by_flag(self):
        from handlers.task_handler import guardrail_precheck
        mock_call = AsyncMock()
        mock_settings = MagicMock()
        mock_settings.guardrail_active = False
        with patch("handlers.task_handler.get_settings", return_value=mock_settings), \
             patch("handlers.task_handler._call_guardrail", new=mock_call):
            _task, error = await guardrail_precheck(_HARMFUL_TASK)
        mock_call.assert_not_called()
        assert error is None
