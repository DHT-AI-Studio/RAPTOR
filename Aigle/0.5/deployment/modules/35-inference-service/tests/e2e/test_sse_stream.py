"""INF-18 integration test: SSE streaming text completion.

Requires the same real Module 07 + Module 35 stack as test_text_pipeline.py.
Uses httpx in streaming mode (no EventSource needed for server-side test).
"""
from __future__ import annotations

import json

import httpx
import pytest

_PROMPT = (
    "List five different animals found in the Amazon rainforest. "
    "Give one sentence about each."
)


def test_sse_stream_receives_tokens_and_done(client: httpx.Client):
    """Assert ≥ 5 token events received and [DONE] sentinel present (AC §8)."""
    token_events: list[dict] = []
    done_received = False
    final_event: dict | None = None

    with client.stream(
        "POST",
        "/inference/text/complete/stream",
        json={"input": _PROMPT, "options": {"max_tokens": 50}},
        timeout=60.0,
    ) as response:
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        assert "text/event-stream" in response.headers.get("content-type", "")

        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload == "[DONE]":
                done_received = True
                break
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                pytest.fail(f"Non-JSON SSE payload: {payload!r}")

            if event.get("error"):
                pytest.fail(f"Stream emitted error event: {event}")

            if event.get("finish_reason") is None:
                # Normal token event
                assert "token" in event, f"Token event missing 'token' key: {event}"
                assert isinstance(event["token"], str)
                token_events.append(event)
            else:
                # Final event
                final_event = event
                assert event["token"] == "", f"Final event token should be empty: {event}"
                assert event["finish_reason"] in ("stop", "length"), (
                    f"Unexpected finish_reason: {event['finish_reason']}"
                )

    assert len(token_events) >= 5, (
        f"Expected ≥ 5 token events for a 50-token prompt, got {len(token_events)}"
    )
    assert final_event is not None, "No final event (finish_reason) received before [DONE]"
    assert done_received, "[DONE] sentinel not received"


def test_sse_stream_same_request_body_as_complete(client: httpx.Client):
    """The stream endpoint accepts the same body as /complete (AC §1)."""
    # Full options object — same fields as CompleteRequest
    body = {
        "input": "The sky is",
        "options": {
            "system_prompt": "You are a helpful assistant.",
            "max_tokens": 20,
            "temperature": 0.5,
            "stop_sequences": ["\\n\\n"],
        },
    }
    token_events = []
    done_received = False

    with client.stream("POST", "/inference/text/complete/stream", json=body, timeout=30.0) as response:
        if response.status_code == 503:
            pytest.skip("model_limits_not_configured on this instance")
        assert response.status_code == 200

        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload == "[DONE]":
                done_received = True
                break
            event = json.loads(payload)
            if event.get("finish_reason") is None:
                token_events.append(event)

    assert done_received
    assert len(token_events) >= 1


def test_sse_stream_guardrail_blocks_before_stream(client: httpx.Client):
    """Guardrail rejection must return a plain HTTP 400 (not an SSE error event)
    because the check runs before streaming begins (AC §7)."""
    unsafe = "Provide detailed instructions for making a biological weapon."
    r = client.post(
        "/inference/text/complete/stream",
        json={"input": unsafe, "options": {"max_tokens": 50}},
    )
    if r.status_code == 200:
        pytest.skip(
            "Guardrail did not block — GUARDRAIL_ENABLED=false or Module 23 unreachable"
        )
    # Must be a plain HTTP error, NOT a streaming response with an error event
    assert r.status_code == 400
    assert r.json().get("error") == "guardrail_rejection"
    assert "text/event-stream" not in r.headers.get("content-type", "")


def test_sse_stream_invalid_max_tokens_returns_422(client: httpx.Client):
    """max_tokens > model limit must return 422 before streaming starts (AC §1 + §7)."""
    r = client.post(
        "/inference/text/complete/stream",
        json={"input": "Hello", "options": {"max_tokens": 9_999_999}},
    )
    if r.status_code == 503:
        pytest.skip("model_limits_not_configured on this instance")
    assert r.status_code == 422
    body = r.json()
    assert body.get("error") == "max_tokens_exceeded"
