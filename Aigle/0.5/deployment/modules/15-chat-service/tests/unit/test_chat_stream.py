"""Chat JSON and SSE response contracts."""
import asyncio
import json
import os
import sys

os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("LLM_BASE_URL", "http://localhost:11434/v1")

_MODULE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(_MODULE_ROOT, "src"))
sys.path.insert(0, _MODULE_ROOT)

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Nested `src.` imports -- see test_compact_context.py for why.
from src.routers import chat as chat_router


class FakeChat:
    def __init__(self, response="今天天氣很好。我們去散步吧！要帶傘嗎？", delay=0.0, fail=False):
        self.response, self.delay, self.fail = response, delay, fail
        self.calls = []

    async def chat(self, user_id, message, history=None, session_id=None):
        self.calls.append({"user_id": user_id, "message": message, "session_id": session_id})
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("graph exploded")
        return {"response": f"<think>hmm</think>{self.response}", "search_triggered": False,
                "tool_calls": [{"tool": "x"}], "search_results": None}


@pytest.fixture
def make_client():
    def _make(service):
        app = FastAPI()
        app.include_router(chat_router.router)
        app.dependency_overrides[chat_router.get_chat_service] = lambda: service
        return TestClient(app)
    return _make


def events(text):
    out = []
    for block in text.split("\n\n"):
        lines = [line for line in block.split("\n") if line and not line.startswith(":")]
        if not lines:
            continue
        name = next(line[7:] for line in lines if line.startswith("event: "))
        data = json.loads(next(line[6:] for line in lines if line.startswith("data: ")))
        out.append((name, data))
    return out


def test_json_answer_carries_request_id(make_client):
    service = FakeChat()
    client = make_client(service)
    body = {"user_id": "u1", "message": "hi", "session_id": "s1"}
    resp = client.post("/chat", json=body, headers={"X-Request-ID": "req-123"})
    assert resp.status_code == 200
    assert resp.json()["request_id"] == "req-123"
    assert resp.json()["response"] == service.response          # think tags stripped
    generated = client.post("/chat", json=body).json()["request_id"]
    assert generated and len(generated) == 32


def test_stream_sends_start_sentences_and_end(make_client):
    service = FakeChat()
    client = make_client(service)
    resp = client.post("/chat", json={"user_id": "u1", "message": "hi", "session_id": "s1", "stream": True},
                       headers={"X-Request-ID": "req-9"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.headers["x-request-id"] == "req-9"
    got = events(resp.text)
    assert got[0] == ("start", {"request_id": "req-9", "session_id": "s1"})
    assert [d["text"] for e, d in got if e == "delta"] == ["今天天氣很好。", "我們去散步吧！", "要帶傘嗎？"]
    assert got[-1] == ("end", {"request_id": "req-9", "response": service.response,
                               "search_triggered": False, "tool_calls": 1})
    assert service.calls == [{"user_id": "u1", "message": "hi", "session_id": "s1"}]


def test_stream_reports_errors_as_an_event(make_client):
    client = make_client(FakeChat(fail=True))
    resp = client.post("/chat", json={"user_id": "u1", "message": "hi", "stream": True})
    got = events(resp.text)
    assert [e for e, _ in got] == ["start", "error"]
    assert got[1][1]["message"] == "graph exploded"


def test_stream_keeps_idle_connections_alive(make_client, monkeypatch):
    monkeypatch.setattr(chat_router, "SSE_PING_S", 0.05)
    client = make_client(FakeChat(delay=0.2))
    resp = client.post("/chat", json={"user_id": "u1", "message": "hi", "stream": True})
    assert ": ping" in resp.text
    assert events(resp.text)[-1][0] == "end"


def test_sentences_keep_text_without_terminator():
    assert chat_router._sentences("第一句。第二句沒有句號") == ["第一句。", "第二句沒有句號"]
    assert chat_router._sentences("line one\nline two") == ["line one\n", "line two"]
    assert chat_router._sentences("") == []
