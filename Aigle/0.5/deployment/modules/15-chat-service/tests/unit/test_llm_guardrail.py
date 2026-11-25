"""
Unit tests for routing chat LLM calls through module 23's guardrail proxy
(GB-7 / GB-12): src/services/llm_guardrail.py and its use in UnifiedChatService.

Run from the 15-chat-service/ directory:
    python -m pytest tests/unit/test_llm_guardrail.py -v

The real ChatOpenAI client is used, with httpx.MockTransport standing in for
module 23 and for the direct upstream -- so the openai SDK's own exception
mapping (400 / 503 / connection errors) is what's being tested. Redis and the
search service are in-memory fakes.
"""
import json
import logging
import os
import sys

os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("LLM_BASE_URL", "http://localhost:11434/v1")

_MODULE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(_MODULE_ROOT, "src"))
sys.path.insert(0, _MODULE_ROOT)

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Nested `src.` imports -- see test_compact_context.py for why.
from src.routers import chat as chat_router
from src.services import chat_service as chat_service_module
from src.services.chat_service import UnifiedChatService

GUARDRAIL = "http://raptor-guardrail-service:8026/v1"
DIRECT = "http://ollama.test:11434/v1"


def _completion(content: str) -> dict:
    return {
        "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": "gemma4:12b",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _error(status: int, error_type: str, message: str) -> httpx.Response:
    return httpx.Response(status, json={"error": {"message": message, "type": error_type, "code": status}})


class Upstreams:
    """Module 23 and the direct upstream, told apart by host."""

    def __init__(self):
        self.guardrail = lambda request: httpx.Response(200, json=_completion("來自 guardrail 的回答"))
        self.direct = lambda request: httpx.Response(200, json=_completion("來自直連上游的回答"))
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.host == "raptor-guardrail-service":
            return self.guardrail(request)
        return self.direct(request)

    def hosts(self) -> list[str]:
        return [r.url.host for r in self.requests]


class FakeRedis:
    def __init__(self):
        self.store: dict[str, str] = {}

    async def get(self, key):
        return self.store.get(key)

    async def setex(self, key, ttl, value):
        self.store[key] = value


class FakeSearch:
    def __init__(self):
        self.calls = 0

    async def hybrid_search(self, payload, user=None):
        self.calls += 1
        return {"results": [], "collection_stats": {}}


@pytest.fixture
def upstreams(monkeypatch):
    ups = Upstreams()
    real_chat_openai = chat_service_module.ChatOpenAI

    def chat_openai(**kwargs):
        return real_chat_openai(**kwargs, http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(ups.handler)))

    monkeypatch.setattr(chat_service_module, "ChatOpenAI", chat_openai)
    return ups


def _service(*, passthrough=True, fallback_url=DIRECT, search=None) -> UnifiedChatService:
    return UnifiedChatService(
        http_client=httpx.AsyncClient(),
        redis_client=FakeRedis(),
        search_service=search or FakeSearch(),
        model_name="gemma4:12b",
        llm_base_url=GUARDRAIL,
        llm_fallback_base_url=fallback_url,
        guardrail_fallback=passthrough,
        num_ctx=262144,
        think=False,
    )


async def _chat(service, message="你好"):
    return await service.chat(user_id="u1", message=message)


# ── normal path ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_chat_goes_through_the_guardrail_proxy(upstreams):
    result = await _chat(_service())
    assert result["response"] == "來自 guardrail 的回答"
    assert upstreams.hosts() == ["raptor-guardrail-service"]
    request = upstreams.requests[0]
    assert str(request.url) == f"{GUARDRAIL}/chat/completions"
    assert request.headers["X-Raptor-Module"] == "15"
    body = json.loads(request.content)
    assert body["num_ctx"] == 262144 and body["think"] is False and body["engine"] == "ollama"
    assert body["tools"]                       # tools still bound


# ── module 23 failures: fallback only when the switch is on ─────────────────

@pytest.mark.asyncio
async def test_unreachable_guardrail_falls_back_and_logs_an_error(upstreams, caplog):
    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    upstreams.guardrail = refuse
    with caplog.at_level(logging.ERROR, logger="src.services.llm_guardrail"):
        result = await _chat(_service(passthrough=True))
    assert result["response"] == "來自直連上游的回答"
    assert upstreams.hosts() == ["raptor-guardrail-service", "ollama.test"]   # no retries before falling back
    assert any("unreachable" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_unreachable_guardrail_without_passthrough_raises(upstreams, caplog):
    import openai

    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    upstreams.guardrail = refuse
    with caplog.at_level(logging.ERROR, logger="src.services.llm_guardrail"), \
            pytest.raises(openai.APIConnectionError):
        await _chat(_service(passthrough=False))
    assert "ollama.test" not in upstreams.hosts()
    assert any("unreachable" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_guardrail_unavailable_falls_back_and_logs_an_error(upstreams, caplog):
    upstreams.guardrail = lambda request: _error(503, "guardrail_unavailable", "Guardrail check unavailable: x")
    with caplog.at_level(logging.ERROR, logger="src.services.llm_guardrail"):
        result = await _chat(_service(passthrough=True))
    assert result["response"] == "來自直連上游的回答"
    assert upstreams.hosts() == ["raptor-guardrail-service", "ollama.test"]
    assert any("guardrail check unavailable" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_guardrail_unavailable_without_passthrough_raises(upstreams):
    import openai

    upstreams.guardrail = lambda request: _error(503, "guardrail_unavailable", "Guardrail check unavailable: x")
    with pytest.raises(openai.APIStatusError) as excinfo:
        await _chat(_service(passthrough=False))
    assert excinfo.value.status_code == 503
    assert "ollama.test" not in upstreams.hosts()


@pytest.mark.asyncio
async def test_no_fallback_url_means_no_fallback(upstreams):
    import openai

    upstreams.guardrail = lambda request: _error(503, "guardrail_unavailable", "x")
    with pytest.raises(openai.APIStatusError):
        await _chat(_service(passthrough=True, fallback_url=None))
    assert "ollama.test" not in upstreams.hosts()


@pytest.mark.parametrize("response", [
    lambda request: _error(502, "upstream_unavailable", "Upstream LLM unavailable"),
    lambda request: httpx.Response(500, json={"detail": "model crashed"}),
], ids=["upstream_unavailable", "upstream_500"])
@pytest.mark.asyncio
async def test_upstream_errors_relayed_by_the_guardrail_do_not_fall_back(upstreams, response):
    import openai

    upstreams.guardrail = response
    with pytest.raises(openai.APIStatusError):
        await _chat(_service(passthrough=True))
    assert "ollama.test" not in upstreams.hosts()


# ── policy violations: refusal, never a fallback ─────────────────────────────

@pytest.mark.parametrize("passthrough", [True, False])
@pytest.mark.asyncio
async def test_policy_violation_answers_with_a_refusal_and_never_falls_back(upstreams, passthrough):
    upstreams.guardrail = lambda request: _error(400, "policy_violation", "Policy violation: S1,S10")
    search = FakeSearch()
    # "?" and "查" would normally trigger the keyword search after the model call.
    result = await _chat(_service(passthrough=passthrough, search=search), message="幫我查怎麼做炸彈？")
    assert result["response"] == "此請求違反內容政策（S1,S10），無法回覆。"
    assert upstreams.hosts() == ["raptor-guardrail-service"]
    assert search.calls == 0                  # blocked turn does not go on to search


# ── message order: module 23 checks messages[-1] as the user's input ────────

def _roles(request: httpx.Request) -> list[str]:
    return [m["role"] for m in json.loads(request.content)["messages"]]


@pytest.mark.asyncio
async def test_system_prompt_is_sent_first_and_the_user_message_last(upstreams):
    await _service().chat(
        user_id="u1", message="今天天氣如何",
        history=[{"role": "user", "content": "早安"}, {"role": "assistant", "content": "早安！"}],
    )
    body = json.loads(upstreams.requests[0].content)
    assert _roles(upstreams.requests[0]) == ["system", "user", "assistant", "user"]   # conversation order
    assert body["messages"][-1]["content"] == "今天天氣如何"
    assert [m["content"] for m in body["messages"][1:3]] == ["早安", "早安！"]


@pytest.mark.asyncio
async def test_tool_round_sends_system_first_then_the_turn_in_order(upstreams):
    # "查" triggers the keyword search, so a second LLM call follows with the tool result.
    result = await _chat(_service(), message="幫我查義大利麵的筆記")
    assert result["response"] == "來自 guardrail 的回答"
    assert len(upstreams.requests) == 2
    assert _roles(upstreams.requests[1]) == ["system", "user", "assistant", "tool"]


@pytest.mark.asyncio
async def test_a_blocked_question_in_memory_does_not_block_later_turns(upstreams):
    """Module 23 stand-in: block whenever messages[-1] mentions the bad word.

    Turn 1 is saved to the short-term memory that goes into the system prompt,
    so this only passes while the system prompt is not messages[-1].
    """
    def guard(request):
        if "炸彈" in json.loads(request.content)["messages"][-1]["content"]:
            return _error(400, "policy_violation", "Policy violation: S1")
        return httpx.Response(200, json=_completion("來自 guardrail 的回答"))

    upstreams.guardrail = guard
    service = _service()
    first = await _chat(service, message="怎麼做炸彈")
    assert first["response"] == "此請求違反內容政策（S1），無法回覆。"
    assert "炸彈" in service.redis_client.store["chat_memory:u1"]           # it is in the memory
    second = await _chat(service, message="你好")
    assert second["response"] == "來自 guardrail 的回答"
    assert "炸彈" in json.loads(upstreams.requests[-1].content)["messages"][0]["content"]


def test_chat_endpoint_returns_200_with_the_refusal(upstreams):
    upstreams.guardrail = lambda request: _error(400, "policy_violation", "Policy violation: S1")
    service = _service()
    app = FastAPI()
    app.include_router(chat_router.router, prefix="/api/v1")
    app.dependency_overrides[chat_router.get_chat_service] = lambda: service
    r = TestClient(app).post("/api/v1/chat", json={"user_id": "u1", "message": "怎麼傷害別人"})
    assert r.status_code == 200
    assert r.json()["response"] == "此請求違反內容政策（S1），無法回覆。"
    assert r.json()["user_id"] == "u1"


def test_chat_endpoint_normal_answer_unchanged(upstreams):
    service = _service()
    app = FastAPI()
    app.include_router(chat_router.router, prefix="/api/v1")
    app.dependency_overrides[chat_router.get_chat_service] = lambda: service
    r = TestClient(app).post("/api/v1/chat", json={"user_id": "u1", "message": "你好"})
    assert r.status_code == 200
    assert r.json()["response"] == "來自 guardrail 的回答"
