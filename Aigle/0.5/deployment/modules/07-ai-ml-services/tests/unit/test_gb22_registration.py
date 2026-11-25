"""GB-22: module 07's startup registration must require both GUARDRAIL_URL and
the explicit enable flag -- not URL presence alone, or the coverage matrix
reports screened=true for a hook that never actually checks anything
(GUARDRAIL_URL/AIML_GUARDRAIL_ENABLED are deliberately independent switches,
see src/api/guardrail_hook.py's own docstring)."""
from __future__ import annotations

import sys
import types
from typing import ClassVar

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

# src.main pulls in datasets_api -> dataset_manager, whose module-level
# DatasetManager() singleton does real lakeFS/MLflow/HuggingFace config
# lookups at import time (conftest's config stub doesn't cover those keys,
# and nothing else in this suite needs the real thing) -- same sys.modules
# stub technique conftest.py already uses for src.core.config.
_fake_datasets_api = types.ModuleType("src.api.datasets_api")
_fake_datasets_api.router = APIRouter()
sys.modules.setdefault("src.api.datasets_api", _fake_datasets_api)

from src.main import app


class _FakeResponse:
    def raise_for_status(self) -> None:
        pass


class _FakeAsyncClient:
    calls: ClassVar[list[tuple[str, dict]]] = []

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> None:
        pass

    async def post(self, url: str, json: dict) -> _FakeResponse:
        _FakeAsyncClient.calls.append((url, json))
        return _FakeResponse()


@pytest.fixture(autouse=True)
def _reset_calls(monkeypatch):
    _FakeAsyncClient.calls = []
    monkeypatch.setattr("src.main.httpx.AsyncClient", _FakeAsyncClient)
    monkeypatch.delenv("GUARDRAIL_URL", raising=False)
    monkeypatch.delenv("GUARDRAIL_ENABLED", raising=False)
    monkeypatch.delenv("AIML_GUARDRAIL_ENABLED", raising=False)


def test_no_registration_when_url_set_but_not_enabled(monkeypatch):
    """The exact bug this guards against: GUARDRAIL_URL populated (as module
    13 needs it to be, regardless of this hook) but AIML_GUARDRAIL_ENABLED
    left at its false default must NOT register -- the hook never calls
    out, so the matrix must say screened=false."""
    monkeypatch.setenv("GUARDRAIL_URL", "http://raptor-guardrail-service:8026")
    with TestClient(app):
        pass
    assert _FakeAsyncClient.calls == []


def test_no_registration_when_enabled_but_no_url(monkeypatch):
    monkeypatch.setenv("AIML_GUARDRAIL_ENABLED", "true")
    with TestClient(app):
        pass
    assert _FakeAsyncClient.calls == []


def test_registers_when_both_url_and_enabled(monkeypatch):
    monkeypatch.setenv("GUARDRAIL_URL", "http://raptor-guardrail-service:8026")
    monkeypatch.setenv("AIML_GUARDRAIL_ENABLED", "true")
    with TestClient(app):
        pass
    assert len(_FakeAsyncClient.calls) == 1
    url, body = _FakeAsyncClient.calls[0]
    assert url == "http://raptor-guardrail-service:8026/api/0.4/guardrail/integration/register"
    assert body == {"entry_point": "module_07_hook", "module": "07-ai-ml-services"}


def test_bare_guardrail_enabled_also_works(monkeypatch):
    """GUARDRAIL_ENABLED (bare) wins over AIML_GUARDRAIL_ENABLED when both are
    set -- same precedence guardrail_hook.py's own _enabled() documents."""
    monkeypatch.setenv("GUARDRAIL_URL", "http://raptor-guardrail-service:8026")
    monkeypatch.setenv("AIML_GUARDRAIL_ENABLED", "false")
    monkeypatch.setenv("GUARDRAIL_ENABLED", "true")
    with TestClient(app):
        pass
    assert len(_FakeAsyncClient.calls) == 1
