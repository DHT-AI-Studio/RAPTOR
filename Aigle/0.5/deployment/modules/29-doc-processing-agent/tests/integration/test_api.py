"""DA-09a: integration tests for POST /process and GET /formats with real fixtures."""
from __future__ import annotations

import pathlib
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def app_with_mocks():
    """App with Module 12 and Module 07 mocked at the adapter/orchestrator level."""
    import os
    os.environ["DA_MODULE12_SYNC_URL"] = "http://mock-m12:8020"
    os.environ["DA_MODULE07_URL"]      = "http://mock-m07:8010"
    os.environ["DA_QDRANT_URL"]        = "http://mock-qdrant:6333"

    from app.adapters.doc12_adapter import Doc12Adapter, ProcessedDoc
    from app.agent.orchestrator import DocAgentOrchestrator
    from app.main import app

    mock_doc = ProcessedDoc(
        filename="test.pdf",
        full_text="Test financial document Q1 2026.",
        chunks=[],
        total_chunks=0,
    )

    # Patch adapter and orchestrator at app.state level via lifespan override
    with patch.object(Doc12Adapter, "process", new=AsyncMock(return_value=mock_doc)):
        with patch.object(
            DocAgentOrchestrator, "run",
            new=AsyncMock(return_value={"task": "summarize", "filename": "test.pdf", "summary": "Mocked summary"}),
        ):
            with TestClient(app) as client:
                yield client


def test_health_endpoint(app_with_mocks):
    resp = app_with_mocks.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "dependencies" in data


def test_get_formats(app_with_mocks):
    resp = app_with_mocks.get("/api/v1/docagent/formats")
    assert resp.status_code == 200
    data = resp.json()
    assert "formats" in data
    assert "supported_tasks" in data
    assert len(data["supported_tasks"]) == 7


@pytest.mark.parametrize("fixture_file,task", [
    ("scope3_emissions.pdf", "summarize"),
    ("sample.docx",          "classify"),
    ("sample.xlsx",          "parse_table"),
    ("sample.png",           "extract"),
])
def test_process_endpoint_with_fixtures(app_with_mocks, fixture_file: str, task: str):
    p = FIXTURES / fixture_file
    if not p.exists():
        pytest.skip(f"Fixture missing: {p}  (run: python tests/integration/fixtures/make_fixtures.py)")

    with open(p, "rb") as f:
        resp = app_with_mocks.post(
            "/api/v1/docagent/process",
            data={"task": task},
            files={"file": (fixture_file, f, "application/octet-stream")},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "task" in data or "summary" in data or "category" in data


def test_process_invalid_task(app_with_mocks):
    p = FIXTURES / "sample.txt"
    if not p.exists():
        pytest.skip("Fixture missing")
    with open(p, "rb") as f:
        resp = app_with_mocks.post(
            "/api/v1/docagent/process",
            data={"task": "invalid_task"},
            files={"file": ("sample.txt", f, "text/plain")},
        )
    assert resp.status_code == 422


def test_process_empty_file(app_with_mocks):
    resp = app_with_mocks.post(
        "/api/v1/docagent/process",
        data={"task": "summarize"},
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert resp.status_code == 422


def test_unsupported_extension_returns_415(app_with_mocks):
    r = app_with_mocks.post(
        "/api/v1/docagent/process",
        files={"file": ("bad.xyz", b"not a document", "application/octet-stream")},
        data={"task": "summarize"},
    )
    assert r.status_code == 415
