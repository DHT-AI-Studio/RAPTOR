"""Shared fixtures for DA-09a tests."""
from __future__ import annotations

import pathlib
import sys

import pytest
from fastapi.testclient import TestClient

# Add module root to path
MODULE_29 = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_29))

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"


# ── App fixture ───────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def app():
    """FastAPI app with mocked external dependencies."""
    import os
    os.environ.setdefault("DA_MODULE12_SYNC_URL", "http://mock-module12:8020")
    os.environ.setdefault("DA_MODULE07_URL",      "http://mock-module07:8010")
    os.environ.setdefault("DA_QDRANT_URL",        "http://mock-qdrant:6333")

    from app.main import app as _app
    return _app


@pytest.fixture(scope="session")
def client(app):
    with TestClient(app) as c:
        yield c


# ── File fixtures ─────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def text_pdf_path() -> pathlib.Path:
    p = FIXTURES_DIR / "scope3_emissions.pdf"
    assert p.exists(), f"Missing fixture: {p}"
    return p


@pytest.fixture(scope="session")
def scanned_pdf_path() -> pathlib.Path:
    return FIXTURES_DIR / "scanned.pdf"


@pytest.fixture(scope="session")
def docx_path() -> pathlib.Path:
    return FIXTURES_DIR / "sample.docx"


@pytest.fixture(scope="session")
def xlsx_path() -> pathlib.Path:
    return FIXTURES_DIR / "sample.xlsx"


@pytest.fixture(scope="session")
def csv_path() -> pathlib.Path:
    return FIXTURES_DIR / "sample.csv"


@pytest.fixture(scope="session")
def txt_path() -> pathlib.Path:
    return FIXTURES_DIR / "sample.txt"


@pytest.fixture(scope="session")
def html_path() -> pathlib.Path:
    return FIXTURES_DIR / "sample.html"


@pytest.fixture(scope="session")
def png_path() -> pathlib.Path:
    return FIXTURES_DIR / "sample.png"
