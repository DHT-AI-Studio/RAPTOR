import pytest
from app.services.temporal_parse import parse_client


@pytest.fixture(autouse=True)
def no_module25(monkeypatch):
    """Unit tests never reach Module 25: `parse` is not available (the interim regex reads dates) unless a
    test says otherwise."""
    monkeypatch.setattr(parse_client, "available", lambda: False)
