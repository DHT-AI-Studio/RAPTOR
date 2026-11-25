"""
Unit tests for `/api/{v}/search/tkg` (search.py): the gateway forwards TMP-4's `entity`
and `keywords` to Module 25 unchanged, and leaves them out when the caller did.

Run from the 13-api-services/ directory:
    python -m pytest tests/unit/test_tkg_proxy.py -v
"""
import os
import sys

os.environ.setdefault("GATEWAY_AUTH_SERVICE_URL", "http://localhost:8800")
os.environ.setdefault("GATEWAY_KEYCLOAK_URL", "http://localhost:8080")
os.environ.setdefault("GATEWAY_KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
os.environ.setdefault("GATEWAY_KAFKA_TOPICS", '{"default": "default-topic"}')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
from app.api.dependencies import get_current_user
from app.core.config import get_settings
from app.main import app
from app.routers.search import get_hybrid_search_service, get_personal_db_service
from fastapi.testclient import TestClient

FAKE_SUB = "jwt-verified-user-42"
TKG = f"/api/{get_settings().api_version}/search/tkg"

client = TestClient(app)


class FakePersonalDB:
    def __init__(self):
        self.bodies = []

    async def ensure_database(self, user_id):
        pass

    async def temporal_search(self, user_id, body):
        self.bodies.append((user_id, body))
        return {"query": body["query"], "matched_entities": [], "subgraph_nodes": [], "subgraph_edges": [],
                "temporal_facts": [{"fact_id": "f1"}], "moment_ids": [], "retrieval": "temporal",
                "complete": True}


class FakeSearch:
    async def enrich_with_urls(self, results, user):
        pass


@pytest.fixture
def db():
    fake = FakePersonalDB()
    overrides = {
        get_current_user: lambda: {"sub": FAKE_SUB},
        get_personal_db_service: lambda: fake,
        get_hybrid_search_service: lambda: FakeSearch(),
    }
    previous = {k: app.dependency_overrides.get(k) for k in overrides}
    app.dependency_overrides.update(overrides)
    yield fake
    for k, v in previous.items():
        if v is None:
            app.dependency_overrides.pop(k, None)
        else:
            app.dependency_overrides[k] = v


def test_entity_and_keywords_are_forwarded(db):
    r = client.post(TKG, json={"query": "LG phone", "keywords": ["LG", "phone"], "entity": "LG",
                               "time_start": "2010-01-01T00:00:00.000+08:00"})
    assert r.status_code == 200
    user, body = db.bodies[0]
    assert user == FAKE_SUB
    assert body["keywords"] == ["LG", "phone"] and body["entity"] == "LG"
    assert r.json()["retrieval"] == "temporal" and r.json()["complete"] is True   # Module 25's fields pass through


def test_absent_entity_and_keywords_are_not_sent(db):
    r = client.post(TKG, json={"query": "What happened after 2011?", "time_start": "2012-01-01"})
    assert r.status_code == 200
    _, body = db.bodies[0]
    assert "entity" not in body and "keywords" not in body


def test_allen_match_fields_are_forwarded_and_labels_pass_through(db):
    """P07: `match` / `relations` / `certainty` reach Module 25 unchanged; its `allen` labels come back as they are."""
    db.temporal_search = _replying(db, [{"fact_id": "barra", "allen": {"vs_window": ["contains"], "certainty": "certain"}}])
    r = client.post(TKG, json={"query": "Hugo Barra", "time_start": "2010", "time_end": "2010",
                               "match": "overlaps", "relations": ["contains", "during"], "certainty": "certain"})
    assert r.status_code == 200
    _, body = db.bodies[0]
    assert (body["match"], body["relations"], body["certainty"]) == ("overlaps", ["contains", "during"], "certain")
    assert r.json()["temporal_facts"][0]["allen"] == {"vs_window": ["contains"], "certainty": "certain"}


def test_absent_allen_fields_are_not_sent(db):
    r = client.post(TKG, json={"query": "What happened after 2011?", "time_start": "2012-01-01"})
    assert r.status_code == 200
    _, body = db.bodies[0]
    assert not {"match", "relations", "certainty"} & set(body)              # Module 25 keeps its defaults


@pytest.mark.parametrize("bad", [{"match": "sideways"}, {"relations": ["sideways"]}, {"certainty": "maybe"}])
def test_unknown_allen_values_are_rejected_at_the_gateway(db, bad):
    r = client.post(TKG, json={"query": "q", **bad})
    assert r.status_code == 422 and db.bodies == []


def _replying(db, facts):
    async def temporal_search(user_id, body):
        db.bodies.append((user_id, body))
        return {"query": body["query"], "matched_entities": [], "subgraph_nodes": [], "subgraph_edges": [],
                "temporal_facts": facts, "moment_ids": [], "retrieval": "entity", "complete": True}
    return temporal_search


# ── P08 / P09: contract v2 ─────────────────────────────────────────────────────

def test_v2_fields_and_unknown_fields_reach_module_25(db):
    r = client.post(TKG, json={"query": "Android", "version": 2, "sort": "relevance", "cursor": "abc",
                               "window": {"start": "2010"}, "timezone": "Asia/Taipei", "as_of": "2026-01-01",
                               "reference": {"text": "x"}, "reference_relations": ["after"]})
    assert r.status_code == 200
    _, body = db.bodies[0]
    assert (body["version"], body["sort"], body["cursor"], body["window"]) == (2, "relevance", "abc", {"start": "2010"})
    assert body["as_of"] == "2026-01-01"               # not dropped here: Module 25 refuses it by name
    assert (body["reference"], body["reference_relations"]) == ({"text": "x"}, ["after"])      # P16


def test_a_refused_v2_request_is_a_structured_400_not_a_500():
    import httpx
    from app.services.personal_db_service import PersonalDBService

    refusal = {"error": "UNSUPPORTED_FEATURES", "features": ["as_of"], "see": "/personal/temporal/capabilities"}

    def handler(request):
        if request.url.path.endswith("/internal/db/init"):
            return httpx.Response(200, json={})
        return httpx.Response(400, json={"detail": refusal})

    svc = PersonalDBService(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), settings=get_settings())
    overrides = {get_current_user: lambda: {"sub": FAKE_SUB}, get_personal_db_service: lambda: svc,
                 get_hybrid_search_service: lambda: FakeSearch()}
    previous = {k: app.dependency_overrides.get(k) for k in overrides}
    app.dependency_overrides.update(overrides)
    try:
        r = client.post(TKG, json={"query": "Android", "version": 2, "as_of": "2026-01-01"})
    finally:
        for k, v in previous.items():
            app.dependency_overrides.pop(k, None) if v is None else app.dependency_overrides.__setitem__(k, v)
    assert r.status_code == 400 and r.json()["detail"] == refusal
