"""P21: POST /api/{v}/temporal/ingest -- inline text becomes an expiring asset sent for processing, an
uploaded asset is sent for analysis, unsupported types are refused, and a retried request is replayed."""
import os
import sys

os.environ.setdefault("GATEWAY_KAFKA_BOOTSTRAP_SERVERS", '["localhost:9092"]')
os.environ.setdefault("GATEWAY_KAFKA_TOPICS", '{"document": "document-processing-requests"}')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
from app.api.dependencies import get_current_user, get_storage_service
from app.core.config import get_settings
from app.main import app
from app.routers import temporal as temporal_router
from app.routers.asset import get_kafka_producer
from app.routers.search import get_personal_db_service
from fastapi import HTTPException
from fastapi.testclient import TestClient

PATH = f"/api/{get_settings().api_version}/temporal/ingest"
client = TestClient(app)


class FakeStorage:
    def __init__(self):
        self.uploads, self.registered, self.exists, self.register_outcome = [], [], False, None

    async def upload_file(self, primary_file, archive_date_or_ttl, destroy_date_or_ttl, user):
        self.uploads.append((primary_file.filename, primary_file.file.read().decode(), archive_date_or_ttl,
                             destroy_date_or_ttl, user.user_id))
        return {"asset_path": "u1/inline", "version_id": "v-new", "primary_filename": primary_file.filename,
                "status": "uploaded", "existence_info": {"exists": self.exists}}

    async def register_asset(self, asset_path, version_id, user):
        self.registered.append((asset_path, version_id))
        if isinstance(self.register_outcome, Exception):
            raise self.register_outcome
        return self.register_outcome or {"outcome": "registered"}


class FakePDB:
    def __init__(self):
        self.expiries, self.expiry_reply = [], None

    async def ensure_database(self, user_id):
        pass

    async def temporal_status(self, user_id, version_id):
        raise HTTPException(status_code=404, detail="no job yet")

    async def temporal_expiry(self, user_id, version_id):
        if self.expiry_reply is None:
            raise HTTPException(status_code=404, detail="none")
        return self.expiry_reply

    async def set_temporal_expiry(self, user_id, version_id, expires_at):
        self.expiries.append((version_id, expires_at))
        return {}


class FakeRedis:
    def __init__(self):
        self.data = {}

    async def get(self, key):
        return self.data.get(key)

    async def setex(self, key, ttl, value):
        self.data[key] = value


@pytest.fixture
def env(monkeypatch):
    storage, pdb, redis, sent = FakeStorage(), FakePDB(), FakeRedis(), []
    overrides = {get_current_user: lambda: {"sub": "u1"}, get_storage_service: lambda: storage,
                 get_personal_db_service: lambda: pdb, get_kafka_producer: lambda: object()}
    previous = {k: app.dependency_overrides.get(k) for k in overrides}
    app.dependency_overrides.update(overrides)

    async def send(self, upload_result, user_id, branch_id, processing_mode, temp_file_path=None):
        sent.append((upload_result["version_id"], upload_result.get("primary_filename"), temp_file_path))
        return {"status": "sent"}

    async def save(upload, result):
        return "/nfs/v-new/x.txt"

    monkeypatch.setattr(temporal_router.KafkaService, "send_processing_request", send)
    monkeypatch.setattr(temporal_router, "_save_to_nfs", save)
    old_redis = getattr(app.state, "redis_client", None)
    app.state.redis_client = redis
    yield storage, pdb, redis, sent
    app.state.redis_client = old_redis
    for k, v in previous.items():
        app.dependency_overrides.pop(k, None) if v is None else app.dependency_overrides.__setitem__(k, v)


TEXT = {"content": "Google acquired Android in July 2005.", "content_type": "text/plain",
        "filename": "notes on android!.md", "idempotency_key": "k1"}


def test_inline_text_becomes_an_expiring_asset_sent_for_processing(env):
    storage, pdb, _, sent = env
    r = client.post(PATH, json=TEXT)
    assert r.status_code == 202
    body = r.json()
    assert (body["source_version_id"], body["job_id"], body["status"], body["duplicate"]) == ("v-new", "v-new", "pending", False)
    assert storage.uploads == [("notes_on_android.txt", TEXT["content"], "7", "7", "u1")]
    assert sent == [("v-new", "notes_on_android.txt", "/nfs/v-new/x.txt")]
    assert pdb.expiries == [("v-new", body["expires_at"])] and body["expires_at"].endswith("Z")
    assert body["status_url"].endswith("/temporal/status/v-new")


def test_a_retry_with_the_same_key_is_replayed_and_a_different_body_refused(env):
    storage, _, _, sent = env
    first = client.post(PATH, json=TEXT).json()
    assert client.post(PATH, json=TEXT).json() == first and len(storage.uploads) == 1 and len(sent) == 1
    r = client.post(PATH, json={**TEXT, "content": "something else"})
    assert r.status_code == 409 and r.json()["detail"]["error"] == "IDEMPOTENCY_KEY_REUSED"


@pytest.mark.parametrize("body, code, error", [
    ({**TEXT, "content_type": "text/html"}, 415, "UNSUPPORTED_CONTENT_TYPE"),
    ({**TEXT, "content_type": "text/csv"}, 415, "UNSUPPORTED_CONTENT_TYPE"),
    ({**TEXT, "content": "   "}, 400, "INVALID_INGEST"),
    ({**TEXT, "content": "x" * 1_000_001}, 413, "CONTENT_TOO_LARGE"),
    ({**TEXT, "asset": {"asset_path": "a", "version_id": "v"}}, 400, "INVALID_INGEST"),
    ({"idempotency_key": "k"}, 400, "INVALID_INGEST"),
])
def test_refusals(env, body, code, error):
    storage, _, _, sent = env
    r = client.post(PATH, json=body)
    assert r.status_code == code and r.json()["detail"]["error"] == error
    assert storage.uploads == [] and sent == []                       # nothing stored, nothing processed


def test_identical_text_returns_its_source_or_says_it_expired(env):
    storage, pdb, _, sent = env
    storage.exists = True
    pdb.expiry_reply = {"expires_at": 1893456000000, "expired": False}
    body = client.post(PATH, json=TEXT).json()
    assert body["duplicate"] and body["expires_at"] == "2030-01-01T00:00:00Z" and sent == []
    pdb.expiry_reply = {"expires_at": 1000, "expired": True}
    r = client.post(PATH, json={**TEXT, "idempotency_key": "k2"})
    assert r.status_code == 409 and r.json()["detail"]["error"] == "SOURCE_EXPIRED"


def test_an_uploaded_asset_is_sent_for_analysis_if_it_needs_it(env):
    storage, pdb, _, sent = env
    storage.register_outcome = {"outcome": "needs_processing",
                                "metadata": {"version_id": "v-up", "asset_path": "u1/report",
                                             "primary_filename": "report.pdf", "status": "uploaded"}}
    body = client.post(PATH, json={"asset": {"asset_path": "u1/report", "version_id": "v-up"},
                                   "idempotency_key": "a1"}).json()
    assert (body["source_version_id"], body["expires_at"]) == ("v-up", None)          # uploads keep their own expiry
    assert sent == [("v-up", "report.pdf", None)] and pdb.expiries == []

    storage.register_outcome = {"outcome": "needs_processing", "metadata": {"primary_filename": "page.html"}}
    r = client.post(PATH, json={"asset": {"asset_path": "u1/page", "version_id": "v-h"}, "idempotency_key": "a2"})
    assert r.status_code == 415 and len(sent) == 1                    # never processed as plain text

    storage.register_outcome = HTTPException(status_code=409, detail="already active")
    assert client.post(PATH, json={"asset": {"asset_path": "u1/report", "version_id": "v-up"},
                                   "idempotency_key": "a3"}).status_code == 202
    storage.register_outcome = HTTPException(status_code=404, detail="not yours")
    assert client.post(PATH, json={"asset": {"asset_path": "x", "version_id": "y"},
                                   "idempotency_key": "a4"}).status_code == 404


def test_capabilities_list_ingest(env):
    async def caps(user_id):
        return {"operations": {"ingest": None}, "tkg": {}}
    _, pdb, _, _ = env
    pdb.temporal_capabilities = caps
    body = client.get(f"/api/{get_settings().api_version}/temporal/capabilities").json()
    assert body["operations"]["ingest"] == "/api/{version}/temporal/ingest"
    assert body["ingest"]["inline_content_types"] == ["text/plain", "text/markdown"]
    assert body["ingest"]["inline_retention_days"] == 7 and "html" not in body["ingest"]["asset_extensions"]


# ── P21: query / ingest_and_query ─────────────────────────────────────────────

@pytest.fixture
def asks(env, monkeypatch):
    from app.services import temporal_ingest
    seen = []

    async def ask(agent_url, question, user_id, authorization, scope, top_k):
        seen.append((question, scope, top_k))
        return {"answer": "In July 2005.", "sources": [{"doc_id": "d"}], "graph_context": "g",
                "temporal_claims": [{"verified": True}]}
    monkeypatch.setattr(temporal_ingest, "ask", ask)
    monkeypatch.setattr(temporal_ingest, "POLL_SECONDS", 0)
    return seen


def _statuses(pdb, sequence):
    it = iter(sequence)

    async def status(user_id, version_id):
        value = next(it, sequence[-1])
        if value is None:
            raise HTTPException(status_code=404, detail="no job yet")
        return {"status": value}
    pdb.temporal_status = status


def test_query_stores_nothing_and_answers_in_scope(env, asks):
    storage, _, _, sent = env
    r = client.post(PATH, json={"operation": "query", "question": "When?", "source_version_ids": ["v1", "v2"]})
    assert r.status_code == 200 and r.json()["answer"] == "In July 2005."
    assert asks == [("When?", ["v1", "v2"], 5)] and storage.uploads == [] and sent == []
    r = client.post(PATH, json={"operation": "query", "question": "When?"})
    assert r.status_code == 200 and asks[-1][1] is None                 # all the caller's sources


def test_ingest_and_query_answers_from_that_source_once_processed(env, asks):
    _, pdb, _, _ = env
    _statuses(pdb, [None, "running", "complete"])
    r = client.post(PATH, json={**TEXT, "operation": "ingest_and_query", "question": "When did Google buy Android?"})
    body = r.json()
    assert r.status_code == 200 and (body["answer_status"], body["status"]) == ("answered", "complete")
    assert body["answer"] == "In July 2005." and body["temporal_claims"] == [{"verified": True}]
    assert asks == [("When did Google buy Android?", ["v-new"], 5)]   # this source only


def test_ingest_and_query_never_answers_without_the_facts(env, asks):
    _, pdb, _, _ = env
    _statuses(pdb, [None, "running"])
    body = client.post(PATH, json={**TEXT, "operation": "ingest_and_query", "question": "q", "wait_seconds": 0}).json()
    assert (body["answer_status"], body["answer"], body["status"]) == ("pending", None, "running") and asks == []
    _statuses(pdb, ["failed"])
    body = client.post(PATH, json={**TEXT, "idempotency_key": "k9", "operation": "ingest_and_query",
                                   "question": "q", "wait_seconds": 0}).json()
    assert (body["answer_status"], body["answer"]) == ("not_answered", None) and asks == []


def test_a_retried_ingest_and_query_waits_and_answers_again(env, asks):
    storage, pdb, _, sent = env
    _statuses(pdb, ["running"])
    first = {**TEXT, "operation": "ingest_and_query", "question": "q", "wait_seconds": 0}
    assert client.post(PATH, json=first).json()["answer_status"] == "pending"
    _statuses(pdb, ["complete"])
    again = client.post(PATH, json={**first, "wait_seconds": 30}).json()    # same key: not ingested twice
    assert again["answer_status"] == "answered" and len(storage.uploads) == 1 and len(sent) == 1


@pytest.mark.parametrize("body, reason", [
    ({"operation": "query"}, "needs a `question`"),
    ({"operation": "query", "question": "q", "content": "x", "content_type": "text/plain"}, "no source material"),
    ({**TEXT, "operation": "ingest_and_query"}, "needs a `question`"),
    ({**TEXT, "source_version_ids": ["v1"]}, "goes with `query`"),
    ({"content": "x", "content_type": "text/plain"}, "`idempotency_key` is required"),
])
def test_operation_refusals(env, asks, body, reason):
    r = client.post(PATH, json=body)
    assert r.status_code == 400 and reason in r.json()["detail"]["reason"] and asks == []
