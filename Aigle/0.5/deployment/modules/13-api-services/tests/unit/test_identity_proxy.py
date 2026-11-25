"""
Unit tests for the people routes (dev plan v1.4 §4.5/§4.6): `/api/{v}/identity/*`
(identity.py) and `/api/{v}/search/people` (search.py).

Run from the 13-api-services/ directory:
    python -m pytest tests/unit/test_identity_proxy.py -v

get_current_user, the Module 25 client and the Module 04 storage client are
replaced through dependency_overrides, so no Keycloak, Module 25 or Module 04 is
needed -- this checks the gateway's own job: forward to the right Module 25 path
as the JWT's user, pass Module 25's errors through, and add presigned URLs for
videos and face images.
"""
import asyncio
import os
import sys

os.environ.setdefault("GATEWAY_AUTH_SERVICE_URL", "http://localhost:8800")
os.environ.setdefault("GATEWAY_KEYCLOAK_URL", "http://localhost:8080")
os.environ.setdefault("GATEWAY_KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
os.environ.setdefault("GATEWAY_KAFKA_TOPICS", '{"default": "default-topic"}')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import httpx
import pytest
from app.api.dependencies import get_current_user, get_http_client, get_storage_service
from app.core.config import get_settings
from app.main import app
from app.routers.search import get_hybrid_search_service, get_personal_db_service
from app.services.consent_check import ConsentCheck, get_consent_check
from fastapi import HTTPException
from fastapi.testclient import TestClient

FAKE_SUB = "jwt-verified-user-42"
V = get_settings().api_version
IDENTITY = f"/api/{V}/identity"

client = TestClient(app)


class FakePersonalDB:
    """Records calls; answers with `reply` or raises `error`."""

    def __init__(self):
        self.calls = []
        self.reply = {}
        self.error = None

    async def ensure_database(self, user_id):
        self.calls.append(("ensure", user_id))

    async def identity(self, method, path, user_id, json=None, params=None):
        self.calls.append((method, path, user_id, json, params))
        if self.error:
            raise self.error
        if callable(self.reply):
            return self.reply(method, path)
        return self.reply

    async def people_search(self, user_id, body):
        self.calls.append(("people", user_id, body))
        return self.reply


class FakeStorage:
    def __init__(self):
        self.presigned = []
        self.uploads = []
        self.batch_calls = []
        self.batch_resolves: set[str] = set()      # keys the batch call "resolves"; default none

    async def upload_identity_photo(self, identity_id, filename, data, content_type, user):
        self.uploads.append((identity_id, filename, data, user.user_id, user.branch_id))
        return {"image_path": f"identity/{user.branch_id}/{identity_id}.jpg", "image_version_id": "c7"}

    async def get_presign_url(self, key, version_id, user):
        self.presigned.append((key, version_id, user.user_id))
        if key.endswith("broken.jpg"):
            raise HTTPException(status_code=404, detail="not found")
        return f"https://s3/{key}?ref={version_id}"

    async def get_presign_urls_batch(self, items, user):
        self.batch_calls.append((tuple(items), user.user_id))
        return {(k, v): f"https://s3-batch/{k}?ref={v}" for k, v in items if k in self.batch_resolves}


class FakeFaceService:
    """Module 11's /embed-face, behind the gateway's shared httpx client."""

    def __init__(self):
        self.calls = []
        self.status = 200
        self.down = False

    async def post(self, url, files=None, timeout=None):
        self.calls.append((url, files["file"][1]))
        if self.down:
            raise httpx.ConnectError("refused")
        body = ({"embedding": [0.1] * 512, "bbox": [1, 2, 3, 4], "confidence": 0.98, "face_count": 2}
                if self.status == 200 else {"detail": "No face detected"})
        return httpx.Response(self.status, json=body)


class FakePrivacy:
    """Module 33's GET /consent, behind the consent check's httpx client."""

    def __init__(self):
        self.calls = []
        self.face_detection = True
        self.status = 200
        self.down = False

    async def get(self, url, headers=None, timeout=None):
        self.calls.append((url, headers["X-User-ID"]))
        if self.down:
            raise httpx.ConnectError("refused")
        return httpx.Response(self.status, request=httpx.Request("GET", url), json={"purposes": [
            {"id": "memory", "granted": True}, {"id": "face_detection", "granted": self.face_detection}]})


class FakeSearch:
    async def enrich_with_urls(self, items, user_dict):
        for item in items:
            if item.get("asset_path"):
                item["asset_url"] = f"https://s3/{item['asset_path']}/{item['filename']}"


@pytest.fixture
def fakes():
    db, storage, faces = FakePersonalDB(), FakeStorage(), FakeFaceService()
    privacy = FakePrivacy()
    overrides = {
        get_consent_check: lambda: ConsentCheck(privacy, get_settings()),
        get_current_user: lambda: {"sub": FAKE_SUB},
        get_personal_db_service: lambda: db,
        get_storage_service: lambda: storage,
        get_http_client: lambda: faces,
        get_hybrid_search_service: lambda: FakeSearch(),
    }
    previous = {k: app.dependency_overrides.get(k) for k in overrides}
    app.dependency_overrides.update(overrides)
    yield db, storage, faces
    for k, v in previous.items():
        if v is None:
            app.dependency_overrides.pop(k, None)
        else:
            app.dependency_overrides[k] = v


@pytest.fixture
def privacy(fakes):
    """The FakePrivacy behind this test's consent check."""
    return app.dependency_overrides[get_consent_check]().client


def _avatar(path="u/videos/v1/face_t1.jpg", ref="c1"):
    return {"face_id": "rep_t1", "kind": "representative", "image_path": path, "image_version_id": ref}


def test_list_forwards_paging_as_the_jwt_user_and_presigns_avatars(fakes):
    db, storage, _ = fakes
    db.reply = {"total": 2, "items": [{"identity_id": "p", "avatar": _avatar()},
                                      {"identity_id": "q", "avatar": None}]}
    r = client.get(f"{IDENTITY}/identities", params={"limit": 10, "offset": 20, "status": "pending"})
    assert r.status_code == 200
    assert db.calls == [("ensure", FAKE_SUB),
                        ("GET", "/identities", FAKE_SUB, None, {"limit": 10, "offset": 20, "status": "pending"})]
    body = r.json()
    assert body["items"][0]["avatar"]["image_url"] == "https://s3/u/videos/v1/face_t1.jpg?ref=c1"
    assert body["items"][1]["avatar"] is None
    assert storage.presigned == [("u/videos/v1/face_t1.jpg", "c1", FAKE_SUB)]


@pytest.mark.parametrize("method,path,body,downstream", [
    ("PATCH", "/p1", {"name": "Peter"}, "/p1"),
    ("DELETE", "/p1", None, "/p1"),
    ("POST", "/upload", {"name": "Peter"}, "/upload"),
    ("GET", "/file/v1/pending-persons", None, "/file/v1/pending-persons"),
    ("POST", "/file/v1/pending-person", {"trace_id": "t1"}, "/file/v1/pending-person"),
    ("POST", "/p1/bind/trace", {"version_id": "v1", "trace_id": "t1"}, "/p1/bind/trace"),
    ("POST", "/p1/bind/group", {"version_id": "v1", "group_id": "g1"}, "/p1/bind/group"),
    ("POST", "/p1/unbind", {"version_id": "v1", "trace_id": "t1"}, "/p1/unbind"),
    ("GET", "/p1/traces", None, "/p1/traces"),
    ("POST", "/p1/mergeinto", {"target_id": "p2"}, "/p1/mergeinto"),
    ("POST", "/merge/h1/undo", None, "/merge/h1/undo"),
    ("POST", "/merge/h1/redo", None, "/merge/h1/redo"),
    ("GET", "/p1", None, "/p1"),
    ("GET", "/p1/profile-image", None, "/p1/profile-image"),
])
def test_each_route_forwards_to_module_25(fakes, method, path, body, downstream):
    db, _, _ = fakes
    r = client.request(method, f"{IDENTITY}{path}", json=body)
    assert r.status_code in (200, 201), r.text
    assert db.calls[-1] == (method, downstream, FAKE_SUB, body, None)


def test_module_25_errors_pass_through(fakes):
    db, _, _ = fakes
    db.error = HTTPException(status_code=409, detail="history h1 expired at 2000-01-01")
    r = client.post(f"{IDENTITY}/merge/h1/undo")
    assert r.status_code == 409 and "expired" in r.json()["detail"]


def test_nested_face_images_are_presigned_once_each(fakes):
    db, storage, _ = fakes
    same = _avatar()
    db.reply = {"unassigned_groups": [{"group_id": "g", "representative": dict(same),
                                       "traces": [{"trace_id": "t1", "representative": dict(same)},
                                                  {"trace_id": "t2", "representative": _avatar("u/broken.jpg")},
                                                  {"trace_id": "t3", "representative": {
                                                      "face_id": "rep_t3", "image_path": None,
                                                      "image_version_id": None}}]}],
                "pending_identities": []}
    body = client.get(f"{IDENTITY}/file/v1/pending-persons").json()
    g = body["unassigned_groups"][0]
    assert g["representative"]["image_url"] == g["traces"][0]["representative"]["image_url"]
    assert g["traces"][1]["representative"]["image_url"] is None     # presign failed: no URL, no error
    assert "image_url" not in g["traces"][2]["representative"]       # upload had failed: nothing to sign
    assert sorted(storage.presigned) == [("u/broken.jpg", "c1", FAKE_SUB),
                                         ("u/videos/v1/face_t1.jpg", "c1", FAKE_SUB)]


def test_a_long_videos_face_images_are_presigned_a_few_at_a_time(fakes, monkeypatch):
    # A film's pending-persons carries thousands of trace pictures; fired all at once
    # they timed out queueing for the httpx pool and came back without URLs
    db, storage, _ = fakes
    in_flight, peak = 0, 0

    async def slow_presign(key, version_id, user):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.001)
        in_flight -= 1
        return f"https://s3/{key}?ref={version_id}"

    monkeypatch.setattr(storage, "get_presign_url", slow_presign)
    traces = [{"trace_id": f"t{i}", "representative": _avatar(f"u/videos/v1/face_t{i}.jpg")} for i in range(300)]
    db.reply = {"unassigned_groups": [{"group_id": "g", "representative": _avatar(), "traces": traces}],
                "pending_identities": []}
    body = client.get(f"{IDENTITY}/file/v1/pending-persons").json()
    signed = [t["representative"]["image_url"] for t in body["unassigned_groups"][0]["traces"]]
    assert all(signed) and len(signed) == 300
    assert 1 < peak <= get_settings().face_image_presign_concurrency


def test_images_the_batch_call_resolves_skip_the_per_file_fallback(fakes):
    """Module 04's /presign-urls-batch is tried first; an image it resolves never falls
    through to the one-at-a-time get_presign_url() path at all."""
    db, storage, _ = fakes
    storage.batch_resolves = {"u/videos/v1/face_t1.jpg"}
    db.reply = {"unassigned_groups": [{"group_id": "g", "representative": _avatar(), "traces": []}],
                "pending_identities": []}
    body = client.get(f"{IDENTITY}/file/v1/pending-persons").json()
    assert body["unassigned_groups"][0]["representative"]["image_url"] == "https://s3-batch/u/videos/v1/face_t1.jpg?ref=c1"
    assert storage.presigned == []                                  # never fell back
    assert storage.batch_calls and storage.batch_calls[0][0] == (("u/videos/v1/face_t1.jpg", "c1"),)


def test_only_the_batchs_misses_fall_back_to_per_file_presign(fakes):
    db, storage, _ = fakes
    storage.batch_resolves = {"u/videos/v1/face_t1.jpg"}             # t2 is a batch miss
    traces = [{"trace_id": "t2", "representative": _avatar("u/videos/v1/face_t2.jpg")}]
    db.reply = {"unassigned_groups": [{"group_id": "g", "representative": _avatar(), "traces": traces}],
                "pending_identities": []}
    body = client.get(f"{IDENTITY}/file/v1/pending-persons").json()
    g = body["unassigned_groups"][0]
    assert g["representative"]["image_url"] == "https://s3-batch/u/videos/v1/face_t1.jpg?ref=c1"
    assert g["traces"][0]["representative"]["image_url"] == "https://s3/u/videos/v1/face_t2.jpg?ref=c1"
    assert storage.presigned == [("u/videos/v1/face_t2.jpg", "c1", FAKE_SUB)]    # only the miss


def test_people_search_adds_video_and_face_urls(fakes):
    db, _, _ = fakes
    db.reply = {"query": "peter", "identities": [{"identity_id": "p"}], "total": 1, "results": [{
        "identity_id": "p", "version_id": "v1", "trace_id": "t1", "first_sec": 1.0, "last_sec": 3.5,
        "asset_path": "u/videos/v1", "filename": "v1.mp4", "keyframe": _avatar()}]}
    r = client.post(f"/api/{V}/search/people", json={"query": "peter"})
    assert r.status_code == 200
    assert db.calls == [("ensure", FAKE_SUB), ("people", FAKE_SUB, {"query": "peter", "limit": 50})]
    hit = r.json()["results"][0]
    assert hit["asset_url"] == "https://s3/u/videos/v1/v1.mp4"
    assert hit["keyframe"]["image_url"] == "https://s3/u/videos/v1/face_t1.jpg?ref=c1"


def test_people_search_rejects_an_empty_query(fakes):
    assert client.post(f"/api/{V}/search/people", json={"query": ""}).status_code == 422


def test_people_search_reads_the_callers_face_detection_consent(fakes, privacy):
    db, _, _ = fakes
    db.reply = {"query": "peter", "identities": [], "total": 0, "results": []}
    assert client.post(f"/api/{V}/search/people", json={"query": "peter"}).status_code == 200
    assert privacy.calls == [(f"{get_settings().privacy_compliance_url}/consent", FAKE_SUB)]


def test_people_search_is_denied_without_face_detection_consent(fakes, privacy):
    db, _, _ = fakes
    privacy.face_detection = False
    r = client.post(f"/api/{V}/search/people", json={"query": "peter"})
    assert r.status_code == 403
    assert r.json()["detail"] == {"error": "PERMISSION_DENIED", "message": "face_detection consent not granted"}
    assert db.calls == []                                           # Module 25 never asked


@pytest.mark.parametrize("failure", ["down", "500"])
def test_people_search_is_denied_when_consent_cant_be_read(fakes, privacy, failure):
    db, _, _ = fakes
    if failure == "down":
        privacy.down = True
    else:
        privacy.status = 500
    r = client.post(f"/api/{V}/search/people", json={"query": "peter"})
    assert r.status_code == 503 and r.json()["detail"]["error"] == "CONSENT_CHECK_UNAVAILABLE"
    assert db.calls == []


def test_people_search_fails_open_when_the_database_cant_be_initialized(fakes):
    """ensure_database never raises; Module 25 then answers an uninitialized user with no one."""
    db, _, _ = fakes

    async def init_failed(user_id):
        db.calls.append(("ensure failed", user_id))
    db.ensure_database = init_failed
    db.reply = {"query": "peter", "identities": [], "total": 0, "results": []}
    r = client.post(f"/api/{V}/search/people", json={"query": "peter"})
    assert r.status_code == 200 and r.json()["results"] == []


def test_people_search_passes_module_25_errors_through(fakes):
    db, _, _ = fakes

    async def boom(user_id, body):
        raise HTTPException(status_code=400, detail="Missing X-Branch-ID header")
    db.people_search = boom
    assert client.post(f"/api/{V}/search/people", json={"query": "peter"}).status_code == 400



# ── seed photo upload: Module 11 -> Module 04 -> Module 25 ──────────────────

JPEG = b"\xff\xd8\xff" + b"\x00" * 32


def _person(status_="active"):
    def reply(method, path):
        if method == "GET":
            return {"identity_id": "p1", "identity_status": status_}
        return {"identity_id": "p1", "identity_status": status_,
                "avatar": {"face_id": "seed_p1", "kind": "seed",
                           "image_path": f"identity/{FAKE_SUB}/p1.jpg", "image_version_id": "c7"}}
    return reply


def _upload():
    return client.post(f"{IDENTITY}/p1/profile-image", files={"photo": ("me.jpg", JPEG, "image/jpeg")})


def test_photo_runs_embed_then_store_then_record(fakes):
    db, storage, faces = fakes
    db.reply = _person()
    r = _upload()
    assert r.status_code == 200, r.text
    assert faces.calls == [("http://raptor-video-face-detection:8033/embed-face", JPEG)]
    assert storage.uploads == [("p1", "me.jpg", JPEG, FAKE_SUB, FAKE_SUB)]
    method, path, user, body, _ = db.calls[-1]
    assert (method, path, user) == ("POST", "/p1/profile-image", FAKE_SUB)
    assert body == {"image_path": f"identity/{FAKE_SUB}/p1.jpg", "image_version_id": "c7", "embedding": [0.1] * 512}
    out = r.json()
    assert out["avatar"]["image_url"] == f"https://s3/identity/{FAKE_SUB}/p1.jpg?ref=c7"
    assert out["photo"] == {"face_count": 2, "confidence": 0.98}


def test_photo_without_a_face_stores_nothing(fakes):
    db, storage, faces = fakes
    db.reply = _person()
    faces.status = 400
    r = _upload()
    assert r.status_code == 400 and "No face" in r.json()["detail"]
    assert storage.uploads == [] and [c[0] for c in db.calls] == ["ensure", "GET"]


@pytest.mark.parametrize("case,code", [("merged", 409), ("unknown", 404), ("face_service_down", 503), ("empty", 400)])
def test_photo_refusals_store_nothing(fakes, case, code):
    db, storage, faces = fakes
    db.reply = _person("merged" if case == "merged" else "active")
    if case == "unknown":
        db.error = HTTPException(status_code=404, detail="identity p1 not found")
    faces.down = case == "face_service_down"
    r = (client.post(f"{IDENTITY}/p1/profile-image", files={"photo": ("me.jpg", b"", "image/jpeg")})
         if case == "empty" else _upload())
    assert r.status_code == code
    assert storage.uploads == []
    assert not any(c[0] == "POST" for c in db.calls)
    if case in ("merged", "unknown", "empty"):
        assert faces.calls == []


# ── GD-11: a seed photo is face data -- face_detection consent first ────────

def test_photo_is_denied_without_face_detection_consent(fakes, privacy):
    db, storage, faces = fakes
    db.reply = _person()
    privacy.face_detection = False
    r = _upload()
    assert r.status_code == 403
    assert r.json()["detail"] == {"error": "PERMISSION_DENIED", "message": "face_detection consent not granted"}
    assert privacy.calls == [(f"{get_settings().privacy_compliance_url}/consent", FAKE_SUB)]
    # Nothing embedded, stored or recorded -- Module 25 not even asked.
    assert faces.calls == [] and storage.uploads == [] and db.calls == []


@pytest.mark.parametrize("failure", ["down", "500"])
def test_photo_is_denied_when_consent_cant_be_read(fakes, privacy, failure):
    db, storage, faces = fakes
    db.reply = _person()
    if failure == "down":
        privacy.down = True
    else:
        privacy.status = 500
    r = _upload()
    assert r.status_code == 503 and r.json()["detail"]["error"] == "CONSENT_CHECK_UNAVAILABLE"
    assert faces.calls == [] and storage.uploads == [] and db.calls == []
