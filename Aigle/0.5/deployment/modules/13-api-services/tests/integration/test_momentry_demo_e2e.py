"""
MTR-10 / VIE01-397: the Momentry demo script, end to end through the gateway.

The ticket's demo script, all six legs:
  1. Upload -> Register -> Module 11 face detection (a non-blocking branch of
     video analysis) -> Module 25 stores the traces and links the same face
     across two videos -> name that person -> POST /search/people returns
     segments with a representative image and a playable clip.
  2. Thumbnail returns the right frame at an arbitrary timestamp.
  3. Unregister -> People Search returns nothing for that video -> Register
     fast path -> its segments come back.
  5. Seed photo -> a new video with that person is matched automatically.
  4. Merge two people -> undo -> the original state is restored.
  6. Compound query "[person] discussing [topic]" -> Module 18 tags it
     people+semantic -> Module 21 runs PeopleAgent with vector search and
     fuses them with RRF (MTR-25 / VIE01-396).
Leg 5 runs before leg 4 because the merge needs a second person with
segments of their own, which leg 5 creates; leg 6 runs last. People Search
needs the user's face_detection consent (MTR-24 / VIE01-395, off by
default): the fixture checks it is refused without, then grants it.

The ticket names POST /personal/index/face-detection-events for chain
assembly + clustering; in the code Module 11 tracks and clusters itself and
writes the result to Module 25's POST /personal/index/face-traces, which
does the cross-video matching. This test follows the code.

Test videos are generated here (PyAV): one of the two face photos already
in Module 11's unit-test fixtures, centred on a background that is red for
0-2 s, green for 2-4 s and blue for 4-6 s -- so a thumbnail's colour says
which second it was taken from -- with a silent audio track. Armstrong is
in videos A and B (cross-video match), the astronaut in the seed photo and
video C.

Needs the full stack, Module 11's face detection included, and takes
several minutes, so it only runs when asked to:
    TEST_MOMENTRY_E2E=1 TEST_KEYCLOAK_ADMIN_USER=... TEST_KEYCLOAK_ADMIN_PASSWORD=... \\
        python -m pytest tests/integration/test_momentry_demo_e2e.py -v
from the 13-api-services/ directory. A throwaway user is created and, at
the end, erased with GDPR self-erasure (every store) and deleted.
"""
from __future__ import annotations

import io
import os
import pathlib
import time
import uuid

import httpx
import pytest

GATEWAY_URL = os.environ.get("TEST_GATEWAY_URL", "http://localhost:8012")
KEYCLOAK_URL = os.environ.get("TEST_KEYCLOAK_URL", "http://localhost:8080")
KEYCLOAK_REALM = os.environ.get("TEST_KEYCLOAK_REALM", "dhtsolution")
KEYCLOAK_CLIENT_ID = os.environ.get("TEST_KEYCLOAK_CLIENT_ID", "raptor")
KEYCLOAK_ADMIN_USER = os.environ.get("TEST_KEYCLOAK_ADMIN_USER", "dhtsolution")
KEYCLOAK_ADMIN_PASSWORD = os.environ.get("TEST_KEYCLOAK_ADMIN_PASSWORD", "")
API = f"{GATEWAY_URL}/api/{os.environ.get('TEST_API_VERSION', '0.5')}"
# Per video: analysis + face detection on a 6 s clip.
TIMEOUT = float(os.environ.get("TEST_MOMENTRY_TIMEOUT_SEC", "900"))

FIXTURES = pathlib.Path(__file__).resolve().parents[3] / "11-video-processing/tests/unit/fixtures"
ARMSTRONG = FIXTURES / "face_armstrong.jpg"
ASTRONAUT = FIXTURES / "face_astronaut.jpg"
BANDS = ((200, 40, 40), (40, 200, 40), (40, 40, 200))   # red, green, blue
BAND_SEC = 2.0


def _reachable() -> bool:
    try:
        httpx.get(f"{GATEWAY_URL}/docs", timeout=3.0)
        return True
    except httpx.HTTPError:
        return False


pytestmark = pytest.mark.skipif(
    os.environ.get("TEST_MOMENTRY_E2E") != "1" or not KEYCLOAK_ADMIN_PASSWORD or not _reachable(),
    reason="opt-in live test: set TEST_MOMENTRY_E2E=1 and TEST_KEYCLOAK_ADMIN_PASSWORD, gateway reachable",
)


# ── Test data ───────────────────────────────────────────────────────────────

def _make_video(face_jpg: pathlib.Path, fps: int = 25, size: tuple[int, int] = (1280, 720)) -> bytes:
    """A 6 s H.264/AAC MP4: the face, enlarged to 360 px, on the RGB bands."""
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    from PIL import Image

    width, height = size
    face = Image.open(face_jpg).convert("RGB").resize((360, 360))
    frames = []
    for color in BANDS:
        canvas = Image.new("RGB", size, color)
        canvas.paste(face, ((width - 360) // 2, (height - 360) // 2))
        frames.append(np.asarray(canvas))

    buf = io.BytesIO()
    out = av.open(buf, "w", format="mp4")
    video = out.add_stream("libx264", rate=fps)
    video.width, video.height, video.pix_fmt = width, height, "yuv420p"
    audio = out.add_stream("aac", rate=48000)
    audio.layout = "mono"
    for i in range(int(len(BANDS) * BAND_SEC * fps)):
        frame = av.VideoFrame.from_ndarray(frames[int(i / fps // BAND_SEC)], format="rgb24")
        for packet in video.encode(frame):
            out.mux(packet)
    for packet in video.encode():
        out.mux(packet)
    samples = 0
    while samples < 48000 * len(BANDS) * BAND_SEC:
        silence = av.AudioFrame.from_ndarray(np.zeros((1, 1024), np.float32), format="fltp", layout="mono")
        silence.sample_rate, silence.pts = 48000, samples
        samples += 1024
        for packet in audio.encode(silence):
            out.mux(packet)
    for packet in audio.encode():
        out.mux(packet)
    out.close()
    return buf.getvalue()


# ── Keycloak and gateway helpers ────────────────────────────────────────────

def _admin_headers() -> dict:
    resp = httpx.post(
        f"{KEYCLOAK_URL}/realms/master/protocol/openid-connect/token",
        data={"grant_type": "password", "client_id": "admin-cli",
              "username": KEYCLOAK_ADMIN_USER, "password": KEYCLOAK_ADMIN_PASSWORD},
        timeout=10.0,
    )
    resp.raise_for_status()
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _create_user(username: str, password: str) -> str:
    h, base = _admin_headers(), f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}"
    httpx.post(f"{base}/users", headers=h, timeout=10.0, json={
        "username": username, "firstName": "Momentry", "lastName": "E2E", "email": f"{username}@example.com",
        "emailVerified": True, "enabled": True, "requiredActions": [],
        "credentials": [{"type": "password", "value": password, "temporary": False}],
    }).raise_for_status()
    user_id = httpx.get(f"{base}/users", headers=h, params={"username": username, "exact": "true"},
                        timeout=10.0).json()[0]["id"]
    client_uuid = httpx.get(f"{base}/clients", headers=h, params={"clientId": KEYCLOAK_CLIENT_ID},
                            timeout=10.0).json()[0]["id"]
    role = httpx.get(f"{base}/clients/{client_uuid}/roles/user_standard", headers=h, timeout=10.0).json()
    httpx.post(f"{base}/users/{user_id}/role-mappings/clients/{client_uuid}", headers=h, json=[role],
               timeout=10.0).raise_for_status()
    return user_id


def _login(username: str, password: str) -> dict:
    resp = httpx.post(f"{API}/sso/login", timeout=10.0,
                      data={"username": username, "password": password, "realm_name": KEYCLOAK_REALM,
                            "client_id": KEYCLOAK_CLIENT_ID})
    resp.raise_for_status()
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _wait(what: str, check, timeout: float = TIMEOUT, every: float = 5.0):
    """Poll check() until it returns something truthy; fail after timeout."""
    deadline = time.monotonic() + timeout
    while True:
        result = check()
        if result:
            return result
        if time.monotonic() > deadline:
            pytest.fail(f"timed out after {timeout:.0f}s waiting for {what}")
        time.sleep(every)


class Gateway:
    def __init__(self, auth: dict):
        self.client = httpx.Client(base_url=API, headers=auth, timeout=60.0)

    def ok(self, method: str, path: str, **kwargs) -> dict:
        resp = self.client.request(method, path, **kwargs)
        assert resp.status_code < 300, f"{method} {path} -> {resp.status_code} {resp.text}"
        return resp.json()

    def upload_and_register(self, filename: str, content: bytes) -> dict:
        """Upload (status 'uploaded'), Register, then wait until the video is
        active and Module 11's face-detection branch has reported."""
        up = self.ok("POST", "/asset/fileupload", files={"primary_file": (filename, content, "video/mp4")})
        video = {"filename": filename, "asset_path": up["asset_path"], "version_id": up["version_id"]}
        reg = self.ok("POST", "/processing/process-file",
                      json={"asset_path": video["asset_path"], "version_id": video["version_id"]})
        video["correlation_id"] = reg["correlation_id"]
        video["face_detection"] = _wait(f"face detection on {filename}", lambda: self._face_detection(video))
        _wait(f"{filename} to become active", lambda: self._status(video) == "active")
        return video

    def _face_detection(self, video: dict) -> dict | None:
        resp = self.client.get(f"/processing/cache/video/{video['correlation_id']}")
        if resp.status_code != 200:
            return None
        state = (resp.json().get("value") or {}).get("face_detection") or {}
        if state.get("status") == "failed":
            pytest.fail(f"face detection failed for {video['filename']}: {state.get('error')}")
        return state if state.get("status") == "completed" else None

    def _status(self, video: dict) -> str | None:
        commits = self.ok("GET", "/asset/users/commits",
                          params={"keyword": video["filename"], "page_size": 50})["commits"]
        status = next((c["status"] for c in commits if c["version_id"] == video["version_id"]), None)
        if status == "failed":
            pytest.fail(f"analysis of {video['filename']} failed")
        return status

    def people(self, query: str) -> dict:
        return self.ok("POST", "/search/people", json={"query": query})

    def segment_versions(self, query: str) -> set[str]:
        return {r["version_id"] for r in self.people(query)["results"]}


@pytest.fixture(scope="module")
def demo():
    tag, password = uuid.uuid4().hex[:10], f"Mtr10-{uuid.uuid4().hex[:12]}!"
    username = f"mtr10-e2e-{tag}"
    user_id = _create_user(username, password)
    gw = Gateway(_login(username, password))
    state = {"gw": gw, "tag": tag}
    try:
        # People Search is refused until the user opts in to face_detection (biometric, default off).
        denied = gw.client.post("/search/people", json={"query": "anyone"})
        assert denied.status_code == 403 and "face_detection consent not granted" in denied.text, denied.text
        gw.ok("PUT", "/gdpr/consent/face_detection", json={"granted": True})
        state["a"] = gw.upload_and_register(f"mtr10_a_{tag}.mp4", _make_video(ARMSTRONG))
        state["b"] = gw.upload_and_register(f"mtr10_b_{tag}.mp4", _make_video(ARMSTRONG))
        yield state
    finally:
        # GDPR self-erasure clears the user's data in every store; then the account goes.
        resp = gw.client.post("/gdpr/erasure")
        if resp.status_code < 300 and resp.json().get("job_id"):
            job = resp.json()["job_id"]
            _wait("cleanup erasure", lambda: httpx.get(f"{API}/gdpr/erasure/jobs/{job}", timeout=10.0)
                  .json().get("status") not in ("pending", "in_progress"), timeout=300)
        httpx.delete(f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/users/{user_id}",
                     headers=_admin_headers(), timeout=10.0)
        gw.client.close()


# ── Leg 1: upload -> register -> faces -> name -> People Search ─────────────

def test_leg1_face_detection_cross_video_match_naming_and_people_search(demo):
    gw, a, b = demo["gw"], demo["a"], demo["b"]

    # The same face in A and B became one pending person holding both videos.
    pending = gw.ok("GET", f"/identity/file/{b['version_id']}/pending-persons")["pending_identities"]
    assert len(pending) == 1, pending
    person = pending[0]
    assert person["identity_status"] == "pending" and person["video_count"] == 2, person

    name = f"Momentry Demo {demo['tag']}"
    named = gw.ok("PATCH", f"/identity/{person['identity_id']}", json={"name": name, "identity_status": "active"})
    assert (named["name"], named["identity_status"]) == (name, "active")
    demo["p1"] = {"identity_id": person["identity_id"], "name": name}

    found = gw.people(name)
    assert [(i["identity_id"], i["exact"]) for i in found["identities"]] == [(person["identity_id"], True)]
    assert {r["version_id"] for r in found["results"]} == {a["version_id"], b["version_id"]}
    for segment in found["results"]:
        assert segment["first_sec"] <= segment["last_sec"] <= len(BANDS) * BAND_SEC + 0.5, segment
        # Representative image and playable clip, both fetchable.
        assert segment["keyframe_path"] == segment["keyframe"]["image_path"], segment
        image = httpx.get(segment["keyframe"]["image_url"], timeout=30.0)
        assert image.status_code == 200 and image.content[:3] == b"\xff\xd8\xff", segment
        clip = httpx.get(segment["asset_url"], headers={"Range": "bytes=0-1023"}, timeout=30.0)
        assert clip.status_code in (200, 206) and b"ftyp" in clip.content[:64], segment


# ── Leg 2: thumbnail at an arbitrary timestamp ──────────────────────────────

def test_leg2_thumbnail_is_the_frame_at_the_requested_second(demo):
    from PIL import Image

    gw, a = demo["gw"], demo["a"]
    path = f"/asset/thumbnail/{a['asset_path']}/{a['version_id']}"
    for second, band in ((1.0, 0), (2.7, 1), (5.3, 2)):
        resp = gw.client.get(path, params={"second": second})
        assert resp.status_code == 200 and resp.headers["content-type"].startswith("image/jpeg"), resp.text
        corner = Image.open(io.BytesIO(resp.content)).convert("RGB").getpixel((10, 10))
        assert max(range(3), key=lambda c: corner[c]) == band, (second, corner)
    assert gw.client.get(path, params={"second": 600}).status_code == 400  # beyond the video


# ── Leg 3: unregister hides the video, Register fast path brings it back ────

def test_leg3_unregister_hides_segments_and_register_restores_them(demo):
    gw, a, b = demo["gw"], demo["a"], demo["b"]
    name = demo.get("p1", {}).get("name") or pytest.skip("needs leg 1's named person")

    unreg = gw.ok("POST", f"/asset/unregister/{a['asset_path']}/{a['version_id']}")
    assert unreg["status"] == "unregistered"
    assert gw.segment_versions(name) == {b["version_id"]}

    reg = gw.ok("POST", "/processing/process-file", json={"asset_path": a["asset_path"], "version_id": a["version_id"]})
    assert reg["outcome"] == "registered" and reg["metadata"]["status"] == "active", reg  # no re-analysis
    assert gw.segment_versions(name) == {a["version_id"], b["version_id"]}


# ── Leg 5: seed photo -> a new video is matched automatically ───────────────

def test_leg5_seed_photo_matches_a_new_video(demo):
    gw = demo["gw"]
    name = f"Momentry Seed {demo['tag']}"
    person = gw.ok("POST", "/identity/upload", json={"name": name})
    seeded = gw.ok("POST", f"/identity/{person['identity_id']}/profile-image",
                   files={"photo": ("seed.jpg", ASTRONAUT.read_bytes(), "image/jpeg")})
    assert seeded["photo"]["face_count"] >= 1, seeded

    c = gw.upload_and_register(f"mtr10_c_{demo['tag']}.mp4", _make_video(ASTRONAUT))
    demo["c"], demo["p2"] = c, {"identity_id": person["identity_id"], "name": name}

    traces = gw.ok("GET", f"/identity/{person['identity_id']}/traces")["traces"]
    assert {t["version_id"] for t in traces} == {c["version_id"]}, traces
    assert gw.segment_versions(name) == {c["version_id"]}
    # A different person: the seed match didn't pull Armstrong's videos in.
    if "p1" in demo:
        assert c["version_id"] not in gw.segment_versions(demo["p1"]["name"])


# ── Leg 4: merge two people, undo within 24 h ───────────────────────────────

def test_leg4_merge_then_undo_restores_the_original_state(demo):
    gw = demo["gw"]
    if "p1" not in demo or "p2" not in demo:
        pytest.skip("needs the people from legs 1 and 5")
    p1, p2, a, b, c = demo["p1"], demo["p2"], demo["a"], demo["b"], demo["c"]
    before = gw.ok("GET", f"/identity/{p2['identity_id']}")

    merged = gw.ok("POST", f"/identity/{p2['identity_id']}/mergeinto",
                   json={"target_id": p1["identity_id"], "keep_source": True})
    assert merged["groups_moved"], merged
    assert gw.ok("GET", f"/identity/{p2['identity_id']}")["identity_status"] == "merged"
    assert gw.segment_versions(p1["name"]) == {a["version_id"], b["version_id"], c["version_id"]}

    undone = gw.ok("POST", f"/identity/merge/{merged['merge_id']}/undo")
    assert undone["history_state"] == "reverted" and not undone["skipped"], undone
    # MTR-23: undoing it again is refused and changes nothing
    assert gw.client.post(f"/identity/merge/{merged['merge_id']}/undo").status_code == 409
    after = gw.ok("GET", f"/identity/{p2['identity_id']}")
    assert (after["identity_status"], after["name"]) == (before["identity_status"], before["name"])
    assert gw.segment_versions(p1["name"]) == {a["version_id"], b["version_id"]}
    assert gw.segment_versions(p2["name"]) == {c["version_id"]}


# ── Leg 6: "[person] discussing [topic]" -> PeopleAgent + vector search -> RRF ─

def test_leg6_compound_people_and_topic_query_fuses_people_results(demo):
    """POST /a2a/query (mode=direct, Module 21's pipeline). The generated videos have no
    speech or text, so the topic itself finds nothing in them; what this checks is the
    routing the ticket asks for: the person named in the question is found by PeopleAgent
    (as the end user, through People Search) and their segments come back in the RRF-fused
    sources. Needs Modules 18 and 21 with the people-agent service running."""
    gw = demo["gw"]
    p1 = demo.get("p1") or pytest.skip("needs leg 1's named person")
    a, b = demo["a"], demo["b"]

    resp = gw.client.post("/a2a/query", timeout=300.0, json={
        "question": f"{p1['name']} discussing the weather", "mode": "direct", "top_k": 10})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    people = body.get("people") or []
    assert people, f"PeopleAgent found no one -- people intent not detected, or no people listed: {body}"
    assert {p["identity_name"] for p in people} == {p1["name"]}
    assert {p["version_id"] for p in people} <= {a["version_id"], b["version_id"]}
    assert all(p["first_sec"] <= p["last_sec"] for p in people), people
    # the person's segments are among the fused, ranked sources
    assert {s.get("version_id") for s in body["sources"]} & {a["version_id"], b["version_id"]}, body["sources"]
