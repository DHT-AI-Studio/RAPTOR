"""
GD-2 / VIE01-247: the export's ZIP layout and manifest match the ticket.

Mocked: every store fetcher, the job row and the S3 upload are stubbed on
the orchestrator; the ZIP parts it would upload are captured and checked.

Run from the 33-privacy-compliance-service/ directory:
    python -m pytest tests/unit/test_export_orchestrator.py -v
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx
import pytest
from app.core.config import settings
from app.models.job import ExportJob, JobStatus
from app.services import export_orchestrator as mod
from app.services.export_orchestrator import ExportOrchestrator

USER = "user-123"
FILES = [{"name": "report.pdf", "type": "application/pdf", "size": 1234, "uploaded_at": "2026-10-01T09:00:00+08:00",
          "download_url": "https://files.example/report.pdf?sig=x"}]
M25 = {"documents": [{"id": "d1"}], "entities": [{"id": "e1"}], "moments": [{"id": "m1"}]}
M26 = {"sessions": [{"session_id": "s-1", "turns": [{"q": "hi"}]}, {"session_id": "../etc/passwd", "turns": []}],
       "longterm": [{"fact": "likes tea"}], "multimedia": [{"file": "a.png"}]}
BENCH = {"runs": [{"run_id": "r1"}], "schemas": [{"id": "s1", "definition": {"test_cases": []}}]}
MOD_FLAGS = [{"id": "f1", "category": "nudity", "status": "flagged"}]
VOICE = {"note": "metadata only", "pending_sessions": 0,
         "recordings": [{"session_id": "rt-1", "created_at": "2026-10-01T09:00:00+00:00",
                         "expires_at": "2026-12-30T09:00:00+00:00", "duration_s": 12, "size_bytes": 384044}]}


@pytest.fixture
def make(monkeypatch):
    monkeypatch.setattr(settings, "module_34_url", "http://m34.test")
    monkeypatch.setattr(settings, "module_35_url", "http://m35.test")
    monkeypatch.setattr(settings, "export_max_zip_bytes", 524288000)

    def _make(fail=(), speaker_embeddings=3):
        orch = ExportOrchestrator(pool=None)
        captured: dict = {"updates": [], "events": []}

        class _Audit:
            async def emit(self, event, *, fail_closed=False):
                captured["events"].append(event)

        orch.audit = _Audit()
        job = ExportJob(job_id="job-1", user_id=USER, requested_at=datetime.now(timezone.utc),
                        requested_by=USER, status=JobStatus.PENDING)

        def stub(name, value):
            async def fetch(user_id):
                assert user_id == USER
                if name in fail:
                    raise RuntimeError(f"{name} down")
                return value
            return fetch

        async def get_job(job_id):
            return job

        async def update_job(job_id, **fields):
            captured["updates"].append(fields)

        async def upload(job_id, user_id, exported_at, parts, part_names):
            captured["parts"] = {name: dict(part) for part, name in zip(parts, part_names)}
            return [f"https://s3.example/{n}" for n in part_names], 42

        monkeypatch.setattr(orch, "get_job", get_job)
        monkeypatch.setattr(orch, "_update_job", update_job)
        monkeypatch.setattr(orch, "_upload_parts", upload)
        monkeypatch.setattr(orch, "_fetch_module_06", stub("module_06", {"username": "alice"}))
        monkeypatch.setattr(orch, "_fetch_module_04", stub("module_04", FILES))
        monkeypatch.setattr(orch, "_fetch_module_25", stub("module_25", M25))
        monkeypatch.setattr(orch, "_fetch_module_26", stub("module_26", M26))
        monkeypatch.setattr(orch, "_fetch_module_03", stub("module_03", BENCH))
        monkeypatch.setattr(orch, "_fetch_module_23", stub("module_23", MOD_FLAGS))
        monkeypatch.setattr(orch, "_fetch_module_34", stub("module_34", VOICE))
        monkeypatch.setattr(orch, "_fetch_module_35", stub("module_35", speaker_embeddings))
        return orch, captured

    return _make


def _final(captured) -> dict:
    state: dict = {}
    for fields in captured["updates"]:
        state.update(fields)
    return state


@pytest.mark.asyncio
async def test_zip_layout_matches_the_ticket(make):
    orch, captured = make()
    await orch.run("job-1")

    [(name, entries)] = captured["parts"].items()
    assert name == "export.zip"
    assert sorted(entries) == sorted([
        "manifest.json",
        "identity/keycloak_profile.json",
        "files/index.json",
        "content/documents.json", "content/entities.json", "content/moments.json",
        "memory/long_term.json", "memory/sessions/s-1.json", "memory/sessions/_etc_passwd.json",
        "memory/multimedia.json",
        "benchmark/runs.json",
        "moderation/flags.json",
        "voice/recordings.json",
    ])
    assert json.loads(entries["files/index.json"]) == FILES
    assert json.loads(entries["memory/long_term.json"]) == M26["longterm"]
    assert json.loads(entries["memory/sessions/s-1.json"]) == M26["sessions"][0]
    assert json.loads(entries["benchmark/runs.json"]) == BENCH
    assert json.loads(entries["content/entities.json"]) == M25["entities"]
    assert json.loads(entries["moderation/flags.json"]) == MOD_FLAGS
    assert json.loads(entries["voice/recordings.json"]) == VOICE


@pytest.mark.asyncio
async def test_files_are_metadata_only_never_bytes(make):
    orch, captured = make()
    await orch.run("job-1")
    entries = captured["parts"]["export.zip"]
    assert not any(path.startswith("files/") and path != "files/index.json" for path in entries)


@pytest.mark.asyncio
async def test_manifest_has_every_ticket_field(make):
    orch, captured = make()
    await orch.run("job-1")
    manifest = json.loads(captured["parts"]["export.zip"]["manifest.json"])

    assert manifest["export_schema_version"] == "1.0"
    assert manifest["user_id"] == USER
    assert datetime.fromisoformat(manifest["exported_at"])
    assert manifest["gdpr_article"] == "Art. 20 Right to Data Portability"
    assert manifest["ccpa_section"] == "§ 1798.100"
    assert manifest["stores_included"] == ["module_06", "module_04", "module_25", "module_26", "module_03",
                                           "module_23", "module_34", "module_35"]
    assert manifest["stores_failed"] == []
    assert manifest["parts"] == ["export.zip"]
    assert manifest["speaker_embeddings_cached"] == 3
    assert set(manifest["omitted"]) == {"audit/violations.json"}

    final = _final(captured)
    assert final["status"] == JobStatus.COMPLETED
    assert final["download_urls"] == ["https://s3.example/export.zip"]


@pytest.mark.asyncio
async def test_module_35_not_deployed_is_explained_not_failed(make, monkeypatch):
    monkeypatch.setattr(settings, "module_35_url", "")
    orch, captured = make()
    await orch.run("job-1")
    manifest = json.loads(captured["parts"]["export.zip"]["manifest.json"])
    assert manifest["speaker_embeddings_cached"] is None
    assert "manifest.speaker_embeddings_cached" in manifest["omitted"]
    assert "module_35" not in manifest["stores_included"] + manifest["stores_failed"]
    assert _final(captured)["status"] == JobStatus.COMPLETED


@pytest.mark.asyncio
async def test_failed_store_is_omitted_and_named(make):
    orch, captured = make(fail=("module_26", "module_35"))
    await orch.run("job-1")
    entries = captured["parts"]["export.zip"]
    manifest = json.loads(entries["manifest.json"])
    assert manifest["stores_failed"] == ["module_26", "module_35"]
    assert manifest["speaker_embeddings_cached"] is None
    assert not any(path.startswith("memory/") for path in entries)
    assert "content/documents.json" in entries  # the rest still exported
    final = _final(captured)
    assert final["status"] == JobStatus.PARTIAL_FAILURE
    assert final["notes"] == "stores_failed: module_26, module_35"


@pytest.mark.asyncio
async def test_split_export_names_parts_and_manifest_lists_them(make, monkeypatch):
    monkeypatch.setattr(settings, "export_max_zip_bytes", 400)
    orch, captured = make()
    await orch.run("job-1")

    names = list(captured["parts"])
    assert len(names) > 1
    assert names == [f"export-part-{i}.zip" for i in range(1, len(names) + 1)]
    manifest = json.loads(captured["parts"]["export-part-1.zip"]["manifest.json"])
    assert manifest["parts"] == names
    assert all("manifest.json" not in captured["parts"][n] for n in names[1:])
    # Every section still lands in exactly one part.
    assert sum(len(p) for p in captured["parts"].values()) == 13


def test_part_names():
    assert mod._part_names(1) == ["export.zip"]
    assert mod._part_names(3) == ["export-part-1.zip", "export-part-2.zip", "export-part-3.zip"]


def test_poll_response_has_the_tickets_download_url():
    now = datetime.now(timezone.utc)
    row = {"job_id": "job-1", "user_id": USER, "requested_at": now, "requested_by": USER, "status": "completed",
           "notes": None, "completed_at": now, "size_bytes": 42, "expires_at": now,
           "download_urls": json.dumps(["https://s3.example/export-part-1.zip", "https://s3.example/export-part-2.zip"])}
    job = mod._row_to_export_job(row)
    assert job.download_url == "https://s3.example/export-part-1.zip"
    assert job.download_urls[1].endswith("export-part-2.zip")
    assert mod._row_to_export_job({**row, "download_urls": None}).download_url is None


# ── Upload: object keys and download filenames ──────────────────────────────

class _S3:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.uploads = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def head_bucket(self, Bucket):
        pass

    async def upload_file(self, path, bucket, key):
        self.uploads.append(key)


@pytest.mark.asyncio
async def test_upload_keys_ttl_and_download_filenames(monkeypatch):
    monkeypatch.setattr(settings, "s3_endpoint", "http://seaweedfs-s3:8333")
    # An underscore-containing host is exactly the real case this needs to handle
    # (botocore's own client construction refuses one -- see
    # _presign_path_style_get's docstring) -- so the fixture value isn't the
    # tidy "files.example" one might otherwise pick.
    monkeypatch.setattr(settings, "s3_public_url", "http://raptor_open_api.example:8333")
    monkeypatch.setattr(settings, "aws_access_key", "AKIATEST")
    monkeypatch.setattr(settings, "aws_secret_key", "secret")
    orch = ExportOrchestrator(pool=None)
    clients: list[_S3] = []
    monkeypatch.setattr(orch.s3_session, "client", lambda *a, **k: clients.append(_S3(**k)) or clients[-1])
    exported_at = datetime(2026, 10, 19, 5, 6, 7, tzinfo=timezone.utc)
    parts = [[("manifest.json", b"{}")], [("files/index.json", b"[]")]]

    urls, size = await orch._upload_parts("job-1", USER, exported_at, parts, mod._part_names(2))

    # Only the upload itself goes through a real client (the internal
    # endpoint); download links are signed directly, with no client ever
    # constructed against the public endpoint (see
    # _presign_path_style_get's docstring for why: boto3's own client
    # construction would reject this fixture's underscore-containing host).
    [uploader] = clients
    assert uploader.kwargs["endpoint_url"] == "http://seaweedfs-s3:8333"
    assert uploader.uploads == ["gdpr-exports/job-1/export-part-1.zip", "gdpr-exports/job-1/export-part-2.zip"]
    assert len(urls) == 2
    for i, url in enumerate(urls, start=1):
        assert url.startswith(
            f"http://raptor_open_api.example:8333/{settings.s3_bucket}/gdpr-exports/job-1/export-part-{i}.zip?"
        )
        assert (
            f"response-content-disposition=attachment%3B%20filename%3D%22"
            f"gdpr-export-{USER}-20261019T050607Z-part-{i}.zip%22"
        ) in url
        assert f"X-Amz-Expires={int(settings.export_ttl_hours * 3600)}" in url
        assert "X-Amz-Signature=" in url
    assert size > 0


# ── GD-10 / VIE01-416: voice recordings from Module 34 ───────────────────


@pytest.mark.asyncio
async def test_module_34_not_wired_omits_voice_and_says_why(make, monkeypatch):
    monkeypatch.setattr(settings, "module_34_url", "")
    orch, captured = make()
    await orch.run("job-1")
    entries = captured["parts"]["export.zip"]
    manifest = json.loads(entries["manifest.json"])
    assert "voice/recordings.json" not in entries
    assert "GD_MODULE_34_URL" in manifest["omitted"]["voice/recordings.json"]
    assert "module_34" not in manifest["stores_included"] + manifest["stores_failed"]
    assert _final(captured)["status"] == JobStatus.COMPLETED


@pytest.mark.asyncio
async def test_module_34_down_is_a_partial_failure(make):
    orch, captured = make(fail=("module_34",))
    await orch.run("job-1")
    entries = captured["parts"]["export.zip"]
    manifest = json.loads(entries["manifest.json"])
    assert "voice/recordings.json" not in entries
    assert manifest["stores_failed"] == ["module_34"]
    assert _final(captured)["status"] == JobStatus.PARTIAL_FAILURE


@pytest.mark.asyncio
async def test_module_23_down_is_a_partial_failure(make):
    """CM-5 / VIE01-408: moderation_flags is fetched like every other store
    -- a failure there omits moderation/flags.json and is named, same as
    every other per-store failure (distinct from audit/violations.json,
    which is never attempted at all -- see _OMITTED_SECTIONS)."""
    orch, captured = make(fail=("module_23",))
    await orch.run("job-1")
    entries = captured["parts"]["export.zip"]
    manifest = json.loads(entries["manifest.json"])
    assert "moderation/flags.json" not in entries
    assert manifest["stores_failed"] == ["module_23"]
    assert _final(captured)["status"] == JobStatus.PARTIAL_FAILURE


@pytest.mark.asyncio
async def test_fetch_module_34_calls_the_recordings_endpoint_and_adds_the_note(monkeypatch):
    monkeypatch.setattr(settings, "module_34_url", "http://m34.test")
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, str(request.url), request.headers.get("X-Internal-Token")))
        return httpx.Response(200, json={"user_id": USER, "pending_sessions": 1,
                                         "recordings": VOICE["recordings"]})

    orch = ExportOrchestrator(pool=None)
    orch.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    body = await orch._fetch_module_34(USER)
    await orch.close()
    assert seen == [("GET", f"http://m34.test/internal/gdpr/user/{USER}/recordings", settings.internal_token)]
    assert body["recordings"] == VOICE["recordings"] and body["pending_sessions"] == 1
    assert "not in this export" in body["note"] and "speaker_embeddings_cached" in body["note"]



# ── V05-10 / VIE01-364: CF-4 gdpr.export audit event ──────────────────────


@pytest.mark.asyncio
async def test_finished_export_writes_one_gdpr_export_event(make):
    orch, captured = make()
    await orch.run("job-1")
    [event] = captured["events"]
    assert (event.event_type, event.outcome, event.resource_id) == ("gdpr.export", "allow", "job-1")
    assert event.metadata["action"] == "export" and event.metadata["stores_failed"] == []
    assert event.metadata["parts"] == 1 and USER not in event.model_dump_json()


@pytest.mark.asyncio
async def test_partial_export_is_an_error_event(make):
    orch, captured = make(fail=("module_26",))
    await orch.run("job-1")
    [event] = captured["events"]
    assert event.outcome == "error" and event.metadata["stores_failed"] == ["module_26"]


@pytest.mark.asyncio
async def test_failed_upload_is_an_error_event(make, monkeypatch):
    orch, captured = make()

    async def broken(*args):
        raise RuntimeError("S3 down")

    monkeypatch.setattr(orch, "_upload_parts", broken)
    await orch.run("job-1")
    [event] = captured["events"]
    assert event.outcome == "error" and event.metadata["upload_failed"] is True
