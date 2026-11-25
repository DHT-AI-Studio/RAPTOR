"""
GD-1 / VIE01-246: erasure fans out to Module 35 (voice-clone speaker
embeddings) and reports Module 34 (voice recordings) as `skipped` while
GD_MODULE_34_URL is empty -- and a skipped store doesn't hold the job back.

GD-10 / VIE01-416: once GD_MODULE_34_URL is set, erasure deletes the user's
recordings through Module 34's DELETE /internal/gdpr/user/{id}, and a
Module 34 failure is a retryable partial failure like any other store.

Mocked: Keycloak, the Kafka tombstone, the DPO email and the six existing
stores are stubbed on the orchestrator; Modules 34 and 35 are real HTTP
calls through httpx.MockTransport, so their URLs and X-Internal-Token are
checked.

Run from the 33-privacy-compliance-service/ directory:
    python -m pytest tests/unit/test_erasure_voice_stores.py -v
"""
from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest
from app.core.config import settings
from app.models.job import (
    STORE_KEYS,
    ErasureJob,
    JobStatus,
    StoreStatus,
    initial_store_statuses,
)
from app.services.erasure_orchestrator import ErasureOrchestrator

USER = "alice"
M34 = "http://m34.test"
M35 = "http://m35.test"


class _FakeRedis:
    async def sadd(self, *args):
        pass

    async def expire(self, *args):
        pass


class _FakeAudit:
    def __init__(self):
        self.events: list = []

    async def emit(self, event, *, fail_closed=False):
        self.events.append((event, fail_closed))


def _job(statuses: dict[str, StoreStatus]) -> ErasureJob:
    return ErasureJob(job_id="job-1", user_id=USER, requested_at=datetime.now(timezone.utc),
                      requested_by=USER, status=JobStatus.PENDING, store_statuses=statuses)


@pytest.fixture
def make_orchestrator(monkeypatch):
    monkeypatch.setattr(settings, "module_34_url", "")
    monkeypatch.setattr(settings, "module_35_url", M35)
    monkeypatch.setattr(settings, "erasure_retry_delay_seconds", 0)

    def _make(job: ErasureJob, m35_status: int = 200, m34_status: int = 200):
        orch = ErasureOrchestrator(pool=None, redis=_FakeRedis(), kafka_producer=None)
        orch.audit = _FakeAudit()
        calls: dict = {"m34": [], "m35": [], "updates": [], "hard_deleted": False, "dpo": None,
                       "audit": orch.audit.events}

        def voice_stores(request: httpx.Request) -> httpx.Response:
            if request.url.host == "m34.test":
                calls["m34"].append(request)
                return httpx.Response(m34_status, json={"user_id": USER, "recordings_deleted": 3})
            calls["m35"].append(request)
            return httpx.Response(m35_status, json={"user_id": USER, "speaker_embeddings_deleted": 2})

        orch.http = httpx.AsyncClient(transport=httpx.MockTransport(voice_stores))

        async def get_job(job_id):
            return job

        async def update_job(job_id, **fields):
            calls["updates"].append(fields)

        async def noop(*args, **kwargs):
            pass

        async def hard_delete(user_id):
            calls["hard_deleted"] = True

        async def notify_dpo(user_id, job_id, failed):
            calls["dpo"] = failed

        async def start(job_id, user_id):  # no legal hold: the run starts
            await update_job(job_id, status=JobStatus.IN_PROGRESS)
            return True

        monkeypatch.setattr(orch, "get_job", get_job)
        monkeypatch.setattr(orch, "_start_unless_held", start)
        monkeypatch.setattr(orch, "_update_job", update_job)
        monkeypatch.setattr(orch, "_disable_keycloak_account", noop)
        monkeypatch.setattr(orch, "_hard_delete_keycloak_account", hard_delete)
        monkeypatch.setattr(orch, "_publish_tombstone", noop)
        monkeypatch.setattr(orch, "_notify_dpo", notify_dpo)
        for key in ("04", "25", "26", "03", "23", "02"):
            monkeypatch.setattr(orch, f"_delete_module_{key}", noop)
        return orch, calls

    return _make


def _final(calls) -> dict:
    """Merge every _update_job call into the job's final state."""
    state: dict = {}
    for fields in calls["updates"]:
        state.update(fields)
    return state


def test_store_keys_include_modules_34_and_35():
    assert {"module_34", "module_35"} <= set(STORE_KEYS)
    assert initial_store_statuses()["module_35"] == StoreStatus.PENDING


@pytest.mark.asyncio
async def test_module_35_is_erased_and_module_34_skipped(make_orchestrator):
    orch, calls = make_orchestrator(_job(initial_store_statuses()))
    await orch.run("job-1")

    [request] = calls["m35"]
    assert request.method == "DELETE"
    assert str(request.url) == f"{M35}/internal/gdpr/user/{USER}"
    assert request.headers["X-Internal-Token"] == settings.internal_token

    final = _final(calls)
    assert final["store_statuses"]["module_35"] == StoreStatus.DELETED
    assert final["store_statuses"]["module_34"] == StoreStatus.SKIPPED
    assert final["status"] == JobStatus.COMPLETED  # skipped doesn't block completion
    assert "skipped" in final["notes"] and "module_34" in final["notes"]
    assert calls["hard_deleted"] is True


@pytest.mark.asyncio
async def test_module_35_failure_leaves_partial_failure(make_orchestrator):
    orch, calls = make_orchestrator(_job(initial_store_statuses()), m35_status=500)
    await orch.run("job-1")

    assert len(calls["m35"]) == settings.erasure_max_attempts
    final = _final(calls)
    assert final["store_statuses"]["module_35"] == StoreStatus.FAILED
    assert final["status"] == JobStatus.PARTIAL_FAILURE
    assert calls["dpo"] == ["module_35"]
    assert calls["hard_deleted"] is False  # account kept until every store confirms


@pytest.mark.asyncio
async def test_module_35_skipped_when_url_unset(make_orchestrator, monkeypatch):
    monkeypatch.setattr(settings, "module_35_url", "")
    orch, calls = make_orchestrator(_job(initial_store_statuses()))
    await orch.run("job-1")

    assert calls["m35"] == []
    final = _final(calls)
    assert final["store_statuses"]["module_35"] == StoreStatus.SKIPPED
    assert final["status"] == JobStatus.COMPLETED


@pytest.mark.asyncio
async def test_retry_reruns_failed_module_35(make_orchestrator):
    statuses = {key: StoreStatus.DELETED for key in STORE_KEYS}
    statuses.update(module_34=StoreStatus.SKIPPED, module_35=StoreStatus.FAILED)
    orch, calls = make_orchestrator(_job(statuses))
    await orch.retry("job-1")

    assert len(calls["m35"]) == 1
    final = _final(calls)
    assert final["store_statuses"]["module_35"] == StoreStatus.DELETED
    assert final["status"] == JobStatus.COMPLETED
    assert calls["hard_deleted"] is True


@pytest.mark.asyncio
async def test_module_34_is_erased_once_wired(make_orchestrator, monkeypatch):
    monkeypatch.setattr(settings, "module_34_url", M34)
    orch, calls = make_orchestrator(_job(initial_store_statuses()))
    await orch.run("job-1")

    [request] = calls["m34"]
    assert request.method == "DELETE"
    assert str(request.url) == f"{M34}/internal/gdpr/user/{USER}"
    assert request.headers["X-Internal-Token"] == settings.internal_token

    final = _final(calls)
    assert final["store_statuses"]["module_34"] == StoreStatus.DELETED
    assert final["store_statuses"]["module_35"] == StoreStatus.DELETED
    assert final["status"] == JobStatus.COMPLETED
    assert "module_34" not in (final.get("notes") or "")  # not reported as skipped
    assert calls["hard_deleted"] is True


@pytest.mark.asyncio
async def test_module_34_failure_is_a_partial_failure_and_retry_finishes_it(make_orchestrator, monkeypatch):
    monkeypatch.setattr(settings, "module_34_url", M34)
    orch, calls = make_orchestrator(_job(initial_store_statuses()), m34_status=503)
    await orch.run("job-1")

    assert len(calls["m34"]) == settings.erasure_max_attempts
    final = _final(calls)
    assert final["store_statuses"]["module_34"] == StoreStatus.FAILED
    assert final["status"] == JobStatus.PARTIAL_FAILURE
    assert calls["dpo"] == ["module_34"]
    assert calls["hard_deleted"] is False  # account kept until every store confirms

    # Module 34 is back: retry reruns only the failed store and completes the job.
    orch, calls = make_orchestrator(_job(final["store_statuses"]))
    await orch.retry("job-1")

    assert len(calls["m34"]) == 1 and calls["m35"] == []
    final = _final(calls)
    assert final["store_statuses"]["module_34"] == StoreStatus.DELETED
    assert final["status"] == JobStatus.COMPLETED
    assert calls["hard_deleted"] is True
