"""CM-2: image moderation at ingestion (Module 10 -> Module 23 image guard).

image_moderation.moderate_image() runs against a fake Module 23 served over real
HTTP on 127.0.0.1, answering in the shapes module 23's README documents for
/infer/google-shieldgemma-2 and /infer/Llama-Guard-3-11B-Vision. The orchestrator
tests run the real join of description, OCR and moderation over a dict-backed Redis.

Run from the 10-image-processing/ directory:
    pytest tests/unit/test_image_moderation.py -v
"""
import asyncio
import json
import os

import pytest
import pytest_asyncio
from aiohttp import web
from conftest import make_request_message
from PIL import Image

pytestmark = pytest.mark.asyncio

SHIELDGEMMA = "/infer/google-shieldgemma-2"
LLAMA_GUARD = "/infer/Llama-Guard-3-11B-Vision"


def _shieldgemma(verdict, unsafe_probability):
    def policy(p):
        return {"unsafe_probability": p, "safe_probability": round(1 - p, 4),
                "verdict": "unsafe" if p > 0.5 else "safe", "policy_definition": "..."}
    return {"overall_verdict": verdict, "policies": {
        "dangerous": policy(0.01), "sexual": policy(unsafe_probability),
        "violence": policy(0.02)}}


class FakeModule23:
    """Answers each endpoint with whatever the test queues; records every call."""

    def __init__(self):
        self.calls = []
        self.answers = {}   # path -> (status, json body or text) or an async callable
        self.url = None
        self.flag_calls = []  # CM-5 / VIE01-408: every POST /moderation/flags body

    async def handle_flag(self, request):
        body = await request.json()
        self.flag_calls.append(body)
        return web.json_response({**body, "id": "00000000-0000-0000-0000-000000000000",
                                  "status": "flagged"}, status=201)

    async def handle(self, request):
        form = await request.post()
        image = form.get("image")
        self.calls.append((request.path, image.filename if image is not None else None,
                           len(image.file.read()) if image is not None else 0))
        answer = self.answers[request.path]
        if callable(answer):
            return await answer(request)
        status, body = answer
        if isinstance(body, str):
            return web.Response(status=status, text=body)
        return web.json_response(body, status=status)


@pytest_asyncio.fixture
async def module23(monkeypatch):
    fake = FakeModule23()
    app = web.Application()
    app.router.add_post(SHIELDGEMMA, fake.handle)
    app.router.add_post(LLAMA_GUARD, fake.handle)
    app.router.add_post("/moderation/flags", fake.handle_flag)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    host, port = runner.addresses[0][:2]
    fake.url = f"http://{host}:{port}"
    monkeypatch.setenv("GUARDRAIL_URL", fake.url)
    monkeypatch.delenv("IMAGE_MODERATION_MODEL", raising=False)
    monkeypatch.delenv("IMAGE_MODERATION_TIMEOUT", raising=False)
    monkeypatch.delenv("IMAGE_MODERATION_FAIL_MODE", raising=False)
    yield fake
    await runner.cleanup()


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "harbour.jpg"
    Image.new("RGB", (32, 24), (40, 120, 200)).save(path, "JPEG")
    return str(path)


# ── moderate_image(): verdicts ────────────────────────────────────────────────


async def test_clean_image_is_safe_with_policy_scores(module23, image):
    import image_moderation
    module23.answers[SHIELDGEMMA] = (200, _shieldgemma("safe", 0.03))

    flags = await image_moderation.moderate_image(image, "harbour.jpg")

    assert flags["verdict"] == "safe"
    assert flags["model"] == "shieldgemma-2"
    assert flags["policies"] == {"dangerous": 0.01, "sexual": 0.03, "violence": 0.02}
    assert isinstance(flags["latency_ms"], float) and flags["latency_ms"] >= 0
    assert flags["checked_at"]
    # The image itself went to Module 23 as multipart field "image".
    assert module23.calls == [(SHIELDGEMMA, "harbour.jpg", os.path.getsize(image))]


async def test_flagged_image_is_unsafe(module23, image):
    import image_moderation
    module23.answers[SHIELDGEMMA] = (200, _shieldgemma("unsafe", 0.97))

    flags = await image_moderation.moderate_image(image, "harbour.jpg")

    assert flags["verdict"] == "unsafe"
    assert flags["policies"]["sexual"] == 0.97


async def test_llama_guard_model_uses_its_endpoint_and_categories(module23, image, monkeypatch):
    import image_moderation
    monkeypatch.setenv("IMAGE_MODERATION_MODEL", "llama-guard-3-vision")
    module23.answers[LLAMA_GUARD] = (200, {"label": "unsafe", "categories": ["S1", "S11"], "raw": "unsafe\nS1,S11"})

    flags = await image_moderation.moderate_image(image, "harbour.jpg")

    assert (flags["verdict"], flags["model"], flags["categories"]) == ("unsafe", "llama-guard-3-vision", ["S1", "S11"])
    assert [c[0] for c in module23.calls] == [LLAMA_GUARD]


async def test_latency_measures_the_module23_call(module23, image):
    import image_moderation

    async def slow(_request):
        await asyncio.sleep(0.2)
        return web.json_response(_shieldgemma("safe", 0.01))
    module23.answers[SHIELDGEMMA] = slow

    flags = await image_moderation.moderate_image(image, "harbour.jpg")

    assert flags["latency_ms"] >= 200


# ── moderate_image(): no verdict ──────────────────────────────────────────────


@pytest.mark.parametrize("answer, reason", [
    ((503, {"detail": "shieldgemma-2 unavailable: not enough GPU memory"}), "HTTP 503"),
    ((502, {"detail": "shieldgemma-2 inference failed"}), "HTTP 502"),
    ((200, "not json"), "JSONDecodeError"),
    ((200, {"label": "safe"}), "expected overall_verdict and policies"),
    ((200, {"overall_verdict": "maybe", "policies": {}}), "expected overall_verdict and policies"),
])
async def test_module23_failure_is_no_verdict(module23, image, answer, reason):
    import image_moderation
    module23.answers[SHIELDGEMMA] = answer

    with pytest.raises(image_moderation.ModerationUnavailable) as exc:
        await image_moderation.moderate_image(image, "harbour.jpg")

    assert reason in str(exc.value)
    flags = exc.value.flags
    assert (flags["verdict"], flags["model"]) == ("error", "shieldgemma-2")
    assert reason in flags["error"]
    assert "latency_ms" in flags and "checked_at" in flags


async def test_timeout_is_no_verdict(module23, image, monkeypatch):
    import image_moderation
    monkeypatch.setenv("IMAGE_MODERATION_TIMEOUT", "0.2")

    async def hang(_request):
        await asyncio.sleep(5)
        return web.json_response(_shieldgemma("safe", 0.01))
    module23.answers[SHIELDGEMMA] = hang

    with pytest.raises(image_moderation.ModerationUnavailable) as exc:
        await image_moderation.moderate_image(image, "harbour.jpg")
    assert exc.value.flags["error"].startswith("TimeoutError")


async def test_unreachable_module23_is_no_verdict(image, monkeypatch):
    import image_moderation
    monkeypatch.setenv("GUARDRAIL_URL", "http://127.0.0.1:9")   # discard port: connection refused
    monkeypatch.delenv("IMAGE_MODERATION_MODEL", raising=False)

    with pytest.raises(image_moderation.ModerationUnavailable) as exc:
        await image_moderation.moderate_image(image, "harbour.jpg")
    assert exc.value.flags["verdict"] == "error"


async def test_unknown_model_setting_is_no_verdict_and_calls_nothing(module23, image, monkeypatch):
    import image_moderation
    monkeypatch.setenv("IMAGE_MODERATION_MODEL", "shieldgemma")

    with pytest.raises(image_moderation.ModerationUnavailable) as exc:
        await image_moderation.moderate_image(image, "harbour.jpg")
    assert "IMAGE_MODERATION_MODEL must be one of" in exc.value.flags["error"]
    assert module23.calls == []


async def test_unreadable_image_is_no_verdict_and_calls_nothing(module23, tmp_path):
    import image_moderation

    with pytest.raises(image_moderation.ModerationUnavailable) as exc:
        await image_moderation.moderate_image(str(tmp_path / "gone.jpg"), "gone.jpg")
    assert exc.value.flags["error"].startswith("FileNotFoundError")
    assert module23.calls == []


@pytest.mark.parametrize("value, expected", [(None, False), ("false", False), ("true", True), ("TRUE", True), ("1", False)])
async def test_enabled_flag(monkeypatch, value, expected):
    import image_moderation
    if value is None:
        monkeypatch.delenv("IMAGE_MODERATION_ENABLED", raising=False)
    else:
        monkeypatch.setenv("IMAGE_MODERATION_ENABLED", value)
    assert image_moderation.enabled() is expected


@pytest.mark.parametrize("value, expected", [(None, False), ("open", False), ("closed", True), ("CLOSED", True)])
async def test_fail_mode(monkeypatch, value, expected):
    import image_moderation
    if value is None:
        monkeypatch.delenv("IMAGE_MODERATION_FAIL_MODE", raising=False)
    else:
        monkeypatch.setenv("IMAGE_MODERATION_FAIL_MODE", value)
    assert image_moderation.fail_closed() is expected


# ── report_flags(): CM-5 / VIE01-408 ──────────────────────────────────────────


async def test_an_unsafe_llama_guard_verdict_reports_each_category(module23, image, monkeypatch):
    import image_moderation
    monkeypatch.setenv("IMAGE_MODERATION_MODEL", "llama-guard-3-vision")
    module23.answers[LLAMA_GUARD] = (200, {"label": "unsafe", "categories": ["S3", "S10"]})
    flags = await image_moderation.moderate_image(image, "harbour.jpg")

    await image_moderation.report_flags(flags, "assets/harbour.jpg", "v1", "branch-1")

    assert {c["category"] for c in module23.flag_calls} == {"S3", "S10"}
    assert all(c["asset_path"] == "assets/harbour.jpg" for c in module23.flag_calls)
    assert all(c["version_id"] == "v1" for c in module23.flag_calls)
    assert all(c["branch_id"] == "branch-1" for c in module23.flag_calls)
    assert all(c["model"] == "llama-guard-3-vision" for c in module23.flag_calls)
    assert all(c["moment_id"] is None and c["chunk_id"] is None for c in module23.flag_calls)


async def test_an_unsafe_shieldgemma_verdict_reports_only_categories_over_threshold(module23, image):
    import image_moderation
    module23.answers[SHIELDGEMMA] = (200, _shieldgemma("unsafe", 0.97))
    flags = await image_moderation.moderate_image(image, "harbour.jpg")

    await image_moderation.report_flags(flags, "assets/harbour.jpg", "v1", "branch-1")

    # dangerous=0.01, sexual=0.97, violence=0.02 -- only sexual crosses 0.5
    assert [c["category"] for c in module23.flag_calls] == ["sexual"]
    assert module23.flag_calls[0]["score"] == 0.97


async def test_a_safe_verdict_reports_nothing(module23, image):
    import image_moderation
    module23.answers[SHIELDGEMMA] = (200, _shieldgemma("safe", 0.02))
    flags = await image_moderation.moderate_image(image, "harbour.jpg")

    await image_moderation.report_flags(flags, "assets/harbour.jpg", "v1", "branch-1")

    assert module23.flag_calls == []


async def test_report_flags_failure_does_not_raise(module23, image, monkeypatch):
    import image_moderation
    module23.answers[SHIELDGEMMA] = (200, _shieldgemma("unsafe", 0.9))
    flags = await image_moderation.moderate_image(image, "harbour.jpg")
    monkeypatch.setenv("GUARDRAIL_URL", "http://127.0.0.1:1")  # nothing listening

    await image_moderation.report_flags(flags, "assets/harbour.jpg", "v1", "branch-1")  # must not raise


# ── orchestrator: the description / OCR / moderation join ────────────────────


class DictRedis:
    """Enough of redis.Redis for RedisStateManager, backed by a dict."""

    def __init__(self):
        self.data = {}

    def setex(self, key, _ttl, value):
        self.data[key] = value

    def get(self, key):
        return self.data.get(key)

    def delete(self, key):
        self.data.pop(key, None)

    def expire(self, key, _ttl):
        return key in self.data

    def close(self):
        pass


class RecordingProducer:
    def __init__(self):
        self.sent = []

    async def send(self, topic, message):
        self.sent.append((topic, message))


@pytest.fixture
def job(handler, tmp_path, image, monkeypatch):
    """A handler with real Redis state and merge files under tmp_path, plus the
    request message for one image already on (fake NFS) disk."""
    import config
    handler.redis_manager.client = DictRedis()
    monkeypatch.setattr(config, "TEMP_FILE_DIR", str(tmp_path))

    message = make_request_message(asset_path="image/jpg/harbour", version_id="v7", correlation_id="corr-7")
    message.update({"timestamp": "2026-11-03T00:00:00Z", "target_service": "image_orchestrator_service",
                    "message_type": "REQUEST", "priority": "MEDIUM", "retry_count": 0, "ttl": 3600})
    message["payload"]["action"] = "image_processing"
    message["payload"]["parameters"].update({"primary_filename": "harbour.jpg", "temp_file_path": image})

    desc = tmp_path / "desc.json"
    desc.write_text(json.dumps({"description": "Two people at a harbour."}))
    ocr = tmp_path / "ocr.json"
    ocr.write_text(json.dumps({"extracted_text": "PIER 7"}))
    results = {
        "description": {"payload": {"status": "success", "results": {"parameters": {"description_result_path": str(desc)}}}},
        "ocr": {"payload": {"status": "success", "results": {"parameters": {"ocr_result_path": str(ocr)}}}},
    }
    return handler, message, results


def _result(results, kind):
    return {"correlation_id": "corr-7", "message_type": "RESPONSE", **results[kind]}


def _indexer_requests(producer):
    from config import KAFKA_TOPIC_INDEXER_REQUEST
    return [m for t, m in producer.sent if t == KAFKA_TOPIC_INDEXER_REQUEST]


def _merged_entries(request):
    with open(request["payload"]["parameters"]["merged_result_path"], encoding="utf-8") as f:
        return json.load(f)


async def _deliver(handler, producer, results, *kinds):
    from config import KAFKA_TOPIC_DESCRIPTION_RESULT, KAFKA_TOPIC_OCR_RESULT
    topics = {"description": KAFKA_TOPIC_DESCRIPTION_RESULT, "ocr": KAFKA_TOPIC_OCR_RESULT}
    for kind in kinds:
        await handler.handle_service_result(_result(results, kind), producer, topics[kind])


async def test_disabled_flag_makes_no_module23_call_and_merges_as_before(job, monkeypatch):
    import image_moderation
    handler, message, results = job
    monkeypatch.delenv("IMAGE_MODERATION_ENABLED", raising=False)

    async def must_not_run(*_a, **_kw):
        raise AssertionError("Module 23 called with moderation disabled")
    monkeypatch.setattr(image_moderation, "moderate_image", must_not_run)
    producer = RecordingProducer()

    await handler.handle_image_processing(message, producer)
    await _deliver(handler, producer, results, "description", "ocr")

    [request] = _indexer_requests(producer)
    summary, text = _merged_entries(request)
    assert "moderation_flags" not in summary["payload"] and "moderation_flags" not in text["payload"]
    assert handler.seaweedfs_client.moderation_calls == []


async def test_disabled_flag_makes_no_http_call_to_module23(job, module23, monkeypatch):
    handler, message, results = job
    monkeypatch.setenv("IMAGE_MODERATION_ENABLED", "false")
    producer = RecordingProducer()

    await handler.handle_image_processing(message, producer)
    await _deliver(handler, producer, results, "description", "ocr")

    assert module23.calls == []
    assert len(_indexer_requests(producer)) == 1


async def test_verdict_reaches_module04_and_the_summary_entry(job, module23, monkeypatch):
    handler, message, results = job
    monkeypatch.setenv("IMAGE_MODERATION_ENABLED", "true")
    module23.answers[SHIELDGEMMA] = (200, _shieldgemma("unsafe", 0.97))
    producer = RecordingProducer()

    await handler.handle_image_processing(message, producer)   # moderation done before description/OCR
    assert _indexer_requests(producer) == []                   # waits for description and OCR
    await _deliver(handler, producer, results, "description", "ocr")

    [request] = _indexer_requests(producer)
    summary, text = _merged_entries(request)
    flags = summary["payload"]["moderation_flags"]
    assert flags["verdict"] == "unsafe" and flags["policies"]["sexual"] == 0.97
    assert "moderation_flags" not in text["payload"]
    # An unsafe image is still indexed: the verdict is recorded, not enforced.
    assert summary["payload"]["summary"] == "Two people at a harbour."
    assert handler.seaweedfs_client.moderation_calls == [("u1", "b1", "image/jpg/harbour", "v7", flags)]
    assert [c[0] for c in module23.calls] == [SHIELDGEMMA]


async def test_merge_waits_for_a_verdict_that_comes_last(job, monkeypatch):
    import image_moderation
    handler, message, results = job
    monkeypatch.setenv("IMAGE_MODERATION_ENABLED", "true")
    release = asyncio.Event()

    async def late_verdict(_path, _filename):
        await release.wait()
        return {"verdict": "safe", "model": "shieldgemma-2", "policies": {}, "latency_ms": 5.0, "checked_at": "t"}
    monkeypatch.setattr(image_moderation, "moderate_image", late_verdict)
    producer = RecordingProducer()

    request_task = asyncio.create_task(handler.handle_image_processing(message, producer))
    await asyncio.sleep(0.05)
    await _deliver(handler, producer, results, "description", "ocr")
    assert _indexer_requests(producer) == []          # both results in, still waiting for the verdict
    assert not request_task.done()                    # and the request isn't finished (not committed)

    release.set()
    await request_task

    [request] = _indexer_requests(producer)
    assert _merged_entries(request)[0]["payload"]["moderation_flags"]["verdict"] == "safe"


async def test_fail_open_indexes_with_an_error_verdict(job, module23, monkeypatch):
    handler, message, results = job
    monkeypatch.setenv("IMAGE_MODERATION_ENABLED", "true")
    module23.answers[SHIELDGEMMA] = (503, {"detail": "shieldgemma-2 unavailable"})
    producer = RecordingProducer()

    await handler.handle_image_processing(message, producer)
    await _deliver(handler, producer, results, "description", "ocr")

    [request] = _indexer_requests(producer)
    flags = _merged_entries(request)[0]["payload"]["moderation_flags"]
    assert flags["verdict"] == "error" and "HTTP 503" in flags["error"]
    assert handler.seaweedfs_client.moderation_calls[0][4] == flags
    assert handler.seaweedfs_client.status_calls == []          # not failed


async def test_fail_closed_fails_the_job_and_never_merges(job, module23, monkeypatch):
    from config import KAFKA_TOPIC_FINAL_RESULT
    handler, message, results = job
    monkeypatch.setenv("IMAGE_MODERATION_ENABLED", "true")
    monkeypatch.setenv("IMAGE_MODERATION_FAIL_MODE", "closed")
    module23.answers[SHIELDGEMMA] = (503, {"detail": "shieldgemma-2 unavailable"})
    producer = RecordingProducer()

    await handler.handle_image_processing(message, producer)
    await _deliver(handler, producer, results, "description", "ocr")   # arrive after the failure

    assert _indexer_requests(producer) == []
    [error] = [m for t, m in producer.sent if t == KAFKA_TOPIC_FINAL_RESULT]
    assert error["payload"]["error"]["code"] == "MODERATION_UNAVAILABLE"
    assert handler.seaweedfs_client.status_calls == [("u1", "b1", "image/jpg/harbour", "v7", "failed")]
    assert handler.seaweedfs_client.moderation_calls == []
    assert handler.redis_manager.get_state("corr-7")["step"] == "error"


async def test_verdict_for_a_job_that_already_failed_is_dropped(job, monkeypatch):
    import image_moderation
    handler, message, _results = job
    monkeypatch.setenv("IMAGE_MODERATION_ENABLED", "true")
    release = asyncio.Event()

    async def late_verdict(_path, _filename):
        await release.wait()
        return {"verdict": "safe", "model": "shieldgemma-2", "policies": {}, "latency_ms": 5.0, "checked_at": "t"}
    monkeypatch.setattr(image_moderation, "moderate_image", late_verdict)
    producer = RecordingProducer()

    request_task = asyncio.create_task(handler.handle_image_processing(message, producer))
    await asyncio.sleep(0.05)
    from config import KAFKA_TOPIC_DESCRIPTION_RESULT
    failed = {"correlation_id": "corr-7", "message_type": "ERROR",
              "payload": {"status": "error", "error": {"message": "VLM out of memory"}}}
    await handler.handle_service_result(failed, producer, KAFKA_TOPIC_DESCRIPTION_RESULT)
    release.set()
    await request_task

    assert _indexer_requests(producer) == []
    assert handler.seaweedfs_client.moderation_calls == []
    assert handler.redis_manager.get_state("corr-7")["step"] == "error"


async def test_a_redelivered_result_does_not_merge_twice(job, module23, monkeypatch):
    handler, message, results = job
    monkeypatch.setenv("IMAGE_MODERATION_ENABLED", "true")
    module23.answers[SHIELDGEMMA] = (200, _shieldgemma("safe", 0.01))
    producer = RecordingProducer()

    await handler.handle_image_processing(message, producer)
    await _deliver(handler, producer, results, "description", "ocr", "ocr")

    assert len(_indexer_requests(producer)) == 1


async def test_a_state_from_before_cm2_does_not_wait_for_moderation(job, monkeypatch):
    """A job started by the previous version (no moderation_done in its state)
    still merges once description and OCR are in."""
    handler, message, results = job
    monkeypatch.setenv("IMAGE_MODERATION_ENABLED", "true")
    handler.redis_manager.set_state("corr-7", {
        "original_message": message, "temp_file_path": message["payload"]["parameters"]["temp_file_path"],
        "description_result": None, "ocr_result": None, "save_result": None,
        "step": "parallel_processing", "branch_id": "b1", "created_at": 0})
    producer = RecordingProducer()

    await _deliver(handler, producer, results, "description", "ocr")

    [request] = _indexer_requests(producer)
    assert "moderation_flags" not in _merged_entries(request)[0]["payload"]
