"""Face detection as the video orchestrator's third parallel branch (dev plan
v1.4 §4.3): sent alongside video/audio analysis, but not part of the completion
condition -- its result or failure is only recorded in the Redis state, and a
face-detection failure must never fail the video (which, since VIE01-255, would
also write status=failed back to Module 04).

Run from the 11-video-processing/ directory:
    pytest tests/unit/ -v
"""
import pytest
from conftest import make_request_message

pytestmark = pytest.mark.asyncio

TEMP = "/tmp/media_processing/video_processing/tmp_x.mp4"


class FakeRedisState:
    """Dict-backed stand-in for RedisStateManager, so tests can read back what was written."""

    def __init__(self):
        self.states = {}

    def set_state(self, cid, state):
        self.states[cid] = dict(state)
        return True

    def get_state(self, cid):
        s = self.states.get(cid)
        return dict(s) if s is not None else None

    def update_state(self, cid, updates):
        if cid not in self.states:
            return False
        self.states[cid].update(updates)
        return True


class FakeProducer:
    def __init__(self, fail_topics=()):
        self.sent = []
        self.fail_topics = fail_topics

    async def send(self, topic, value):
        if topic in self.fail_topics:
            raise RuntimeError("broker unavailable")
        self.sent.append((topic, value))


@pytest.fixture
def orch(handler):
    handler.redis_manager = FakeRedisState()
    handler.redis_manager.set_state("corr-1", {
        "original_message": make_request_message(asset_path="u1/videos/clip"),
        "temp_file_path": TEMP,
        "video_analysis_result": None,
        "audio_analysis_result": None,
        "step": "parallel_analysis",
        "branch_id": "b1",
    })
    return handler


async def _fan_out(orch, producer):
    await orch.send_parallel_analysis_requests(
        make_request_message(asset_path="u1/videos/clip"), producer, TEMP, "f.mp4", moments=[])


async def test_face_detection_can_be_switched_off(orch, monkeypatch):
    import kafka_handler
    monkeypatch.setattr(kafka_handler, "FACE_DETECTION_ENABLED", False)
    producer = FakeProducer()
    await _fan_out(orch, producer)
    assert [t for t, _ in producer.sent] == ["video-analysis-requests", "audio-analysis-requests"]
    assert "face_detection" not in orch.redis_manager.get_state("corr-1")


async def test_fan_out_sends_face_detection_as_third_request(orch):
    producer = FakeProducer()
    await _fan_out(orch, producer)

    assert [t for t, _ in producer.sent] == [
        "video-analysis-requests", "audio-analysis-requests", "video-face-detection-requests"]
    face = producer.sent[2][1]
    assert face["target_service"] == "video_face_detection_service"
    assert face["correlation_id"] == "corr-1"
    params = face["payload"]["parameters"]
    assert params["video_file_path"] == TEMP
    assert (params["asset_path"], params["version_id"]) == ("u1/videos/clip", "v1")
    # so the face service can fetch its own copy once the orchestrator deletes TEMP
    assert params["download_headers"] == {"X-User-ID": "u1", "X-Branch-ID": "b1"}
    assert orch.redis_manager.get_state("corr-1")["face_detection"] == {"status": "processing"}


async def test_face_request_send_failure_does_not_break_fan_out(orch):
    producer = FakeProducer(fail_topics=("video-face-detection-requests",))
    await _fan_out(orch, producer)   # must not raise
    assert [t for t, _ in producer.sent] == ["video-analysis-requests", "audio-analysis-requests"]


def _face_result(message_type="RESPONSE", payload=None):
    return {"correlation_id": "corr-1", "message_type": message_type, "payload": payload or {}}


async def test_face_detection_error_does_not_fail_the_video(orch, monkeypatch):
    failed = []

    async def record_failure(*args, **kwargs):
        failed.append(args)

    monkeypatch.setattr(orch, "handle_processing_error", record_failure)
    producer = FakeProducer()
    error = _face_result("ERROR", {"status": "error", "error": {"code": "FACE_DETECTION_FAILED", "message": "decode failed"}})
    await orch.handle_service_result(error, producer, "video-face-detection-results")

    assert failed == []
    assert producer.sent == []                               # no error sent to the final-results topic
    assert orch.seaweedfs_client.status_calls == []          # no status=failed written to Module 04
    state = orch.redis_manager.get_state("corr-1")
    assert state["face_detection"]["status"] == "failed"
    assert state["step"] == "parallel_analysis"


async def test_face_detection_result_recorded_without_driving_completion(orch, monkeypatch):
    async def must_not_run(*args, **kwargs):
        raise AssertionError("a face result must not drive the completion check")

    monkeypatch.setattr(orch, "check_analysis_completion", must_not_run)
    result = _face_result(payload={"status": "success", "results": {"face_detections": 16}})
    await orch.handle_service_result(result, FakeProducer(), "video-face-detection-results")

    state = orch.redis_manager.get_state("corr-1")
    assert state["face_detection"] == {"status": "completed", "results": {"face_detections": 16}}


async def test_face_detection_skipped_for_consent_is_recorded_as_skipped(orch):
    """GD-11: the face service answers {"skipped": reason} without detecting anything."""
    result = _face_result(payload={"status": "success", "results": {"skipped": "no_consent"}})
    await orch.handle_service_result(result, FakeProducer(), "video-face-detection-results")

    state = orch.redis_manager.get_state("corr-1")
    assert state["face_detection"] == {"status": "skipped", "results": {"skipped": "no_consent"}}
    assert state["step"] == "parallel_analysis"           # the video itself carries on


async def test_face_result_after_completion_is_still_recorded(orch):
    orch.redis_manager.set_state("corr-1", {"step": "complete", "summary": "s", "text": [], "branch_id": "b1"})
    result = _face_result(payload={"status": "success", "results": {"face_detections": 3}})
    await orch.handle_service_result(result, FakeProducer(), "video-face-detection-results")

    state = orch.redis_manager.get_state("corr-1")
    assert state["step"] == "complete"
    assert state["face_detection"]["results"] == {"face_detections": 3}


async def test_completion_keeps_a_face_result_that_arrived_first(orch):
    """Both whole-state rewrites -- save (-> indexing) and Module 25's index completion
    (-> complete) -- keep a face result that came back before them."""
    orch.redis_manager.update_state("corr-1", {"face_detection": {"status": "completed", "results": {"face_detections": 5}}})
    state = orch.redis_manager.get_state("corr-1")
    state["summary_result"] = {"results": {}}

    await orch.handle_save_result({"payload": {"results": {"text_entries": []}}}, FakeProducer(), state, "corr-1")
    saved = orch.redis_manager.get_state("corr-1")
    assert saved["step"] == "indexing"
    assert saved["face_detection"]["results"] == {"face_detections": 5}

    await orch.handle_index_completed({}, FakeProducer(), saved, "corr-1")
    final = orch.redis_manager.get_state("corr-1")
    assert final["step"] == "complete"
    assert final["face_detection"]["results"] == {"face_detections": 5}


async def test_completion_keeps_a_face_result_that_arrived_while_indexing(orch):
    state = orch.redis_manager.get_state("corr-1")
    state["summary_result"] = {"results": {}}
    await orch.handle_save_result({"payload": {"results": {"text_entries": []}}}, FakeProducer(), state, "corr-1")
    saved = orch.redis_manager.get_state("corr-1")
    assert "face_detection" not in saved

    # the face result lands after the save, while Module 25 is still indexing
    result = _face_result(payload={"status": "success", "results": {"face_detections": 7}})
    await orch.handle_service_result(result, FakeProducer(), "video-face-detection-results")

    await orch.handle_index_completed({}, FakeProducer(), saved, "corr-1")   # handed the pre-face state
    final = orch.redis_manager.get_state("corr-1")
    assert final["step"] == "complete"
    assert final["face_detection"]["results"] == {"face_detections": 7}
