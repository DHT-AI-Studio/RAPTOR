"""VIE01-255: claim-before-process + status-report-after-process, for the
video orchestrator (ported from 09-audio-processing's reference implementation).
Video additionally routes some failures through handle_processing_error(),
not just send_error_response() -- both are tested here.

Run from the 11-video-processing/ directory:
    pytest tests/unit/ -v
"""
import pytest
from conftest import make_request_message

pytestmark = pytest.mark.asyncio


class FakeMessage:
    def __init__(self, value):
        self.value = value


class FakeConsumer:
    def __init__(self, messages):
        self._messages = messages
        self.commit_calls = 0

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for m in self._messages:
            yield m

    async def commit(self):
        self.commit_calls += 1


async def test_claim_processing_called_on_valid_request(handler):
    from api_client import ClaimOutcome
    handler.seaweedfs_client.claim_outcome = ClaimOutcome.CLAIMED
    called = []
    handler.handle_video_request = lambda msg, producer: called.append(msg) or _noop()

    msg = make_request_message()
    consumer = FakeConsumer([FakeMessage(msg)])
    await handler.process_requests(consumer, producer=None)

    assert handler.seaweedfs_client.claim_calls == [
        ("document/txt/foo", "v1", "corr-1", "u1", "b1")
    ]
    assert called == [msg]
    assert consumer.commit_calls == 1


async def test_claim_409_acks_and_discards(handler):
    from api_client import ClaimOutcome
    handler.seaweedfs_client.claim_outcome = ClaimOutcome.CONFLICT
    called = []
    handler.handle_video_request = lambda msg, producer: called.append(msg) or _noop()

    consumer = FakeConsumer([FakeMessage(make_request_message())])
    await handler.process_requests(consumer, producer=None)

    assert called == []
    assert consumer.commit_calls == 1


async def test_claim_5xx_does_not_ack(handler):
    from api_client import ClaimOutcome
    handler.seaweedfs_client.claim_outcome = ClaimOutcome.TRANSIENT_ERROR
    called = []
    handler.handle_video_request = lambda msg, producer: called.append(msg) or _noop()

    consumer = FakeConsumer([FakeMessage(make_request_message())])
    await handler.process_requests(consumer, producer=None)

    assert called == []
    assert consumer.commit_calls == 0


async def test_update_asset_status_active_only_after_indexing_completes(handler):
    """17cb591: active is reported off Module 25's real completion, not when the
    index-request is merely enqueued at save time."""
    correlation_id = "corr-2"
    original_message = make_request_message(correlation_id=correlation_id)
    state = {
        "original_message": original_message,
        "summary_result": {}, "video_analysis_result": {}, "audio_analysis_result": {},
        "branch_id": "b1",
    }

    class FakeProducer:
        async def send(self, topic, message):
            pass

    await handler.handle_save_result({"payload": {}}, FakeProducer(), state, correlation_id)
    assert handler.seaweedfs_client.status_calls == []          # only enqueued for Module 25 so far

    await handler.handle_index_completed({}, FakeProducer(), state, correlation_id)
    assert handler.seaweedfs_client.status_calls == [("u1", "b1", "document/txt/foo", "v1", "active")]


async def test_update_asset_status_failed_via_send_error_response(handler):
    correlation_id = "corr-3"
    original_message = make_request_message(correlation_id=correlation_id)

    class FakeProducer:
        async def send(self, topic, message):
            pass

    await handler.send_error_response(
        FakeProducer(), original_message, "boom", "PROCESSING_FAILED", correlation_id,
    )

    assert handler.seaweedfs_client.status_calls == [("u1", "b1", "document/txt/foo", "v1", "failed")]


async def test_update_asset_status_failed_via_handle_processing_error(handler):
    """Video-specific: mid-pipeline errors route through handle_processing_error(),
    a separate helper from send_error_response()."""
    correlation_id = "corr-5"
    original_message = make_request_message(correlation_id=correlation_id)
    state = {"original_message": original_message, "temp_file_path": None}

    class FakeProducer:
        async def send(self, topic, message):
            pass

    await handler.handle_processing_error(FakeProducer(), state, "boom", "PROCESSING_FAILED")

    assert handler.seaweedfs_client.status_calls == [("u1", "b1", "document/txt/foo", "v1", "failed")]


async def test_status_report_failure_does_not_block_pipeline(handler):
    """Module 04 unreachable during status report must not stop the final
    Kafka message from being sent."""
    handler.seaweedfs_client.status_raises = True
    correlation_id = "corr-4"
    original_message = make_request_message(correlation_id=correlation_id)

    sent = []

    class FakeProducer:
        async def send(self, topic, message):
            sent.append((topic, message))

    await handler.send_error_response(
        FakeProducer(), original_message, "boom", "PROCESSING_FAILED", correlation_id,
    )

    assert sent
    assert handler.seaweedfs_client.status_calls


async def _noop():
    pass
