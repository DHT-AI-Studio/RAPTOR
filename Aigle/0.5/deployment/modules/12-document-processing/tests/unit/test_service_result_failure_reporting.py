"""handle_service_result()'s outer except Exception -- any unexpected bug
in the dispatch logic itself (not a downstream service reporting its own
failure) used to only log, leaving the asset stuck in "processing" forever
with no failure ever reported to Module 04. Confirmed live (module 11): a
Redis race left state missing "original_message", and the handler's
KeyError on that field was swallowed this way.

Run from the 12-document-processing/ directory:
    pytest tests/unit/test_service_result_failure_reporting.py -v
"""
import pytest
from conftest import make_request_message

pytestmark = pytest.mark.asyncio


class FakeProducer:
    async def send(self, topic, message):
        pass


async def test_unexpected_exception_reports_failure_when_state_has_original_message(handler):
    """A bug elsewhere in the dispatch logic (anything, not just the Redis
    race) must still flip the asset to "failed" -- not strand it silently."""
    import kafka_handler as kh

    correlation_id = "corr-1"
    original_message = make_request_message(correlation_id=correlation_id)
    state = {"original_message": original_message, "branch_id": "b1"}
    handler.redis_manager.get_state = lambda cid: state

    async def _boom(*a, **kw):
        raise ValueError("boom")
    handler.handle_analysis_result = _boom

    message = {
        "payload": {"status": "success", "results": {}},
        "message_type": "RESPONSE",
        "correlation_id": correlation_id,
    }
    await handler.handle_service_result(message, FakeProducer(), kh.KAFKA_TOPIC_ANALYSIS_RESULT)

    assert handler.seaweedfs_client.status_calls == [("u1", "b1", "document/txt/foo", "v1", "failed")]


async def test_unexpected_exception_does_not_crash_further_when_state_has_no_original_message(handler):
    """The one case handle_processing_error() itself can't recover from
    (it also needs original_message) -- must not raise a second,
    unhandled exception; just log and give up cleanly."""
    import kafka_handler as kh

    correlation_id = "corr-2"
    # Exactly the shape confirmed live via redis-cli after the race --
    # no original_message, no temp_file_path, nothing past step/branch_id.
    state = {"step": "queued", "branch_id": "b1"}
    handler.redis_manager.get_state = lambda cid: state

    async def _boom(*a, **kw):
        raise ValueError("boom")
    handler.handle_analysis_result = _boom

    message = {
        "payload": {"status": "success", "results": {}},
        "message_type": "RESPONSE",
        "correlation_id": correlation_id,
    }
    await handler.handle_service_result(message, FakeProducer(), kh.KAFKA_TOPIC_ANALYSIS_RESULT)

    assert handler.seaweedfs_client.status_calls == []
