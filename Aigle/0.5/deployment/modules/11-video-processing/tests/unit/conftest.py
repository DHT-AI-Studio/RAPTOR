import os
import sys

_MODULE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SERVICE_DIR = os.path.join(_MODULE_ROOT, "worker", "services", "video_orchestrator_service")
sys.path.insert(0, _SERVICE_DIR)

# CM-5 / VIE01-408: kafka_handler.py imports this shared helper from /app (the
# shared media-worker image's root, see 08-media-worker/Dockerfile) in production.
sys.path.append(os.path.join(os.path.dirname(_MODULE_ROOT), "common"))

# redis_manager.py does REDIS_PORT = int(os.getenv("REDIS_PORT")) at import
# time -- crashes on None without this.
os.environ.setdefault("REDIS_PORT", "6379")

import pytest


@pytest.fixture(autouse=True)
def patched_redis(monkeypatch):
    """RedisStateManager.__init__ calls self.client.ping() and re-raises on
    failure -- construct the handler class in tests without needing a live
    Redis."""
    import redis_manager

    class _FakeRedisClient:
        def ping(self):
            return True

        def __getattr__(self, name):
            return lambda *a, **kw: None

    monkeypatch.setattr(redis_manager.redis, "Redis", lambda **kw: _FakeRedisClient())
    yield


class FakeSeaweedFSClient:
    """Records claim_processing()/update_asset_status() calls; returns a
    configurable ClaimOutcome."""

    def __init__(self):
        from api_client import ClaimOutcome
        self.claim_outcome = ClaimOutcome.CLAIMED
        self.claim_calls = []
        self.status_calls = []
        self.status_raises = False
        self.moderation_calls = []

    async def claim_processing(self, asset_path, version_id, correlation_id, user_id, branch_id):
        self.claim_calls.append((asset_path, version_id, correlation_id, user_id, branch_id))
        return self.claim_outcome

    async def update_asset_status(self, user_id, branch_id, asset_path, version_id, status):
        self.status_calls.append((user_id, branch_id, asset_path, version_id, status))
        if self.status_raises:
            raise RuntimeError("Module 04 unreachable")

    async def update_moderation(self, user_id, branch_id, asset_path, version_id, flags):
        self.moderation_calls.append((user_id, branch_id, asset_path, version_id, flags))

    def cleanup_temp_file(self, file_path):
        pass

    async def register_keyframes(self, *a, **kw):
        pass


@pytest.fixture
def handler(patched_redis):
    from kafka_handler import VideoOrchestratorKafkaHandler
    h = VideoOrchestratorKafkaHandler()
    h.seaweedfs_client = FakeSeaweedFSClient()
    return h


def make_request_message(asset_path="document/txt/foo", version_id="v1", correlation_id="corr-1", user_id="u1", branch_id="b1"):
    return {
        "message_id": "msg-1",
        "correlation_id": correlation_id,
        "source_service": "video_orchestrator_service",
        "payload": {
            "user_id": user_id,
            "request_id": "req-1",
            "parameters": {"asset_path": asset_path, "version_id": version_id, "branch_id": branch_id, "primary_filename": "f.mp4"},
            "asset_managemant_download_header": {"X-User-ID": user_id, "X-Branch-ID": branch_id},
        },
    }
