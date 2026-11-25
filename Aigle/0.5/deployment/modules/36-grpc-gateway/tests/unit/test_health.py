"""Health check coverage: grpc.health.v1.Health is registered alongside
RaptorService (server.py) and exempted from authentication (same as
reflection) so health probes work without credentials -- this is documented
in README.md but was never actually exercised by a test until now.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "generated"))
from grpc_health.v1 import health_pb2, health_pb2_grpc
from server import SERVICE_NAME


@pytest.mark.asyncio
async def test_health_check_overall_serving_without_credentials(grpc_channel):
    stub = health_pb2_grpc.HealthStub(grpc_channel)
    resp = await stub.Check(health_pb2.HealthCheckRequest(service=""))
    assert resp.status == health_pb2.HealthCheckResponse.SERVING


@pytest.mark.asyncio
async def test_health_check_raptor_service_serving_without_credentials(grpc_channel):
    stub = health_pb2_grpc.HealthStub(grpc_channel)
    resp = await stub.Check(health_pb2.HealthCheckRequest(service=SERVICE_NAME))
    assert resp.status == health_pb2.HealthCheckResponse.SERVING


# Health.Watch was never exercised: it failed on a deployed server with
# "'_ServicerContext' object has no attribute 'add_callback'" because the synchronous
# HealthServicer was registered on a grpc.aio server (Check works either way).


@pytest.mark.asyncio
@pytest.mark.parametrize("service", ["", SERVICE_NAME])
async def test_health_watch_first_message_is_serving(grpc_channel, service):
    stub = health_pb2_grpc.HealthStub(grpc_channel)
    call = stub.Watch(health_pb2.HealthCheckRequest(service=service))
    try:
        first = await call.read()
    finally:
        call.cancel()
    assert first.status == health_pb2.HealthCheckResponse.SERVING


@pytest.mark.asyncio
async def test_health_watch_of_an_unknown_service_reports_service_unknown(grpc_channel):
    # The Health protocol: Watch on a name that is not registered streams SERVICE_UNKNOWN
    # (Check on it is NOT_FOUND).
    stub = health_pb2_grpc.HealthStub(grpc_channel)
    call = stub.Watch(health_pb2.HealthCheckRequest(service="no.such.Service"))
    try:
        first = await call.read()
    finally:
        call.cancel()
    assert first.status == health_pb2.HealthCheckResponse.SERVICE_UNKNOWN


@pytest.mark.asyncio
async def test_health_check_of_an_unknown_service_is_not_found(grpc_channel):
    import grpc

    stub = health_pb2_grpc.HealthStub(grpc_channel)
    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await stub.Check(health_pb2.HealthCheckRequest(service="no.such.Service"))
    assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND


@pytest.mark.asyncio
async def test_health_watchers_do_not_consume_the_per_user_stream_cap():
    """A probe has no user and a Watch is long-lived: with a cap of 1, several concurrent
    Watches must all be served and never touch the per-user stream counter."""
    import grpc
    from server import build_server

    class Redis:
        def __init__(self):
            self.touched = []

        async def incr(self, key):
            self.touched.append(key)
            return 1

        async def decr(self, key):
            self.touched.append(key)
            return 0

        async def expire(self, key, ttl):
            pass

    redis = Redis()
    server = build_server(redis_client=redis, max_streams_per_user=1)
    port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    try:
        stub = health_pb2_grpc.HealthStub(channel)
        calls = [stub.Watch(health_pb2.HealthCheckRequest(service="")) for _ in range(3)]
        try:
            firsts = [await c.read() for c in calls]
        finally:
            for c in calls:
                c.cancel()
        assert [f.status for f in firsts] == [health_pb2.HealthCheckResponse.SERVING] * 3
        assert redis.touched == []
    finally:
        await channel.close()
        await server.stop(grace=None)
