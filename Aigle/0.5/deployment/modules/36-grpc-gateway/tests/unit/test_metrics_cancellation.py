"""A server-streaming call that a client cancels must be counted as a cancellation.

Found on a deployed gateway: 30 cancelled downloads left grpc_server_stream_cancellations_total
without a single sample and were counted as OK. Against a real client, a cancel often arrives
while grpc.aio is blocked writing the previous response, and then it does not cancel the task:
it closes the handler's async generator at its `yield` (GeneratorExit). The existing end-to-end
test cancels while the handler is awaiting inside the generator (CancelledError), so only that
shape was ever exercised.
"""
from __future__ import annotations

import asyncio

import grpc
import pytest
from interceptors.metrics import (
    ACTIVE_STREAMS_GAUGE,
    REQUEST_COUNTER,
    STREAM_CANCELLATIONS_COUNTER,
    MetricsInterceptor,
)
from raptor.v1 import raptor_pb2

METHOD = "/test.Service/Stream"


class _Details:
    method = METHOD
    invocation_metadata = ()


class _Context:
    def code(self):
        return None


def _value(counter, **labels):
    return counter.labels(**labels)._value.get()


async def _wrapped_stream(behavior):
    async def continuation(_):
        return grpc.unary_stream_rpc_method_handler(behavior)

    handler = await MetricsInterceptor().intercept_service(continuation, _Details())
    return handler.unary_stream(None, _Context())


def _snapshot():
    return {
        "cancelled": _value(REQUEST_COUNTER, grpc_method=METHOD, grpc_code="CANCELLED"),
        "ok": _value(REQUEST_COUNTER, grpc_method=METHOD, grpc_code="OK"),
        "cancellations": _value(STREAM_CANCELLATIONS_COUNTER, grpc_method=METHOD),
        "active": _value(ACTIVE_STREAMS_GAUGE, grpc_method=METHOD),
    }


@pytest.mark.asyncio
async def test_a_stream_closed_at_its_yield_is_recorded_as_cancelled():
    async def behavior(request, context):
        for i in range(1000):
            yield raptor_pb2.PingResponse(message=str(i))

    before = _snapshot()
    stream = await _wrapped_stream(behavior)
    await stream.__anext__()  # the server is now suspended at a yield, waiting to be asked for the next one
    await stream.aclose()     # grpc.aio closing the generator because the client went away
    after = _snapshot()

    assert after["cancelled"] == before["cancelled"] + 1
    assert after["ok"] == before["ok"]  # NOT counted as a success
    assert after["cancellations"] == before["cancellations"] + 1
    assert after["active"] == before["active"]  # the slot is released either way


@pytest.mark.asyncio
async def test_a_task_cancelled_inside_the_generator_is_still_recorded_as_cancelled():
    async def behavior(request, context):
        yield raptor_pb2.PingResponse(message="first")
        await asyncio.sleep(60)
        yield raptor_pb2.PingResponse(message="never")

    before = _snapshot()
    stream = await _wrapped_stream(behavior)
    await stream.__anext__()
    pending = asyncio.ensure_future(stream.__anext__())
    await asyncio.sleep(0.05)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    after = _snapshot()

    assert after["cancelled"] == before["cancelled"] + 1
    assert after["cancellations"] == before["cancellations"] + 1


@pytest.mark.asyncio
async def test_a_stream_read_to_the_end_is_recorded_as_ok_not_cancelled():
    async def behavior(request, context):
        yield raptor_pb2.PingResponse(message="only")

    before = _snapshot()
    stream = await _wrapped_stream(behavior)
    assert [r.message async for r in stream] == ["only"]
    after = _snapshot()

    assert after["ok"] == before["ok"] + 1
    assert after["cancelled"] == before["cancelled"]
    assert after["cancellations"] == before["cancellations"]
