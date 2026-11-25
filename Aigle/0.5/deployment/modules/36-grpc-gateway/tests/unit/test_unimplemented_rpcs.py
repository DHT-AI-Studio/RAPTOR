"""The four RPCs that are in the proto but not built must answer UNIMPLEMENTED.

docs/_internal/KNOWN_ISSUES.md listed this as a test gap: deleting one of their `context.abort(UNIMPLEMENTED, ...)`
lines would not have been caught by anything. They are still part of the published contract, so a
client calling one must get a clear UNIMPLEMENTED, not an empty success or an INTERNAL error.
"""
from __future__ import annotations

import grpc
import pytest
from conftest import make_token
from raptor.v1 import raptor_pb2

AUTH = (("authorization", f"Bearer {make_token()}"),)

# (rpc name, request, is a server-streaming call)
_UNBUILT = [
    ("GetAsset", raptor_pb2.GetAssetRequest(asset_path="a"), False),
    ("Chat", raptor_pb2.ChatRequest(message="hi"), True),
    ("GetJobStatus", raptor_pb2.GetJobStatusRequest(correlation_id="j", media_type="video"), False),
    ("StreamVideo", raptor_pb2.StreamVideoRequest(asset_path="a"), True),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("name", "request_message", "streaming"), _UNBUILT, ids=[u[0] for u in _UNBUILT])
async def test_an_unbuilt_rpc_answers_unimplemented(raptor_stub, name, request_message, streaming):
    call = getattr(raptor_stub, name)(request_message, metadata=AUTH)
    with pytest.raises(grpc.aio.AioRpcError) as caught:
        if streaming:
            async for _ in call:
                pass
        else:
            await call
    assert caught.value.code() == grpc.StatusCode.UNIMPLEMENTED
    assert name in caught.value.details() or "not implemented" in caught.value.details().lower()


def test_the_list_covers_every_rpc_the_servicer_aborts_with_unimplemented():
    """If another RPC is added as a placeholder, it must be added to _UNBUILT too."""
    import inspect

    from server import RaptorServiceServicer

    placeholders = {
        name
        for name, member in inspect.getmembers(RaptorServiceServicer, inspect.isfunction)  # coroutines and generators
        if "UNIMPLEMENTED" in inspect.getsource(member) and "not implemented in this phase" in inspect.getsource(member)
    }
    assert placeholders == {u[0] for u in _UNBUILT}
