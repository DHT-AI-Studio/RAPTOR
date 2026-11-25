"""docs/grpc-guide-section.md is pasted into the user guide; keep it tied to the code it describes."""
from __future__ import annotations

import re
from pathlib import Path

from config import Settings
from raptor.v1 import raptor_pb2

_MODULE = Path(__file__).parent.parent.parent
_DOC = (_MODULE / "docs" / "grpc-guide-section.md").read_text()


def test_every_rpc_in_the_proto_is_in_the_guide():
    proto = (_MODULE / "proto" / "raptor" / "v1" / "raptor.proto").read_text()
    rpcs = re.findall(r"^\s*rpc\s+(\w+)\(", proto, re.MULTILINE)
    assert rpcs
    missing = [rpc for rpc in rpcs if f"`{rpc}`" not in _DOC]
    assert not missing, f"RPCs missing from the guide section: {missing}"


def test_every_environment_variable_it_names_is_a_real_setting():
    names = set(re.findall(r"\bGRPC_GATEWAY_[A-Z0-9_]+\b", _DOC))
    assert names
    known = {"GRPC_GATEWAY_" + field.upper() for field in Settings.model_fields}
    assert names <= known, f"not settings of the gateway: {sorted(names - known)}"


def test_the_code_examples_compile_and_use_real_message_types():
    blocks = [b for b in re.findall(r"```python\n(.*?)```", _DOC, re.DOTALL) if b.startswith("# example")]
    assert len(blocks) >= 6
    for block in blocks:
        compile(block, block.splitlines()[0], "exec")
        for name in re.findall(r"\bpb\.([A-Za-z_][A-Za-z0-9_]*)", block):
            assert hasattr(raptor_pb2, name), f"pb.{name} does not exist in raptor_pb2"


def test_the_files_it_points_at_exist():
    for relative in re.findall(r"`((?:docs|proxy-test|scripts|client)/[A-Za-z0-9_./-]+)`", _DOC):
        if relative.endswith("/") or "*" in relative:
            continue
        assert (_MODULE / relative).exists(), relative
