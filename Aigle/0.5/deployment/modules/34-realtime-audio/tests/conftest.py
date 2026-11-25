"""Shared by tests/unit and tests/integration: the RA-6 gRPC stubs.

proto/raptor/audio/v1/audio.proto is compiled into a temporary directory at the start of every
run (put first on sys.path), so a stale generated/ never hides a proto change.  Without
grpcio-tools nothing is generated and the gRPC tests skip themselves (pytest.importorskip).
"""
from __future__ import annotations

import atexit
import importlib.util
import os
import shutil
import sys
import tempfile

_MODULE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _generate_stubs() -> None:
    if importlib.util.find_spec("grpc_tools") is None:
        return
    spec = importlib.util.spec_from_file_location("_rt_gen_proto", os.path.join(_MODULE_DIR, "scripts", "gen_proto.py"))
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    out = tempfile.mkdtemp(prefix="rt-audio-proto-")
    atexit.register(shutil.rmtree, out, True)
    gen.generate(out)
    sys.path.insert(0, out)


_generate_stubs()
